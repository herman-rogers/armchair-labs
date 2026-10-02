"""Conservative, inspectable college↔NFL identity resolution. Never fuzzy-name joins."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import polars as pl

from patron.data.college import unique


def normalized(value: str | None) -> str:
    value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", value.lower())


def name_key(value: str | None) -> str:
    value = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv|v)\s*$", "", value or "", flags=re.I)
    return normalized(value)


def school_key(value: str | None) -> str:
    # Explicit spelling equivalences, never a fuzzy institutional match.
    aliases = {
        "southerncalifornia": "usc",
        "centralflorida": "ucf",
        "mississippi": "olemiss",
        "texaschristian": "tcu",
        "louisianastate": "lsu",
        "miamifl": "miami",
    }
    key = normalized(value)
    return aliases.get(key, key)


def build_links(
    college: pl.DataFrame,
    nfl: pl.DataFrame,
    crosswalk: pl.DataFrame,
    overrides: list[dict] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """One source athlete ID may resolve to only one canonical NFL ID.

    Exact provider IDs still require corroborated names and chronology. Otherwise
    accept unique exact name+DOB or exact name+school+entry timing, with no DOB
    contradiction. Competing candidates and many-to-one collisions go to review.
    Curated exceptions require an explicit reason and evidence URL.
    """
    unique(nfl, ["gsis_id"])
    college_people: dict[str, dict] = {}
    people = college.group_by("college_id").agg(
        pl.col("player_name").drop_nulls().unique(),
        pl.col("college_team").drop_nulls().unique(),
        pl.col("birth_date").drop_nulls().unique(),
        pl.col("season").min().alias("first_season"),
        pl.col("season").max().alias("last_season"),
    )
    for group in people.to_dicts():
        cid = str(group["college_id"])
        college_people[cid] = {
            "college_id": cid,
            "names": sorted(group["player_name"]),
            "name_keys": {name_key(v) for v in group["player_name"]},
            "schools": {school_key(v) for v in group["college_team"]},
            "birth_dates": set(group["birth_date"]),
            "first_season": int(group["first_season"]),
            "last_season": int(group["last_season"]),
        }
    extra: dict[str, list[dict]] = defaultdict(list)
    for row in crosswalk.to_dicts():
        if row.get("gsis_id"):
            extra[row["gsis_id"]].append(row)
    by_id: dict[str, set[str]] = defaultdict(set)
    by_name: dict[str, set[str]] = defaultdict(set)
    nfl_people: dict[str, dict] = {}
    for row in nfl.to_dicts():
        pid = row["gsis_id"]
        related = extra.get(pid, [])
        ids = {
            str(int(str(v).split(".")[0]))
            for v in [row.get("espn_id"), *[r.get("espn_id") for r in related]]
            if v is not None and re.fullmatch(r"[1-9][0-9]*(?:\.0+)?", str(v))
        }
        names = {
            name_key(v) for v in [row.get("display_name"), *[r.get("name") for r in related]] if v
        }
        entry = row.get("rookie_season") or row.get("draft_year")
        dob = str(row.get("birth_date") or "")[:10]
        nfl_people[pid] = {
            **row,
            "entry_year": int(entry) if entry else None,
            "dob": dob,
            "name_keys": names,
            "schools": {
                school_key(v)
                for v in [row.get("college_name"), *[r.get("college") for r in related]]
                if v
            },
            "espn_ids": ids,
            "cfbref_ids": sorted({r["cfbref_id"] for r in related if r.get("cfbref_id")}),
        }
        for eid in ids:
            by_id[eid].add(pid)
        for name in names:
            by_name[name].add(pid)

    manual: dict[str, dict] = {}
    for row in overrides or []:
        cid, pid = str(row["college_id"]), row["player_id"]
        if cid in manual or cid not in college_people or pid not in nfl_people:
            raise ValueError("Invalid/duplicate college identity override")
        if not row.get("reason") or not row.get("source_url", "").startswith("https://"):
            raise ValueError("Identity overrides require a reason and HTTPS evidence URL")
        manual[cid] = row
    output = []
    for cid, person in college_people.items():
        candidate_ids = set(by_id.get(cid, set()))
        if not candidate_ids:
            for name in person["name_keys"]:
                candidate_ids.update(by_name.get(name, set()))
        candidates = []
        conflicts = sorted(by_id.get(cid, set())) if len(by_id.get(cid, set())) > 1 else []
        for pid in sorted(candidate_ids):
            pro = nfl_people[pid]
            entry = pro["entry_year"]
            exact_id = pid in by_id.get(cid, set())
            name_agrees = bool(person["name_keys"] & pro["name_keys"])
            chronology = entry is not None and person["last_season"] < entry
            dob_conflict = bool(
                pro["dob"] and person["birth_dates"] and person["birth_dates"] != {pro["dob"]}
            )
            if not name_agrees or not chronology or dob_conflict:
                if exact_id:
                    conflicts.append(pid)
                continue
            if exact_id:
                method = "provider_id_name_chronology"
            elif pro["dob"] and person["birth_dates"] == {pro["dob"]}:
                method = "exact_name_birth_date"
            elif person["schools"] & pro["schools"] and 1 <= entry - person["last_season"] <= 2:
                method = "exact_name_school_entry_window"
            else:
                continue
            candidates.append((pid, method))
        row = {
            "college_id": cid,
            "college_name": person["names"][0] if person["names"] else None,
            "first_college_season": person["first_season"],
            "last_college_season": person["last_season"],
            "player_id": None,
            "nfl_name": None,
            "entry_year": None,
            "status": "unmatched",
            "method": None,
            "evidence": None,
            "candidate_player_ids": sorted(candidate_ids),
        }
        if cid in manual:
            chosen = manual[cid]
            pid, method = chosen["player_id"], "reviewed_override"
            row.update(evidence=json_evidence(chosen))
        elif conflicts or len(candidates) > 1:
            row.update(
                status="review", evidence="Provider identity conflict or multiple candidates"
            )
            output.append(row)
            continue
        elif len(candidates) == 1:
            pid, method = candidates[0]
        else:
            output.append(row)
            continue
        pro = nfl_people[pid]
        row.update(
            player_id=pid,
            nfl_name=pro["display_name"],
            entry_year=pro["entry_year"],
            status="linked",
            method=method,
        )
        output.append(row)
    reverse: dict[str, list[dict]] = defaultdict(list)
    for row in output:
        if row["status"] == "linked":
            reverse[row["player_id"]].append(row)
    for rows in reverse.values():
        if len(rows) > 1 and not all(r["method"] == "reviewed_override" for r in rows):
            for row in rows:
                row.update(status="review", evidence="Multiple college IDs resolve to one NFL ID")
    links = pl.DataFrame(output, infer_schema_length=None)
    unique(links, ["college_id"])
    # The registry carries typed namespaces. Numerical IDs in different providers
    # are never interchangeable merely because their strings happen to match.
    registry = []
    for row in output:
        linked = row["status"] == "linked"
        canonical = f"nfl:{row['player_id']}" if linked else f"college:espn:{row['college_id']}"
        registry.append(
            {
                "canonical_id": canonical,
                "namespace": "espn_college",
                "external_id": row["college_id"],
                "link_status": row["status"],
                "method": row["method"],
            }
        )
    linked_pids = {r["player_id"] for r in output if r["status"] == "linked"}
    # Keep NFL-only identities too: adding a college source later must not require
    # inventing a new canonical identity for an already known professional.
    for pid in sorted(nfl_people):
        pro = nfl_people[pid]
        for namespace, ids in [
            ("gsis", [pid]),
            ("espn_nfl", sorted(pro["espn_ids"])),
            ("cfb_reference", pro["cfbref_ids"]),
        ]:
            for identifier in ids:
                registry.append(
                    {
                        "canonical_id": f"nfl:{pid}",
                        "namespace": namespace,
                        "external_id": identifier,
                        "link_status": "linked" if pid in linked_pids else "nfl_only",
                        "method": "nflverse_crosswalk",
                    }
                )
    identifiers = pl.DataFrame(registry, infer_schema_length=None).unique()
    collisions = (
        identifiers.group_by("namespace", "external_id")
        .agg(pl.col("canonical_id").unique().sort().alias("candidate_canonical_ids"))
        .filter(pl.col("candidate_canonical_ids").list.len() > 1)
    )
    # A bad supplemental alias must never create an ambiguous registry join.
    # Keep the collision in the review ledger and withhold that alias entirely.
    identifiers = identifiers.join(
        collisions.select("namespace", "external_id"), on=["namespace", "external_id"], how="anti"
    )
    unique(identifiers, ["namespace", "external_id"])
    audit = {
        "college_players": len(output),
        "linked_college_ids": sum(r["status"] == "linked" for r in output),
        "linked_nfl_players": len(linked_pids),
        "identifier_conflicts": collisions.to_dicts(),
        "statuses": links.group_by("status").len().to_dicts(),
        "methods": links.filter(pl.col("status") == "linked").group_by("method").len().to_dicts(),
        "unmatched_meaning": "No accepted NFL identity link; NOT evidence of zero NFL production",
        "review_rows": links.filter(pl.col("status") == "review").to_dicts(),
    }
    return links, identifiers, audit


def json_evidence(row: dict) -> str:
    return f"{row['reason']} | {row['source_url']}"
