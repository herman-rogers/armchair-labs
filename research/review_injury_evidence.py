"""Import specifically reviewed RB announcements and dated 2009 corroboration.

Decisions below are explicit reviews, not a headline classifier. A news statement
of a diagnosis alone is not equivalent to a confirmed full-period absence.
"""

import json
from datetime import datetime

import polars as pl
from build_injury_archive import CACHE, OUT, ensure_open

from patron.data.historical_evidence import EvidenceCapture, article_content

# player, season, known date, first unavailable calendar week, certainty, source,
# verification phrase and date basis. Last week is regular-season calendar end.
REVIEWS = [
    (
        "00-0022984",
        "Ryan Grant",
        2010,
        "2010-09-14",
        2,
        "confirmed",
        "https://www.nfl.com/news/ankle-leg-injuries-will-force-packers-grant-to-miss-season-09000d5d81a87f7f",
        "Ryan Grant is out for the season",
        "publication_and_modification_same_day",
    ),
    (
        "00-0027155",
        "Rashad Jennings",
        2011,
        "2011-09-03",
        1,
        "confirmed",
        "https://www.nfl.com/news/knee-injury-will-keep-jaguars-rb-jennings-out-for-season-09000d5d821f151b",
        "Rashad Jennings on season-ending injured reserve",
        "publication_and_modification_same_day",
    ),
    (
        "00-0036158",
        "J.K. Dobbins",
        2023,
        "2023-09-10",
        2,
        "confirmed",
        "https://www.nfl.com/news/ravens-rb-j-k-dobbins-torn-achilles-texans",
        "will miss the remainder of the 2023 season",
        "later_publication_or_update",
    ),
    (
        "00-0034791",
        "Nick Chubb",
        2023,
        "2023-09-19",
        3,
        "confirmed",
        "https://www.clevelandbrowns.com/news/rb-nick-chubb-officially-out-for-the-remainder-of-the-season",
        "officially out for the remainder of the season",
        "later_publication_or_update",
    ),
    (
        "00-0033699",
        "Austin Ekeler",
        2025,
        "2025-09-12",
        3,
        "uncertain",
        "https://www.nfl.com/news/commanders-rb-austin-ekeler-believed-to-have-torn-achilles-will-undergo-mri",
        "will undergo an MRI to confirm",
        "later_publication_or_update",
    ),
    (
        "00-0033699",
        "Austin Ekeler",
        2025,
        "2025-09-13",
        3,
        "confirmed",
        "https://www.nfl.com/news/nfl-news-roundup-latest-league-updates-from-saturday-sept-13",
        "Ekeler will miss the remainder of the 2025 season",
        "later_publication_or_update",
    ),
    (
        "00-0034844",
        "Saquon Barkley",
        2020,
        "2020-09-22",
        3,
        "confirmed",
        "https://www.giants.com/news/saquon-barkley-mri-confirms-torn-acl-will-undergo-surgery",
        "2020 season ended",
        "reviewed_dated_team_announcement; corroborated_by_NFL_2020-09-21; "
        "later_CMS_modification_retained",
    ),
    (
        "00-0036158",
        "J.K. Dobbins",
        2021,
        "2021-08-30",
        1,
        "confirmed",
        "https://www.baltimoreravens.com/news/j-k-dobbins-leaves-third-preseason-game-with-knee-injury",
        "season-ending torn ACL",
        "reviewed_explicit_2021-08-29_embedded_report; next_day_conservative; "
        "later_CMS_modification_retained",
    ),
    (
        "00-0031687",
        "Raheem Mostert",
        2021,
        "2021-09-15",
        2,
        "confirmed",
        "https://www.nfl.com/news/49ers-rb-raheem-mostert-to-undergo-season-ending-knee-surgery",
        "opted for season-ending surgery",
        "reviewed_player_announcement_dated_2021-09-14; next_day_conservative; "
        "later_CMS_modification_retained",
    ),
    (
        "00-0036414",
        "Cam Akers",
        2021,
        "2021-07-21",
        1,
        "uncertain",
        "https://www.nfl.com/news/rams-rb-cam-akers-suffers-torn-achilles-training",
        "torn Achilles",
        "diagnosis_confirmed_but_full_period_eligibility_not_established; no_hard_zero",
    ),
]


def main():
    ensure_open()
    capture = EvidenceCapture(CACHE)
    decisions = []
    for pid, name, year, known, first, certainty, url, phrase, basis in REVIEWS:
        rec, html = capture.fetch(url)
        content = article_content(html)
        if rec["status_code"] != 200 or phrase not in content["text"]:
            raise ValueError(f"Reviewed statement missing: {url}")
        published = content["published_at"]
        if not published or published[:10] > known:
            raise ValueError(f"Publication occurs after reviewed knowledge date: {name}")
        if (
            basis in {"publication_and_modification_same_day", "later_publication_or_update"}
            and content["modified_at"][:10] > known
        ):
            raise ValueError(f"Later unreviewed revision: {name}")
        decisions.append(
            dict(
                evidence_id=f"archive-{pid}-{year}-{known}",
                absence_id=f"archive-{pid}-{year}-injury",
                player_id=pid,
                player_name=name,
                season=year,
                kind="injury",
                certainty=certainty,
                source_published_on=published[:10],
                source_published_at=published,
                known_on=known,
                verified_on="2026-09-25",
                unavailable_games=[],
                unavailable_weeks=list(range(first, 19 if year >= 2021 else 18))
                if certainty == "confirmed"
                else [],
                source_url=url,
                source_sha256=rec["sha256"],
                source_capture_file=rec["capture_file"],
                source_modified_at=content["modified_at"],
                retrieved_at=rec["retrieved_at"],
                review_basis=basis,
                summary=f"Reviewed {name} {year} announcement; {certainty}; "
                f"regular-season weeks {first} onward only if confirmed.",
            )
        )
    (OUT / "reviewed_additions.json").write_text(json.dumps(decisions, indent=2))
    # Four manually checked 2009 RB entries in the same dated weekly report.
    url = "https://www.nfl.com/news/mcnabb-bryant-among-fantasy-players-who-could-miss-week-2-09000d5d812b8c34"
    rec, html = capture.fetch(url)
    content = article_content(html)
    reports = pl.read_parquet(OUT / "injury_reports.parquet")
    additions = []
    for name, injury, status, phrase in [
        ("James Davis", "Shoulder", "Questionable", "Davis (shoulder) is listed as questionable"),
        ("Jonathan Stewart", "Heel", "Probable", "Stewart (heel) is listed as probable"),
        ("Pierre Thomas", "Knee", "Questionable", "Thomas (knee) is listed as questionable"),
        (
            "LaDainian Tomlinson",
            "Ankle",
            "Out",
            "Tomlinson (ankle) has already been ruled out for Sunday",
        ),
    ]:
        assert phrase in content["text"]
        row = reports.filter(
            (pl.col("season") == 2009)
            & (pl.col("week") == 2)
            & (pl.col("full_name") == name)
            & (pl.col("game_type") == "REG")
        ).row(0, named=True)
        assert (
            row["date_modified"] is None
            and row["report_primary_injury"] == injury
            and row["report_status"] == status
        )
        additions.append(
            {
                **{
                    k: row[k]
                    for k in [
                        "season",
                        "week",
                        "game_type",
                        "gsis_id",
                        "position",
                        "full_name",
                        "team",
                        "report_primary_injury",
                        "report_status",
                    ]
                },
                "record_id": f"corroborated:{row['record_id']}:{rec['sha256'][:12]}",
                "original_record_id": row["record_id"],
                "date_modified": datetime.fromisoformat(
                    max(content["published_at"], content["modified_at"]).replace("Z", "+00:00")
                ),
                "practice_status": None,
                "practice_primary_injury": None,
                "source_url": url,
                "source_sha256": rec["sha256"],
                "source_capture_file": rec["capture_file"],
                "source_published_at": content["published_at"],
                "source_modified_at": content["modified_at"],
                "retrieved_at": rec["retrieved_at"],
                "date_basis": "later_of_article_publication_and_modification",
                "match_basis": "manually_reviewed_player_section_injury_status",
                "evidence_text": phrase,
            }
        )
    pl.DataFrame(additions, infer_schema_length=None).write_parquet(
        OUT / "manual_corroborated_reports.parquet"
    )
    print(f"Reviewed {len(decisions)} announcements; {len(additions)} timestamp corroborations")


if __name__ == "__main__":
    main()
