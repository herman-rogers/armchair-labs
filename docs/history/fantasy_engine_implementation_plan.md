# Patron Saints Analytics Engine — Implementation Plan

> Historical implementation record. Release IDs, results, commands, and interface descriptions reflect the recorded change; use the [operations guides](../operations/README.md) to run the system today.

*A continuously-running system that pulls the Sweaty Plays league state from ESPN, joins it against NFL production data, and re-ranks everything under the league's exact scoring — the draft-night pipeline, pointed at the season.*

---

## 1. Goal

Reproduce and automate what was done manually during draft prep: compute league-exact fantasy value for every NFL player from raw data, layer situational metrics and flags on top, and rank. Extend it with live league awareness: my roster, every opponent's roster, the free-agent pool, matchups, and FAAB history — refreshed on a short cycle (~10 minutes for league state) so the wire is always ranked and tripwires fire before the rest of the league notices.

The end state is a dashboard-and-alerts loop: every player broken down by the metrics below, tagged with roster status (mine / opponent X / free agent), with diffs highlighted when something changes.

## 2. Data sources

**nflverse, via the `nflreadpy` Python package (PyPI).** The open-source ecosystem all serious NFL analytics sits on; CC-BY licensed. Note the older `nfl_data_py` package is deprecated — tutorials referencing it should be translated. Datasets used in the draft build, all via `load_*` functions returning Polars frames:

- `load_player_stats(seasons)` — weekly per-player stat lines (passing/rushing/receiving components, targets, `target_share`, `air_yards_share`, `wopr`, EPA columns, kicker FG distance brackets, team defensive columns, and a precomputed `fantasy_points_ppr`). This is the workhorse table.
- `load_team_stats(seasons)` — official team attempts, sacks/dropbacks, and carries.
- `load_pbp(seasons)` — full play-by-play (~370 columns, back to 1999). Reduced immediately for touchdown distance, red-zone/end-zone targets, goal-line carries, and conversion rates.
- `load_participation(seasons)` — offensive players on each play, used for the explicitly labeled route-opportunity proxy and targets per route opportunity.
- `load_rosters(season)` — birth dates (age computation), team assignments, `gsis_id` join keys.
- `load_depth_charts(season)` — latest published NFL team, position, depth rank, and snapshot date.
- `load_injuries(seasons)` — official practice/injury-report history for expected-games modeling.
- `load_schedules(season)` — game results for points-allowed (DST), bye weeks, and matchup lookups.

**ESPN Fantasy, via the `espn-api` package (PyPI, cwendt94).** ESPN's fantasy API is unofficial and undocumented but stable and community-maintained; the raw v3 endpoints live under `lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/...` if the package ever lags an endpoint change. Provides: league settings, all rosters, free agents, box scores, live/projected points, draft results, and the transaction log (FAAB bid history — market intelligence). Private-league auth requires two cookies copied from a logged-in browser session (`SWID`, `espn_s2`) passed to the `League` constructor. Treat these as secrets; they expire rarely but do expire.

**Web news/injury layer (manual or semi-automated).** The draft process demonstrated that the model's blind spots are trades, coaching changes, and injuries — the DJ Moore/Kittle/Garrett category. Automating this fully is scope creep; the plan is a lightweight manual-override file (see §5) plus ESPN's own injury status field (`Q/O/IR` per player), which comes free with the roster pull.

## 3. Scoring model (the foundation everything sits on)

League-exact points, computed from stat components rather than trusted from any site's projection. Settings encoded once as a config dict:

Full PPR (1/reception) · 0.1/rush-rec yard · 0.04/pass yard (1 per 25) · 4 pass TD · 6 rush/rec TD · −2 INT and fumble lost · 2-pt conversions = 2 · big-play TD bonuses +2 (40–49 yd) and +3 (50+, higher bracket only) applied to the passer AND receiver on passing TDs, rusher on rushing TDs · kickers by distance bracket: 3 (0–39), 4 (40–49), 5 (50–59), 6 (60+), PAT 1, miss −1 · DST per league sheet (sacks 1, INT/FR 2, TDs 6, points-allowed and yards-allowed brackets).

Implementation note from the draft build: nflverse's `fantasy_points_ppr` matches this league's base scoring exactly (same PPR, 4-pt pass TD, turnover values), so base points can be taken from that column and only the bonus layer computed from play-by-play. Keeping an explicit component-based scorer as well is worth it for auditability and in case a league setting ever changes.

DST is the one place the draft build approximated: the proxy (sacks + 2×takeaways + 6×defTD, plus average points allowed from schedules) ranked units directionally without modeling the PA/YA brackets per-week. A faithful per-week DST scorer from play-by-play is listed as Phase 4 polish; the proxy was adequate for a position drafted last.

## 4. Metrics: the goal, and where the catalog lives

> The metric catalog is no longer duplicated here. The single source of truth for
> which metrics exist is `src/patron/config/metric_report.yaml` (`metrics:` block);
> [`metrics.md`](../reference/metrics.md) is authoritative for metric behavior and carries the
> graveyard of everything the backtest disproved.

**The goal** (stated fully at the top of `metrics.md`): rank the top of the draft
board correctly on next-season **season points = per-active-game PPG × games played**,
measured as hit rate @K against the naive baselines and the FantasyPros market.
Availability and role security are the demonstrated repeatable edge; team-context
narratives and touchdown-regression stories were the demonstrated noise.

A historical note this document owes the reader: the original draft-prep design
(preserved in earlier revisions of this section) called TD-over-expectation "the
single most exploitable signal the model produced." The 22-fold backtest disproved
that — TDOE carried no repeatable next-season signal, and it was removed along with
weighted opportunity, the WOPR blend, and the team-context scalar
(`docs/reference/v2_metrics_review.md` §4–§6a). The surviving core is deliberately plain:
league-exact PPG, VOR, target/carry shares of official team volume, floor/volatility
as descriptors, the RB age curve, depth-chart role, and injury-based expected games.

**Kicker bracket profile** and the **DST proxy score** (per §3) remain positional
sideshows outside the fitted model. **Live-state overlays from ESPN** (ownership,
lineup placement, transactions, injury display) tag the board but never feed the
football projection; nflverse supplies production, usage, depth charts, and
injury-history modeling.

## 5. Architecture

Six small components, one data directory, no heroics:

**Data layer.** Pulls nflverse tables and caches them locally as Parquet keyed by season+week. nflverse data is effectively static between updates, so this layer refreshes on a slow clock and everything downstream reads from cache.

**Scoring engine.** The config-driven league scorer of §3. Input: weekly stat rows (+ TD-distance table). Output: league points per player-week.

**Metrics engine.** Aggregates player-weeks into the §4 catalog per player, per season — and once 2026 games exist, per rolling window (last 4 weeks vs season, for trend detection).

**Board builder.** Sorts by VOR into an overall board plus positional boards, applies flags, merges the manual-override file — a small YAML of situational adjustments with reasons ("kittle: -25, Achilles Jan-26"; "dj-moore: +10, BUF target tree"), which formalizes the live-info overrides that kept beating the raw model on draft night.

**ESPN sync.** Pulls league state via espn-api: all 10 rosters, free agents, matchups, injury tags, projections, transaction log. Tags every player in the board with owner status. Derives the interesting diffs: adds/drops since last poll, FAAB spends, lineup changes, and opponent-roster weaknesses (the Dan-TE-room scan).

**Reporter.** Emits three artifacts per cycle: (1) the ranked wire — free agents above replacement, flags attached; (2) my-roster health — Q tags, byes ahead, floor/ceiling per starter vs this week's opponent; (3) an alerts feed for tripwire events. Output as JSON + a rendered Markdown/HTML page; JSON makes it trivially scrapeable into the homelab stack (Grafana panel, Home Assistant card, or just an iPad bookmark).

**Alert tripwires worth wiring from day one:** any player on my roster gains an injury tag; a top-15-VOR free agent appears (someone dropped real value); an opponent drops a player my board ranks above my worst bench player; snap/target thresholds on named watches (Hunter offensive snaps ≥60%, Sutton target share vs Waddle, Dart designed runs, CMC workload); and any FAAB transaction clearing above $15 (market repricing).

## 6. Scheduling and cadence

The 10-minute ambition is right for *league* state, wrong for *NFL* state — the two sources update on different clocks, so poll them on different clocks:

- **ESPN sync: every 10 minutes** during waking hours (rosters, FA pool, injury tags, transactions move intraday; this is where 10-minute freshness pays). During live game windows it also carries live scoring if wanted.
- **nflverse refresh: nightly** (stats and play-by-play land after games and finalize overnight; polling it faster is wasted motion), plus an extra pull Monday/Tuesday morning when the week's data completes and the weekly re-rank matters most for waivers (bids process at 11 AM ET).
- **Full board rebuild:** after any nflverse refresh, and after any ESPN diff that changes player availability. Cheap enough (seconds on cached Parquet) to run every cycle if simpler.
- **Politeness:** both sources are free community resources — cache aggressively, back off on errors, and never hammer ESPN faster than the 10-minute tick; it's an unofficial API and the correct posture is a quiet, respectful houseguest.

Runtime shape: a single long-running process with two timers (or two cron/systemd-timer entries sharing the data directory — the homelab-native choice). State between runs lives entirely in the Parquet cache + last-poll snapshot for diffing.

## 7. Build phases

**Phase 1 — Reproduce the draft pipeline (static).** Scoring engine + metrics engine + board builder against 2023–25 data; verify outputs match the draft boards (CMC #2, McBride #9, Stafford #33, the flags). This is the correctness baseline and it's mostly transcription of what already ran.

**Phase 2 — League awareness.** ESPN sync with cookie auth; owner-tag the board; first ranked-wire report. Manual runs. Deliverable: the §5 reporter artifacts on demand.

**Phase 3 — The loop.** Scheduler, diffing, alert tripwires, JSON/dashboard output. This is the "every 10 minutes" state. Deliverable: it notices a drop/injury before I do.

**Phase 4 — In-season intelligence.** 2026 weekly data blended in with recency weighting (early-season: shrink 2026 small samples toward 2025 priors; by ~week 6 let current season dominate); rolling 4-week trend columns; opportunity-delta detection (target/carry share week-over-week — the volume-inheritor detector that finds the Kaelon Blacks *before* the injury news); proper per-week DST scoring; cumulative-touch workload column for the age model; optional FAAB bid-history model of league-mates' price behavior.

## 8. Known gotchas

**Name joining is the tax.** nflverse keys on `gsis_id`; ESPN uses its own player IDs and display names ("D.J. Moore" vs "DJ Moore", "Sr."/"Jr." suffixes, Marquise/Hollywood problems). Plan a normalization function plus a small manual alias map, and log unmatched names loudly — silent join failures are how a ranked wire quietly omits the one player that matters. (nflverse publishes an ID-crosswalk table that covers most of it.)

**Rookies and movers.** The next-generation report now scores rookies through a
separate walk-forward NFL-draft-capital model; it never pretends they have prior NFL
production. Age/combine/college context remains measurable but was rejected from the
core after losing to draft capital alone. The connected production board still treats
that challenger as shadow-only and keeps its ESPN fallback until the challenger wins
the stated gates. Team changes (the Moore/Doubs/Waddle class)
make prior-season situational stats stale even when the player row looks healthy; the
override file is the bridge until current-season usage data takes over.

**ESPN endpoint drift.** The unofficial API changes base URLs every couple of years; the espn-api package tracks it, so pin the version, watch its repo when something 401s, and keep the raw-endpoint fallback in mind.

**Cookie custody.** `espn_s2`/`SWID` grant full league access — homelab secret handling, not in the repo.

**Early-season statistics.** Weeks 1–3 metrics on 2026 data are noise; the shrinkage-to-prior design in Phase 4 exists precisely so the system doesn't demand dropping Davante Adams because of one quiet Thursday.

---

*Sources for everything above: nflverse via nflreadpy (play-by-play, weekly stats, rosters, schedules; CC-BY), ESPN Fantasy v3 unofficial API via espn-api (league state; private-league cookie auth). All metrics computed under Sweaty Plays scoring as encoded in §3.*
