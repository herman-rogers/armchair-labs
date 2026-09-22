# Market backfill provenance

Two committed static files extend the dated market-price history behind the
backtest. Every row carries its own source URL and match method; this manifest
records the collection policy and the known limitations. Built 2026-08-31.

## `market_ecr_backfill.csv` — FantasyPros ECR, 2011–2019 forecasts

- **Source**: Internet Archive (Wayback Machine) captures of FantasyPros redraft
  cheatsheet pages. For each season and page the chosen capture is the **latest
  crawl on or before August 31**; the crawl timestamp is part of every
  `source_url`, so the vintage is independently verifiable.
- **Page preference**: PPR pages when captured, otherwise standard-scoring pages
  (the league is PPR; QB pages have no scoring variant). 2011 predates
  FantasyPros PPR pages entirely, so 2011 is standard-scoring — recorded per row
  in `scoring`.
- **Derived positional pages**: where no positional page was captured (2011
  WR/TE, 2018 TE, 2019 QB/RB), the position's rank order is derived from the
  chosen overall page's within-position sequence (`match_method` ends
  `+derived_from_overall`, `sd` is null). Downstream scoring uses only ordering
  within position, so this is an honest substitute.
- **Capture dates** are mostly late August; a few pages are July (worst case
  2011 WR/TE via the August 14 overall page). The per-row `scrape_date` is the
  capture date, never the cutoff.
- **Identity**: `id` is the FantasyPros id (real where the page embeds it —
  2018–2019 — or recoverable via the ffverse crosswalk); players who predate the
  ffverse id map carry a synthetic negative id derived from their GSIS number,
  which `market_backfill.extend_crosswalk` also injects into the crosswalk so the
  standard join works. ~99% of rows resolve to a GSIS id; unresolved rows keep
  their text for audit but never join a forecast.
- **Excluded seasons**: 2010 captures exist but preserve only the top ~20 rows
  per page (paginated view) — too shallow for RB24/WR36 top-K scoring, so 2010
  is deliberately absent rather than silently thin. 2007–2009 have no honestly
  dated ECR source at all.
- **Bonus column**: `fp_adp` is FantasyPros' own dated ADP column where the page
  carried one (2011–2012, 2018–2019). It is a second, independent dated price.

The 2020 forecast season needs no backfill file: the DynastyProcess archive's
summer-2020 combined `redraft-offense` page is normalized into canonical page
types by `market_backfill.normalize_offense_pages` (last pre-cutoff snapshot
2020-08-13). 2021+ come from the archive unchanged.

## `market_adp_backfill.csv` — observed preseason ADP, 2008–2026 forecasts

- **MFL (2011–2026, `preferred` rows)**: MyFantasyLeague real-league draft ADP,
  `TYPE=adp&PERIOD=AUG15&IS_MOCK=0&IS_PPR=1&IS_KEEPER=N`. These are actual
  drafts (1,400–8,500 per season) starting after August 15; the window runs to
  the last preseason drafts, so it can include days after August 31 but never
  regular-season information. Joined by `mfl_id` through the ffverse crosswalk
  (no name matching). The all-scoring variant (`scoring=all`) is retained
  alongside.
- **FFC (2008–2026)**: Fantasy Football Calculator mock-draft ADP. Preferred for
  2008–2010 only (standard 2008–2009, PPR 2010 — the formats that exist). The
  API serves each year's final preseason window (typically September 3–9,
  pre-kickoff); several old years report a corrupted `end_date`, recorded as-is
  in `window`. FFC remaps historical teams to current codes, so resolution uses
  name+position against era rosters, not team.
- ADP is drafter behavior — the acquisition price — and is deliberately kept out
  of the ECR baseline; it loads via `market_backfill.load_backfill_adp` for the
  market-disagreement / value-capture analysis.
- **2007**: no honest source (FFC empty, MFL empty, no usable Wayback captures);
  the ADP series starts at 2008.
