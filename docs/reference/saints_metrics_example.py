"""
Patron Saints draft/board metrics — the scripts as run during draft prep (Aug 2026).
Source data: nflverse via nflreadpy (pip install nflreadpy polars).
League: Sweaty Plays (ESPN, 10-team) — full PPR, 4pt pass TD, 6pt rush/rec TD,
40-49yd TD bonus +2 / 50+yd +3 (higher bracket only), -2 INT/fumble,
K distance brackets 3/4/5/6, PAT 1, miss -1.

Run order:
  1. build_bonuses()        -> bonuses.parquet        (40+/50+ TD bonus pts per player-season)
  2. build_player_seasons() -> player_seasons.parquet (league-scored season lines + core metrics)
  3. build_board()          -> board_2025.parquet     (VOR, flags, floor — the draft board)
  4. build_kickers()        -> kickers.parquet
  5. build_dst()            -> dst.parquet
"""

import datetime as dt
import nflreadpy as nfl
import polars as pl

SEASONS = [2023, 2024, 2025]
BOARD_SEASON = 2025

# VOR replacement baselines: positional rank of the freely-available player,
# 10-team league (QB12 / RB25 / WR35 / TE12).
VOR_BASELINE_RANK = {"QB": 12, "RB": 25, "WR": 35, "TE": 12}


# ---------------------------------------------------------------- 1. bonuses
def build_bonuses() -> pl.DataFrame:
    """40-49yd TD = +2, 50+ = +3 (higher bracket only).
    Pass TD: bonus to passer AND receiver. Rush TD: to rusher."""
    pbp = nfl.load_pbp(SEASONS)
    tds = pbp.filter(pl.col("touchdown") == 1).select(
        ["season", "yards_gained", "pass_touchdown", "rush_touchdown",
         "passer_player_id", "receiver_player_id", "rusher_player_id"]
    ).with_columns(
        pl.when(pl.col("yards_gained") >= 50).then(3)
          .when(pl.col("yards_gained") >= 40).then(2)
          .otherwise(0).alias("bonus_pts")
    ).filter(pl.col("bonus_pts") > 0)

    parts = []
    for src, flag in [("passer_player_id", "pass_touchdown"),
                      ("receiver_player_id", "pass_touchdown"),
                      ("rusher_player_id", "rush_touchdown")]:
        parts.append(
            tds.filter(pl.col(flag) == 1)
               .group_by(["season", src])
               .agg(pl.col("bonus_pts").sum())
               .rename({src: "player_id"})
        )
    bonuses = pl.concat(parts).group_by(["season", "player_id"]).agg(
        pl.col("bonus_pts").sum())
    bonuses.write_parquet("bonuses.parquet")
    return bonuses


# -------------------------------------------------------- 2. player seasons
def build_player_seasons() -> pl.DataFrame:
    """Season aggregates under league scoring.
    Note: nflverse fantasy_points_ppr == this league's base scoring exactly
    (full PPR, 4pt pass TD, 1/25 pass yd, -2 turnovers, 2pt conv = 2);
    league_pts = fantasy_points_ppr + big-play bonuses."""
    stats = nfl.load_player_stats(SEASONS).filter(pl.col("season_type") == "REG")
    bonuses = pl.read_parquet("bonuses.parquet")

    skill = stats.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
    agg = skill.group_by(["player_id", "player_display_name", "position", "season"]).agg([
        pl.col("team").last().alias("team"),
        pl.len().alias("games"),
        pl.col("fantasy_points_ppr").sum().alias("base_pts"),
        pl.col("targets").sum(), pl.col("receptions").sum(),
        pl.col("receiving_yards").sum(), pl.col("receiving_tds").sum(),
        pl.col("carries").sum(), pl.col("rushing_yards").sum(), pl.col("rushing_tds").sum(),
        pl.col("passing_yards").sum(), pl.col("passing_tds").sum(),
        pl.col("passing_interceptions").sum(),
        pl.col("target_share").mean().alias("tgt_share"),
        pl.col("air_yards_share").mean().alias("ay_share"),
        pl.col("wopr").mean().alias("wopr"),
    ])

    df = (agg.join(bonuses, on=["player_id", "season"], how="left")
             .with_columns(pl.col("bonus_pts").fill_null(0))
             .with_columns((pl.col("base_pts") + pl.col("bonus_pts")).alias("league_pts"))
             .with_columns([
                 (pl.col("league_pts") / pl.col("games")).alias("ppg"),
                 # Weighted opportunity: a PPR target ~ 2.2x a carry
                 (pl.col("carries") + 2.2 * pl.col("targets")).alias("wtd_opp"),
                 (pl.col("rushing_tds") + pl.col("receiving_tds")).alias("total_tds"),
             ]))

    # TD-over-expectation: actual TDs minus touches * position-average TD rate
    pos_rates = df.group_by("position").agg(
        (pl.col("total_tds").sum() /
         (pl.col("carries").sum() + pl.col("targets").sum())).alias("pos_td_rate"))
    df = (df.join(pos_rates, on="position")
            .with_columns(((pl.col("carries") + pl.col("targets"))
                           * pl.col("pos_td_rate")).alias("exp_tds"))
            .with_columns((pl.col("total_tds") - pl.col("exp_tds")).alias("td_over_exp")))

    df.write_parquet("player_seasons.parquet")
    return df


# ---------------------------------------------------------------- 3. board
def build_board() -> pl.DataFrame:
    """Draft board for BOARD_SEASON: VOR, age, weekly floor/stddev, flags."""
    df = pl.read_parquet("player_seasons.parquet")
    stats = nfl.load_player_stats([BOARD_SEASON]).filter(pl.col("season_type") == "REG")

    # Age as of Sept 1 of the upcoming season
    rosters = (nfl.load_rosters([BOARD_SEASON])
                 .select(["gsis_id", "birth_date"]).unique(subset=["gsis_id"]))
    df = (df.join(rosters.rename({"gsis_id": "player_id"}), on="player_id", how="left")
            .with_columns(((dt.date(BOARD_SEASON + 1, 9, 1) - pl.col("birth_date"))
                           .dt.total_days() / 365.25).round(1).alias("age_next")))

    # Weekly consistency: 25th-pct floor and stddev of weekly points
    wk = (stats.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
              .group_by("player_id").agg([
                  pl.col("fantasy_points_ppr").std().round(1).alias("wk_stddev"),
                  pl.col("fantasy_points_ppr").quantile(0.25).round(1).alias("wk_floor")]))

    s = df.filter(pl.col("season") == BOARD_SEASON).join(wk, on="player_id", how="left")

    # VOR baselines from the board season's PPG ranks
    baselines = {}
    for pos, rank in VOR_BASELINE_RANK.items():
        pool = (s.filter((pl.col("position") == pos) & (pl.col("games") >= 8))
                  .sort("ppg", descending=True))
        baselines[pos] = pool["ppg"][rank - 1] if pool.height >= rank else pool["ppg"][-1]

    s = (s.with_columns(pl.col("position")
             .replace_strict(baselines, return_dtype=pl.Float64).alias("repl_ppg"))
          .with_columns((pl.col("ppg") - pl.col("repl_ppg")).round(2).alias("vor_ppg"))
          .with_columns([
              (pl.col("td_over_exp") >= 4).alias("td_regress_down"),         # SELL
              ((pl.col("td_over_exp") <= -2.5) &
               (pl.col("wtd_opp") >= 150)).alias("td_regress_up"),           # BUY
              ((pl.col("position") == "RB") &
               (pl.col("age_next") >= 27.5)).alias("rb_age_cliff"),
          ]))

    s.write_parquet("board_2025.parquet")
    return s


# --------------------------------------------------------------- 4. kickers
def build_kickers() -> pl.DataFrame:
    """League brackets: FG 0-39=3, 40-49=4, 50-59=5, 60+=6, PAT=1, miss=-1."""
    stats = nfl.load_player_stats([BOARD_SEASON]).filter(pl.col("season_type") == "REG")
    k = (stats.filter(pl.col("position") == "K")
             .group_by(["player_id", "player_display_name"]).agg([
                 pl.col("team").last(), pl.len().alias("games"),
                 ((pl.col("fg_made_0_19") + pl.col("fg_made_20_29")
                   + pl.col("fg_made_30_39")).sum() * 3
                  + pl.col("fg_made_40_49").sum() * 4
                  + pl.col("fg_made_50_59").sum() * 5
                  + pl.col("fg_made_60_").sum() * 6
                  + pl.col("pat_made").sum()
                  - pl.col("fg_missed").sum()).alias("league_pts"),
                 pl.col("fg_made_50_59").sum().alias("fg50s"),
                 pl.col("fg_made_60_").sum().alias("fg60s")])
             .with_columns((pl.col("league_pts") / pl.col("games")).round(2).alias("ppg"))
             .sort("league_pts", descending=True))
    k.write_parquet("kickers.parquet")
    return k


# ------------------------------------------------------------------ 5. DST
def build_dst() -> pl.DataFrame:
    """Directional proxy only: sacks + 2*takeaways + 6*defTD, plus avg points
    allowed from schedules. League PA/YA brackets not modeled per-week."""
    stats = nfl.load_player_stats([BOARD_SEASON]).filter(pl.col("season_type") == "REG")
    sched = nfl.load_schedules([BOARD_SEASON]).filter(pl.col("game_type") == "REG")

    pa = pl.concat([
        sched.select(pl.col("home_team").alias("team"),
                     pl.col("away_score").alias("pts_allowed")),
        sched.select(pl.col("away_team").alias("team"),
                     pl.col("home_score").alias("pts_allowed")),
    ]).group_by("team").agg(pl.col("pts_allowed").mean().round(1).alias("avg_pts_allowed"))

    dst = (stats.group_by("team").agg([
               pl.col("def_sacks").sum().alias("sacks"),
               (pl.col("def_interceptions").sum()
                + pl.col("fumble_recovery_opp").sum()).alias("takeaways"),
               pl.col("def_tds").sum().alias("def_tds")])
           .join(pa, on="team")
           .with_columns((pl.col("sacks") + 2 * pl.col("takeaways")
                          + 6 * pl.col("def_tds")).alias("proxy_pts"))
           .sort("proxy_pts", descending=True))
    dst.write_parquet("dst.parquet")
    return dst


if __name__ == "__main__":
    build_bonuses()
    build_player_seasons()
    board = build_board()
    build_kickers()
    build_dst()
    # Top of the overall board, as sanity check (should lead with CMC/Nacua tier)
    top = (board.filter(pl.col("games") >= 6)
                .sort("vor_ppg", descending=True)
                .select(["player_display_name", "position", "team",
                         "ppg", "vor_ppg", "td_over_exp", "wk_floor"]))
    with pl.Config(tbl_rows=25):
        print(top.head(25))
