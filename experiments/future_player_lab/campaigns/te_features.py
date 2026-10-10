"""Fixed, outcome-independent receiving/usage feature contract for tight ends.

Keep the common history windows and missingness semantics. Restrict player box,
tracking and play summaries to offensive role; team passing remains context.
No feature is selected using validation outcomes or current-season coverage.
"""

RECEIVING_NGS = {
    "avg_cushion",
    "avg_separation",
    "avg_intended_air_yards",
    "percent_share_of_intended_air_yards",
    "receptions",
    "targets",
    "catch_percentage",
    "yards",
    "rec_touchdowns",
    "avg_yac",
    "avg_expected_yac",
    "avg_yac_above_expectation",
}
BOX_COUNTS = {
    "carries",
    "receptions",
    "targets",
    "racr",
    "target_share",
    "air_yards_share",
    "wopr",
    "fantasy_points",
    "fantasy_points_ppr",
}
PLAY_FIELDS = {
    "observations",
    "yards_gained",
    "air_yards",
    "yards_after_catch",
    "yardline_100",
    "down",
    "ydstogo",
    "shotgun",
    "no_huddle",
    "pass_length",
    "epa",
    "wpa",
    "air_epa",
    "yac_epa",
    "comp_air_epa",
    "comp_yac_epa",
    "cp",
    "cpoe",
    "first_down",
    "first_down_pass",
    "first_down_rush",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "complete_pass",
    "incomplete_pass",
    "receiving_yards",
    "rushing_yards",
    "rush_attempt",
    "pass_attempt",
    "fumble",
    "fumble_lost",
    "tackled_for_loss",
    "out_of_bounds",
    "success",
    "redzone_count",
    "inside5_count",
    "xyac_epa",
    "xyac_mean_yardage",
    "xyac_median_yardage",
    "xyac_success",
    "xyac_fd",
    "participation_defenders_in_box",
    "participation_number_of_pass_rushers",
    "participation_ngs_air_yards",
}
COLLEGE_FIELDS = {
    "rushingAttempts",
    "rushingYards",
    "yardsPerRushAttempt",
    "rushingTouchdowns",
    "longRushing",
    "receptions",
    "receivingYards",
    "yardsPerReception",
    "receivingTouchdowns",
    "longReception",
    "fumbles",
    "fumblesLost",
    "observations",
}


def relevant_features(names):
    """Apply the same explicit mask for every historical and inference year."""
    selected = []
    for name in names:
        parts = name.split("__")
        field = parts[-1]
        if name.startswith("college__"):
            keep = field in COLLEGE_FIELDS
        elif name.startswith(("team_prior__", "team_current__")):
            keep = field in BOX_COUNTS | {
                "attempts",
                "completions",
                "observations",
            } or field.startswith(("passing_", "receiving_", "rushing_"))
        elif len(parts) >= 3 and parts[1] == "box":
            keep = field in BOX_COUNTS or field.startswith(("receiving_", "rushing_"))
        elif len(parts) >= 3 and parts[1] == "ngs":
            keep = field in RECEIVING_NGS
        elif len(parts) >= 3 and parts[1] == "snaps":
            keep = field in {"offense_snaps", "offense_pct"}
        elif len(parts) >= 3 and parts[1].startswith("pbp_"):
            keep = parts[1] in {"pbp_target", "pbp_rush"} and field in PLAY_FIELDS
        else:
            # Age, exposure, history, combine, draft, roster and injury windows.
            keep = True
        if keep:
            selected.append(name)
    return selected
