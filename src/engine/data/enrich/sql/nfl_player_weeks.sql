-- Completed-season player weeks with repaired positions and league scoring.
-- The current season comes from the weekly capture (current_weeks), never from here.
WITH stats AS (
    SELECT * REPLACE (CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS week),
           CASE upper(position) WHEN 'FB' THEN 'RB' WHEN 'HB' THEN 'RB' ELSE upper(position) END AS raw_position
    FROM raw_stats_player_week
    WHERE season_type = 'REG' AND {history_filter}
),
repaired AS (
    SELECT s.* EXCLUDE (position),
           coalesce(p.historical_position, s.raw_position) AS repaired_position,
           CASE WHEN p.historical_position IS NOT NULL THEN 'weekly_roster' ELSE 'stat_feed_fallback' END AS repaired_source
    FROM stats s
    LEFT JOIN weekly_positions p USING (player_id, season, week)
),
scored AS (
    SELECT r.* EXCLUDE (repaired_position, repaired_source),
           {position_expression} AS position,
           CASE WHEN {override_flag} THEN 'league_override' ELSE repaired_source END AS position_source,
           {component_expression} AS component_pts,
           coalesce(b.bonus_pts, 0.0) AS bonus_pts
    FROM repaired r
    LEFT JOIN touchdown_bonuses b USING (player_id, season, week)
)
SELECT *, component_pts + bonus_pts AS league_points
FROM scored
ORDER BY season, week, player_id
