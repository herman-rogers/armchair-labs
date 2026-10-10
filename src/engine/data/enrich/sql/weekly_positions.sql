-- The weekly roster position describes that week. Conflicting same-week
-- classifications stay unresolved rather than being picked by input order.
SELECT gsis_id AS player_id, CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS week,
       CASE WHEN count(DISTINCT canonical) = 1 THEN any_value(canonical) END AS historical_position,
       count(DISTINCT canonical) > 1 AS position_conflict
FROM (
    SELECT gsis_id, season, week,
           CASE upper(position) WHEN 'FB' THEN 'RB' WHEN 'HB' THEN 'RB' ELSE upper(position) END AS canonical
    FROM raw_roster_weekly
    WHERE game_type = 'REG' AND gsis_id IS NOT NULL AND position IS NOT NULL
)
GROUP BY ALL
