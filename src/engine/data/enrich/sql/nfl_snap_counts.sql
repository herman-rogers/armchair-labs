SELECT * REPLACE (
    CAST(season AS INTEGER) AS season,
    CAST(week AS INTEGER) AS week,
    CASE upper(position) WHEN 'FB' THEN 'RB' WHEN 'HB' THEN 'RB' ELSE upper(position) END AS position
)
FROM raw_snap_counts
WHERE {history_filter}
ORDER BY season, week, game_id, pfr_player_id
