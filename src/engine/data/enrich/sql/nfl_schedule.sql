SELECT * REPLACE (CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS week)
FROM raw_schedules ORDER BY season, week, game_id
