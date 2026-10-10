-- Big-play touchdown bonuses per player-week, reduced from play-by-play.
-- A passing touchdown pays both passer and receiver; a rushing touchdown pays the rusher.
WITH scoring_plays AS (
    SELECT season, week, pass_touchdown, rush_touchdown,
           passer_player_id, receiver_player_id, rusher_player_id,
           {bonus_expression} AS bonus_pts
    FROM raw_pbp
    WHERE touchdown = 1 AND season_type = 'REG'
),
paying AS (SELECT * FROM scoring_plays WHERE bonus_pts > 0),
credits AS (
    SELECT season, week, passer_player_id AS player_id, bonus_pts FROM paying
    WHERE pass_touchdown = 1 AND passer_player_id IS NOT NULL
    UNION ALL
    SELECT season, week, receiver_player_id, bonus_pts FROM paying
    WHERE pass_touchdown = 1 AND receiver_player_id IS NOT NULL
    UNION ALL
    SELECT season, week, rusher_player_id, bonus_pts FROM paying
    WHERE rush_touchdown = 1 AND rusher_player_id IS NOT NULL
)
SELECT CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS week, player_id,
       sum(bonus_pts) AS bonus_pts
FROM credits
GROUP BY ALL
