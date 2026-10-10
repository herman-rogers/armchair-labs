-- Expected credits from the play flags alone, against what landed on a player.
WITH scoring_plays AS (
    SELECT pass_touchdown, rush_touchdown, passer_player_id, receiver_player_id, rusher_player_id,
           {bonus_expression} AS bonus_pts
    FROM raw_pbp
    WHERE touchdown = 1 AND season_type = 'REG'
),
paying AS (SELECT * FROM scoring_plays WHERE bonus_pts > 0)
SELECT count(*) AS scoring_plays,
       coalesce(sum(bonus_pts * (2 * (pass_touchdown = 1)::INT + (rush_touchdown = 1)::INT)), 0) AS expected_points,
       (SELECT coalesce(sum(bonus_pts), 0) FROM touchdown_bonuses) AS credited_points,
       coalesce(sum((pass_touchdown = 1 AND passer_player_id IS NULL)::INT
                  + (pass_touchdown = 1 AND receiver_player_id IS NULL)::INT
                  + (rush_touchdown = 1 AND rusher_player_id IS NULL)::INT), 0) AS dropped_credits
FROM paying
