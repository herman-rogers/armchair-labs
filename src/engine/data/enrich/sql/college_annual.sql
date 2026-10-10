-- One row per college athlete and season; multi-team seasons stay inspectable.
SELECT college_id, season,
       min(player_name) AS player_name,
       min(college_position) AS college_position,
       list_sort(list_distinct(list(college_team) FILTER (WHERE college_team IS NOT NULL))) AS teams,
       count(DISTINCT team_id) AS team_count,
       min(birth_date) AS birth_date,
       bool_and(complete_team_season) AS complete_team_season,
       min(coverage) AS coverage,
       max(season_end_date) AS season_end_date,
       sum(scheduled_games) AS scheduled_games,
       sum(observed_stat_games) AS observed_stat_games,
       sum(passing_yards) AS passing_yards, sum(passing_tds) AS passing_tds, sum(passing_ints) AS passing_ints,
       sum(carries) AS carries, sum(rushing_yards) AS rushing_yards, sum(rushing_tds) AS rushing_tds,
       sum(receptions) AS receptions, sum(receiving_yards) AS receiving_yards, sum(receiving_tds) AS receiving_tds,
       avg(share_receiving_yards) AS share_receiving_yards, avg(share_receptions) AS share_receptions,
       avg(share_rushing_yards) AS share_rushing_yards, avg(share_carries) AS share_carries
FROM college_seasons
GROUP BY college_id, season
ORDER BY college_id, season
