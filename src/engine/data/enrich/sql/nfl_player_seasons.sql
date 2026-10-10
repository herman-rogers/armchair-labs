-- One row per player-season: counting totals, games observed, and league points.
SELECT player_id, season,
       arg_max(player_display_name, week) AS player_display_name,
       arg_max(team, week) AS team,
       arg_max(position, week) AS position,
       count(*) AS games,
       sum(coalesce(completions, 0)) AS completions,
       sum(coalesce(attempts, 0)) AS attempts,
       sum(coalesce(passing_yards, 0)) AS passing_yards,
       sum(coalesce(passing_tds, 0)) AS passing_tds,
       sum(coalesce(passing_interceptions, 0)) AS passing_interceptions,
       sum(coalesce(carries, 0)) AS carries,
       sum(coalesce(rushing_yards, 0)) AS rushing_yards,
       sum(coalesce(rushing_tds, 0)) AS rushing_tds,
       sum(coalesce(targets, 0)) AS targets,
       sum(coalesce(receptions, 0)) AS receptions,
       sum(coalesce(receiving_yards, 0)) AS receiving_yards,
       sum(coalesce(receiving_tds, 0)) AS receiving_tds,
       sum(coalesce(rushing_fumbles_lost, 0) + coalesce(receiving_fumbles_lost, 0) + coalesce(sack_fumbles_lost, 0)) AS fumbles_lost,
       sum(coalesce(special_teams_tds, 0)) AS special_teams_tds,
       sum(component_pts) AS component_pts,
       sum(bonus_pts) AS bonus_pts,
       sum(league_points) AS total_pts,
       sum(league_points) / count(*) AS ppg,
       sum(coalesce(fantasy_points_ppr, 0)) AS fantasy_points_ppr
FROM nfl_player_weeks
GROUP BY player_id, season
ORDER BY player_id, season
