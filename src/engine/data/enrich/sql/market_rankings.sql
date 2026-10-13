-- FantasyPros expert consensus ranks by scrape date, linked to NFL identities.
-- Only the redraft and weekly pages the league uses; dynasty and IDP pages are left out.
SELECT CAST(r.scrape_date AS DATE) AS scrape_date, r.page_type, r.ecr_type,
       r.id AS fantasypros_id, x.gsis_id AS player_id,
       r.player AS player_name, r.pos AS position, r.team,
       r.ecr, r.sd, r.best, r.worst, r.rank_delta
FROM raw_ff_rankings r
LEFT JOIN (SELECT fantasypros_id, gsis_id FROM raw_ff_playerids
           WHERE fantasypros_id IS NOT NULL AND gsis_id IS NOT NULL
           QUALIFY count(*) OVER (PARTITION BY fantasypros_id) = 1) x
  ON x.fantasypros_id = r.id
WHERE r.page_type IN ('redraft-overall', 'redraft-qb', 'redraft-rb', 'redraft-wr', 'redraft-te',
                      'weekly-offense', 'weekly-qb', 'weekly-rb', 'weekly-wr', 'weekly-te')
  AND r.scrape_date IS NOT NULL AND r.id IS NOT NULL
QUALIFY row_number() OVER (PARTITION BY r.scrape_date, r.page_type, r.id ORDER BY r.ecr) = 1
ORDER BY scrape_date, page_type, ecr
