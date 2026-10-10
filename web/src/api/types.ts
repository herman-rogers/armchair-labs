export type Position = 'QB' | 'RB' | 'WR' | 'TE'

/** `/api/status`: what is published, with no selected or promoted product anywhere. */
export interface Status {
  tables: {
    available: boolean; catalog_sha256?: string | null; published_at?: string | null
    table_count?: number; observation_cutoff?: { season: number | null; through_week: number | null }; issue?: string
  }
  campaigns: { available: boolean; count: number; campaigns: import('./chairstats').CampaignSummary[]; issue: string | null }
  league: { name: string; team_count: number; board_season: number; draft_season: number; seasons: number[] }
}

/** `/api/tables/catalog`: the pinned DuckDB table catalog. */
export interface DataCatalog {
  catalog_sha256: string
  published_at: string | null
  observation_cutoff: { season: number | null; through_week: number | null }
  tables: { name: string; rows: number; built_at: string; observation_cutoff: { season?: number; through_week?: number }; description: string }[]
}
