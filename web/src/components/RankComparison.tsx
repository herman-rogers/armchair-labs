import type { Column } from './DataTable'

type Consensus = { position: string; ecr_overall?: unknown; ecr_position?: unknown; ecr_overall_date?: unknown; ecr_position_date?: unknown }
export function consensusColumns<T extends Consensus>({ showDates = true }: { showDates?: boolean } = {}): Column<T>[] {
  return [
    { key: 'ecr_overall', label: 'ECR · overall', title: 'FantasyPros Expert Consensus Ranking from the dated preseason snapshot. Original decimal consensus values are preserved; lower is better.', initial: 'asc', render: r => <span className="consensus-rank">{typeof r.ecr_overall === 'number' ? r.ecr_overall.toFixed(1) : '—'}{showDates && <small>{String(r.ecr_overall_date ?? 'Not available')}</small>}</span> },
    { key: 'ecr_position', label: 'ECR · position', title: 'FantasyPros positional consensus, rounded to the nearest whole rank and labeled by position (for example, RB8). Sorting uses the original decimal value. Missing, late or mismatched-position data stays unavailable.', initial: 'asc', render: r => <span className="consensus-rank">{typeof r.ecr_position === 'number' && Number.isFinite(r.ecr_position) && r.position ? `${r.position}${Math.round(r.ecr_position)}` : '—'}{showDates && <small>{String(r.ecr_position_date ?? 'Not available')}</small>}</span> },
  ]
}
