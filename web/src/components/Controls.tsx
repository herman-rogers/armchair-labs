import type { ReactNode } from 'react'
import { POSITION_FILTERS } from '../positions'

/** `<option>`s for a position filter `<select>`. */
export function PositionOptions({ positions = POSITION_FILTERS }: { positions?: readonly string[] }) {
  return <>{positions.map(position => <option key={position} value={position}>{position}</option>)}</>
}

/**
 * Previous/Next controls for a paginated result set. Pass the page from `useUrlPage`
 * (or any `{ offset, limit, onPage }`); `noun` labels the buttons ("Previous rookies").
 */
export function Pager({ page: { offset, limit, onPage }, total, noun, summary, children }: {
  page: { offset: number; limit: number; onPage: (offset: number) => void }
  total: number
  noun?: string
  /** Replaces the default "1–50 of 312" range. */
  summary?: ReactNode
  /** Extra actions in the same row, such as an export button. */
  children?: ReactNode
}) {
  const label = (direction: string) => noun ? `${direction} ${noun}` : direction
  return <div className="analysis-controls" aria-label="Results pages">
    <span>{summary ?? `${Math.min(offset + 1, total)}–${Math.min(offset + limit, total)} of ${total}`}</span>
    <button type="button" className="button" disabled={offset === 0} onClick={() => onPage(Math.max(0, offset - limit))}>{label('Previous')}</button>
    <button type="button" className="button" disabled={offset + limit >= total} onClick={() => onPage(offset + limit)}>{label('Next')}</button>
    {children}
  </div>
}

/** The error notice for a failed query, or nothing. `label` prefixes the message; children follow it. */
export function QueryError({ query, label, children }: { query: { error: Error | null }; label?: string; children?: ReactNode }) {
  if (!query.error) return null
  return <p className="notice" role="alert">{label && `${label}: `}{query.error.message}{children}</p>
}
