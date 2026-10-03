import { ProfileRouteLink } from './PlayerLink'
import { Fragment, useId, useState } from 'react'
import type { Key, ReactNode } from 'react'
import { useUrlParams } from '../navigation'

export type Direction = 'asc' | 'desc'

export interface Column<Row> {
  /** Row field the column sorts on; a virtual key is fine when `value` supplies the sort value. */
  key: Extract<keyof Row, string> | (string & {})
  label: string
  /** Hover documentation. Every column must explain itself. */
  title: string
  align?: 'left'
  /** Default sort direction when this column is first clicked. */
  initial?: Direction
  /** Sort value for a row; defaults to the field named by `key`. */
  value?: (row: Row) => unknown
  render?: (row: Row) => ReactNode
}

const cellValue = <Row,>(row: Row, column: Column<Row>) =>
  column.value ? column.value(row) : (row as Record<string, unknown>)[column.key]

type Sort = { key: string; direction: Direction }
/** Sort token, as stored in the URL: `points` ascending, `-points` descending. */
const sortToken = ({ key, direction }: Sort) => direction === 'desc' ? `-${key}` : key
const parseSort = (token: string | null): Sort | null =>
  token ? { key: token.replace(/^-/, ''), direction: token.startsWith('-') ? 'desc' : 'asc' } : null

/**
 * The one sortable data table. Column sets differ per screen; the mechanics —
 * click-to-sort headers, null-last ordering, hover titles — do not.
 *
 * A page's primary table passes `sortParam` so its order lives in the URL (see
 * `navigation.ts`); secondary and nested tables sort locally. The expanded row is
 * always local, ephemeral state.
 */
export function DataTable<Row>({
  rows,
  columns,
  defaultSort,
  rowKey,
  rowClass,
  emptyMessage,
  renderDetails,
  profileHref,
  rowLabel,
  pagination,
  sortParam,
  serverSide = false,
  rowNumbers = false,
}: {
  rows: Row[]
  /** API has already sorted and paginated the complete filtered result. */
  serverSide?: boolean
  /** Leading "#" column: each row's position in the current order, counted across pages. */
  rowNumbers?: boolean
  /** `param` names the URL page param (from `useUrlPage`); a URL sort change clears it in the same navigation. */
  pagination?: { offset: number; limit: number; onPage: (offset: number) => void; param?: string }
  /** Search param that stores the sort, e.g. `sort`. Omit for local sort state. */
  sortParam?: string
  columns: Column<Row>[]
  defaultSort: string
  rowKey: (row: Row) => Key
  rowClass?: (row: Row) => string | undefined
  /** When set, an empty row list renders as a notice instead of a headers-only table. */
  emptyMessage?: string
  profileHref?: (row: Row) => string | undefined
  renderDetails?: (row: Row) => ReactNode
  rowLabel?: (row: Row) => string
}) {
  const detailsId = useId()
  const [expanded, setExpanded] = useState<Key | null>(null)
  const defaults: Sort = { key: defaultSort, direction: columns.find((column) => column.key === defaultSort)?.initial ?? 'desc' }
  const basis = sortToken(defaults)
  const [local, setLocal] = useState<{ sort: Sort; basis: string } | null>(null)
  const [params, updateParams] = useUrlParams()

  // The table stays mounted when the metric tab changes, so a local sort only applies
  // under the default it was chosen with: otherwise V2 could keep sorting on V1's
  // adj_vor, making two genuinely different boards appear identical. A sort naming a
  // column this table doesn't have (an old link, another view's column) also falls back
  // to the default rather than leaving the rows unsorted.
  const requested = sortParam ? parseSort(params.get(sortParam)) : local?.basis === basis ? local.sort : null
  const { key: sortKey, direction } = requested && columns.some((column) => column.key === requested.key) ? requested : defaults

  const sortColumn = columns.find((column) => column.key === sortKey)
  // Not memoized: callers build `columns` inline, so the sort column changes every render.
  const sorted = serverSide || !sortColumn ? rows : [...rows].sort((a, b) => {
    const left = cellValue(a, sortColumn)
    const right = cellValue(b, sortColumn)
    // Nulls sort last in both directions: an unknown value is not a small one, and
    // floating it to the top would misrepresent the ranking.
    if (left == null && right == null) return 0
    if (left == null) return 1
    if (right == null) return -1
    const comparison =
      typeof left === 'number' && typeof right === 'number'
        ? left - right
        : String(left).localeCompare(String(right))
    return direction === 'asc' ? comparison : -comparison
  })

  if (!rows.length && emptyMessage) {
    return <div className="notice">{emptyMessage}</div>
  }

  const onSort = (column: Column<Row>) => {
    const next: Sort = { key: column.key, direction: column.key === sortKey ? (direction === 'asc' ? 'desc' : 'asc') : column.initial ?? 'desc' }
    if (sortParam) {
      // Sorting returns to the first page, in the same navigation when the page is in the URL too.
      updateParams({ [sortParam]: sortToken(next) === basis ? null : sortToken(next), ...(pagination?.param ? { [pagination.param]: null } : {}) })
      if (!pagination?.param) pagination?.onPage(0)
    } else {
      setLocal({ sort: next, basis })
      pagination?.onPage(0)
    }
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {rowNumbers && <th className="row-number" title="Position in the current sort and filters, counted across pages.">#</th>}
            {columns.map((column) => (
              <th
                key={column.key}
                className={column.align === 'left' ? 'left' : undefined}
                title={column.title}
                aria-sort={sortKey === column.key ? (direction === 'asc' ? 'ascending' : 'descending') : 'none'}
              >
                <button type="button" className="table-sort" onClick={() => onSort(column)}>
                {column.label}
                {sortKey === column.key && (
                  <span className="dir">{direction === 'asc' ? '↑' : '↓'}</span>
                )}
                </button>
              </th>
            ))}
            {renderDetails && <th className="left">Details</th>}
          </tr>
        </thead>
        <tbody>
          {(pagination && !serverSide ? sorted.slice(pagination.offset, pagination.offset + pagination.limit) : sorted).map((row, index) => {
            const key = rowKey(row)
            const href = profileHref?.(row)
            const open = expanded === key
            const id = `${detailsId}-${String(key)}`
            return <Fragment key={key}>
              <tr className={rowClass?.(row)}>
                {rowNumbers && <td className="row-number">{(pagination?.offset ?? 0) + index + 1}</td>}
                {columns.map((column) => (
                  <td key={column.key} className={column.align === 'left' ? 'left' : undefined}>
                    {href && ['player_display_name', 'college_name'].includes(column.key)
                      ? <ProfileRouteLink to={href}>{column.render ? column.render(row) : String(cellValue(row, column) ?? '—')}</ProfileRouteLink>
                      : column.render ? column.render(row) : String(cellValue(row, column) ?? '—')}
                  </td>
                ))}
                {renderDetails && <td className="left">
                  <button type="button" className="detail-toggle" aria-expanded={open}
                    aria-controls={open ? id : undefined}
                    aria-label={`${open ? 'Hide' : 'Show'} details for ${rowLabel?.(row) ?? key}`}
                    onClick={() => setExpanded(open ? null : key)}>{open ? 'Close' : 'Details'}</button>
                </td>}
              </tr>
              {open && renderDetails && <tr className="expanded-row"><td colSpan={columns.length + 1 + Number(rowNumbers)}>
                <div id={id}>{renderDetails(row)}</div>
              </td></tr>}
            </Fragment>
          })}
        </tbody>
      </table>
    </div>
  )
}
