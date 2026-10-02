import { ProfileRouteLink } from './PlayerLink'
import { Fragment, useEffect, useId, useMemo, useState } from 'react'
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

/** URL sort token: `points` ascending, `-points` descending. */
const sortToken = (key: string, direction: Direction) => direction === 'desc' ? `-${key}` : key

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
}: {
  rows: Row[]
  /** API has already sorted and paginated the complete filtered result. */
  serverSide?: boolean
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
  const defaultDirection = columns.find((column) => column.key === defaultSort)?.initial ?? 'desc'
  const [localSortKey, setLocalSortKey] = useState(defaultSort)
  const [localDirection, setLocalDirection] = useState<Direction>(defaultDirection)
  const [params, updateParams] = useUrlParams()

  // The table stays mounted when the metric tab changes. Resetting is required:
  // otherwise V2 can keep sorting on V1's adj_vor (or V1 can keep V2's score), making
  // two genuinely different boards appear identical.
  useEffect(() => {
    setLocalSortKey(defaultSort)
    setLocalDirection(defaultDirection)
  }, [defaultSort, defaultDirection])

  // A URL sort naming a column this table doesn't have (an old link, another view's
  // column) falls back to the default rather than leaving the rows unsorted.
  const urlSort = sortParam ? params.get(sortParam) : null
  const urlSortKey = urlSort?.replace(/^-/, '')
  const fromUrl = urlSortKey != null && columns.some((column) => column.key === urlSortKey)
  const sortKey = sortParam ? (fromUrl ? urlSortKey : defaultSort) : localSortKey
  const direction: Direction = sortParam ? (fromUrl ? (urlSort!.startsWith('-') ? 'desc' : 'asc') : defaultDirection) : localDirection

  const sortColumn = columns.find((column) => column.key === sortKey)
  const sorted = useMemo(() => {
    if (serverSide || !sortColumn) return rows
    const copy = [...rows]
    copy.sort((a, b) => {
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
    return copy
  }, [rows, sortColumn, direction, serverSide])

  if (!rows.length && emptyMessage) {
    return <div className="notice">{emptyMessage}</div>
  }

  const onSort = (column: Column<Row>) => {
    const nextDirection: Direction = column.key === sortKey ? (direction === 'asc' ? 'desc' : 'asc') : column.initial ?? 'desc'
    if (sortParam) {
      const token = column.key === defaultSort && nextDirection === defaultDirection ? null : sortToken(column.key, nextDirection)
      updateParams({ [sortParam]: token, ...(pagination?.param ? { [pagination.param]: null } : {}) })
      if (!pagination?.param) pagination?.onPage(0)
      return
    }
    pagination?.onPage(0)
    setLocalSortKey(column.key)
    setLocalDirection(nextDirection)
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
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
          {(pagination && !serverSide ? sorted.slice(pagination.offset, pagination.offset + pagination.limit) : sorted).map((row) => {
            const key = rowKey(row)
            const href = profileHref?.(row)
            const open = expanded === key
            const id = `${detailsId}-${String(key)}`
            return <Fragment key={key}>
              <tr className={rowClass?.(row)}>
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
              {open && renderDetails && <tr className="expanded-row"><td colSpan={columns.length + 1}>
                <div id={id}>{renderDetails(row)}</div>
              </td></tr>}
            </Fragment>
          })}
        </tbody>
      </table>
    </div>
  )
}
