import { useEffect, useMemo, useState } from 'react'
import type { Key, ReactNode } from 'react'

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

/**
 * The one sortable data table. Column sets differ per screen; the mechanics —
 * click-to-sort headers, null-last ordering, hover titles — do not.
 */
export function DataTable<Row>({
  rows,
  columns,
  defaultSort,
  rowKey,
  rowClass,
  emptyMessage,
}: {
  rows: Row[]
  columns: Column<Row>[]
  defaultSort: string
  rowKey: (row: Row) => Key
  rowClass?: (row: Row) => string | undefined
  /** When set, an empty row list renders as a notice instead of a headers-only table. */
  emptyMessage?: string
}) {
  const defaultDirection = columns.find((column) => column.key === defaultSort)?.initial ?? 'desc'
  const [sortKey, setSortKey] = useState(defaultSort)
  const [direction, setDirection] = useState<Direction>(defaultDirection)

  // The table stays mounted when the metric tab changes. Resetting is required:
  // otherwise V2 can keep sorting on V1's adj_vor (or V1 can keep V2's score), making
  // two genuinely different boards appear identical.
  useEffect(() => {
    setSortKey(defaultSort)
    setDirection(defaultDirection)
  }, [defaultSort, defaultDirection])

  const sortColumn = columns.find((column) => column.key === sortKey)
  const sorted = useMemo(() => {
    if (!sortColumn) return rows
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
  }, [rows, sortColumn, direction])

  if (!rows.length && emptyMessage) {
    return <div className="notice">{emptyMessage}</div>
  }

  const onSort = (column: Column<Row>) => {
    if (column.key === sortKey) {
      setDirection((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(column.key)
      setDirection(column.initial ?? 'desc')
    }
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
                onClick={() => onSort(column)}
              >
                {column.label}
                {sortKey === column.key && (
                  <span className="dir">{direction === 'asc' ? '↑' : '↓'}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={rowKey(row)} className={rowClass?.(row)}>
              {columns.map((column) => (
                <td key={column.key} className={column.align === 'left' ? 'left' : undefined}>
                  {column.render ? column.render(row) : String(cellValue(row, column) ?? '—')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
