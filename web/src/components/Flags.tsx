/** Renders the board's flag string as coloured chips.
 *
 * These are historical evidence, not guarantees. Colours distinguish the signal
 * families; the tooltip states the uncertainty the short label cannot.
 */

const CLASSES: Record<string, string> = {
  age: 'age',
  'ESPN-only': 'sample',
}

const TITLES: Record<string, string> = {
  age: 'Running back at or past 27.5. V2 uses a gradual age curve; this historical flag is supporting evidence only.',
  'ESPN-only': 'No prior NFL production is available to Patron. This row is placed by ESPN’s current PPR draft-room rank, and its model metrics are intentionally blank.',
}

export function Flags({ value }: { value: string }) {
  if (!value) return <span className="faint">—</span>

  return (
    <span className="flags">
      {value.split('/').map((flag) => {
        const isSample = /^\d+gms$/.test(flag)
        return (
          <span
            key={flag}
            className={`flag ${isSample ? 'sample' : (CLASSES[flag] ?? 'sample')}`}
            title={
              isSample
                ? `Produced in only ${flag.replace('gms', '')} games. The per-game number is real; the sample is not large.`
                : TITLES[flag]
            }
          >
            {flag}
          </span>
        )
      })}
    </span>
  )
}
