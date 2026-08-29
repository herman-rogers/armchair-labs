/** Renders the board's flag string as coloured chips.
 *
 * The colours carry the meaning the flags carry: green is an opportunity to buy,
 * red is a price to sell into, amber is a discount, blue is "verify this sample".
 */

const CLASSES: Record<string, string> = {
  BUY: 'buy',
  'TD-luck': 'sell',
  age: 'age',
}

const TITLES: Record<string, string> = {
  BUY: 'Scored well under expectation with real volume behind it — the price is wrong in your favour.',
  'TD-luck': 'Scored well over expectation — the price is built on luck that will not repeat.',
  age: 'Running back at or past the 27.5 age cliff. A discount, not a disqualification.',
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
