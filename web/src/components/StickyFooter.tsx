import type { ReactNode } from 'react'

/** In-flow footer that stays at the viewport bottom without covering the last row. */
export function StickyFooter({ children, label }: { children: ReactNode; label: string }) {
  return <footer className="sticky-footer" role="region" aria-label={label}><div className="analysis-controls">{children}</div></footer>
}
