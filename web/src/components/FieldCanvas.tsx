import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from 'react'
import type { TrackingPlay } from '../api/tracking'
import { FIELD_LENGTH, FIELD_WIDTH, lastObserved, sample } from '../replay'

export interface FieldCanvasHandle {
  /** Draw the play at fractional frame `t`. Called from the playback loop, outside React renders. */
  draw: (t: number) => void
}

interface Props {
  play: TrackingPlay
  selectedId: string | null
  onSelect: (id: string | null) => void
}

/** Hash marks sit 70 feet 9 inches from each sideline. */
const HASH = 70.75 / 3
const cssVar = (name: string, fallback: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback

function palette() {
  const dark = document.documentElement.dataset.theme === 'dark'
  return {
    turf: dark ? '#1d3b27' : '#3a7a48',
    endzone: dark ? '#162f20' : '#2f6a3d',
    lines: dark ? 'rgba(220, 232, 226, .55)' : 'rgba(255, 255, 255, .75)',
    numbers: dark ? 'rgba(220, 232, 226, .5)' : 'rgba(255, 255, 255, .7)',
    offense: cssVar('--wr', '#396795'),
    defense: cssVar('--sell', '#863729'),
    ball: dark ? '#d9a066' : '#7a4a1e',
    los: '#4cc3ff',
    firstDown: '#ffd23f',
    label: dark ? '#f1f5f9' : '#ffffff',
    selected: dark ? '#fbbf24' : '#ffe08a',
    text: cssVar('--text', '#2d383d'),
  }
}

/**
 * The field and everything on it, drawn imperatively on a 2D canvas so playback runs
 * at animation-frame rate without re-rendering React. Field x runs left to right over
 * 120 yards; field y is drawn upward so orientation angles keep their NFL convention.
 */
export const FieldCanvas = forwardRef<FieldCanvasHandle, Props>(function FieldCanvas({ play, selectedId, onSelect }, ref) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const frame = useRef(0)
  const positions = useRef<{ id: string; px: number; py: number }[]>([])

  const draw = useCallback((t: number) => {
    frame.current = t
    const el = canvas.current
    if (!el) return
    const ctx = el.getContext('2d')
    if (!ctx) return
    const width = el.clientWidth
    const height = Math.round(width * FIELD_WIDTH / FIELD_LENGTH)
    const dpr = window.devicePixelRatio || 1
    if (el.width !== Math.round(width * dpr) || el.height !== Math.round(height * dpr)) {
      el.width = Math.round(width * dpr); el.height = Math.round(height * dpr); el.style.height = `${height}px`
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const scale = width / FIELD_LENGTH
    const X = (x: number) => x * scale
    const Y = (y: number) => height - y * scale
    const colors = palette()

    // Turf, end zones, yard lines, hashes and numbers.
    ctx.fillStyle = colors.turf; ctx.fillRect(0, 0, width, height)
    ctx.fillStyle = colors.endzone; ctx.fillRect(0, 0, X(10), height); ctx.fillRect(X(110), 0, X(10), height)
    ctx.strokeStyle = colors.lines; ctx.lineWidth = 1
    for (let x = 10; x <= 110; x += 5) { ctx.beginPath(); ctx.moveTo(X(x), 0); ctx.lineTo(X(x), height); ctx.stroke() }
    ctx.beginPath(); ctx.rect(X(0) + .5, .5, X(120) - 1, height - 1); ctx.stroke()
    ctx.lineWidth = 1
    for (let x = 11; x < 110; x++) {
      if (x % 5 === 0) continue
      for (const y of [1, HASH, FIELD_WIDTH - HASH, FIELD_WIDTH - 1]) {
        ctx.beginPath(); ctx.moveTo(X(x), Y(y) - 3); ctx.lineTo(X(x), Y(y) + 3); ctx.stroke()
      }
    }
    if (scale >= 5) {
      ctx.fillStyle = colors.numbers; ctx.font = `${Math.max(10, scale * 1.6)}px ui-sans-serif, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
      for (let x = 20; x <= 100; x += 10) {
        const label = String(x <= 60 ? x - 10 : 110 - x)
        ctx.fillText(label, X(x), Y(FIELD_WIDTH - 6)); ctx.fillText(label, X(x), Y(6))
      }
    }

    // Line of scrimmage and first-down marker.
    const marker = (x: number | null, color: string) => {
      if (x == null) return
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(x), 0); ctx.lineTo(X(x), height); ctx.stroke()
    }
    marker(play.line_of_scrimmage, colors.los)
    marker(play.first_down_line, colors.firstDown)

    if (play.ball_land) {
      const px = X(play.ball_land.x), py = Y(play.ball_land.y), r = Math.max(4, scale * .6)
      ctx.strokeStyle = colors.firstDown; ctx.lineWidth = 2
      ctx.beginPath(); ctx.moveTo(px - r, py - r); ctx.lineTo(px + r, py + r); ctx.moveTo(px - r, py + r); ctx.lineTo(px + r, py - r); ctx.stroke()
    }

    // Trail for the selected entity.
    const selected = play.entities.find(e => e.id === selectedId)
    if (selected) {
      ctx.strokeStyle = colors.selected; ctx.lineWidth = 2; ctx.beginPath()
      let started = false
      for (let i = 0; i <= Math.min(Math.floor(t), play.frames - 1); i++) {
        const x = selected.x[i], y = selected.y[i]
        if (x == null || y == null) continue
        if (started) ctx.lineTo(X(x), Y(y)); else { ctx.moveTo(X(x), Y(y)); started = true }
      }
      ctx.stroke()
    }

    const radius = Math.max(4.5, scale * .6)
    positions.current = []
    const fontPx = Math.max(9, Math.round(scale * 1.2))
    for (const entity of play.entities) {
      let x = sample(entity.x, t), y = sample(entity.y, t)
      let stale = false
      if (x == null || y == null) {
        const last = lastObserved(entity.x, t)
        if (last < 0) continue
        x = entity.x[last]!; y = entity.y[last]!; stale = true
      }
      const px = X(x), py = Y(y)
      positions.current.push({ id: entity.id, px, py })
      if (entity.kind === 'ball') {
        ctx.fillStyle = colors.ball; ctx.beginPath(); ctx.ellipse(px, py, radius * .75, radius * .5, 0, 0, Math.PI * 2); ctx.fill()
        ctx.strokeStyle = colors.label; ctx.lineWidth = 1; ctx.stroke()
        continue
      }
      ctx.globalAlpha = stale ? .35 : 1
      ctx.fillStyle = entity.side === 'defense' ? colors.defense : colors.offense
      ctx.beginPath(); ctx.arc(px, py, radius, 0, Math.PI * 2); ctx.fill()
      ctx.lineWidth = entity.id === selectedId ? 3 : 1.25
      ctx.strokeStyle = entity.id === selectedId ? colors.selected : colors.label
      ctx.stroke()
      const o = sample(entity.o, t)
      if (o != null && !stale) {
        const rad = o * Math.PI / 180
        ctx.strokeStyle = colors.label; ctx.lineWidth = 2; ctx.beginPath()
        ctx.moveTo(px, py); ctx.lineTo(px + Math.sin(rad) * radius * 1.7, py - Math.cos(rad) * radius * 1.7); ctx.stroke()
      }
      if (scale >= 6 && entity.position) {
        ctx.fillStyle = colors.label; ctx.font = `600 ${fontPx}px ui-sans-serif, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'top'
        ctx.fillText(entity.id === selectedId ? entity.name : entity.position, px, py + radius + 2)
      }
      ctx.globalAlpha = 1
    }
  }, [play, selectedId])

  useImperativeHandle(ref, () => ({ draw }), [draw])

  // Redraw at the current frame whenever the play, selection, size or theme changes.
  useEffect(() => {
    draw(frame.current)
    const el = canvas.current
    if (!el) return
    const observer = new ResizeObserver(() => draw(frame.current))
    observer.observe(el)
    const theme = new MutationObserver(() => draw(frame.current))
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => { observer.disconnect(); theme.disconnect() }
  }, [draw])

  const pick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const mx = event.clientX - rect.left, my = event.clientY - rect.top
    let best: { id: string; d: number } | null = null
    for (const p of positions.current) {
      const d = Math.hypot(p.px - mx, p.py - my)
      if (d <= 16 && (!best || d < best.d)) best = { id: p.id, d }
    }
    onSelect(best ? (best.id === selectedId ? null : best.id) : null)
  }

  return <canvas ref={canvas} className="field-canvas" role="img" aria-label="Field replay" onClick={pick} />
})
