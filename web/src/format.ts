/**
 * Display formatting for every table and summary. A missing or non-finite value renders
 * as "—" (unknown, never zero) unless the caller supplies a different placeholder.
 */
const MISSING = '—'
const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)

/** Fixed decimals: `fixed(12.345)` → `12.3`. */
export const fixed = (value: unknown, digits = 1) => isNumber(value) ? value.toFixed(digits) : MISSING

/** Locale digit grouping with `minDigits`–`digits` decimals: `grouped(4312)` → `4,312`. */
export const grouped = (value: unknown, digits = 0, minDigits = digits) => isNumber(value)
  ? value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: minDigits }) : MISSING

/** A 0–1 share as a percentage: `percent(0.42)` → `42%`. */
export const percent = (value: unknown, digits = 0, missing = MISSING) =>
  isNumber(value) ? `${(value * 100).toFixed(digits)}%` : missing

/** A change with an explicit sign: `signed(3)` → `+3`, `signed(-0.25, 2)` → `-0.25`. */
export const signed = (value: unknown, digits?: number) =>
  isNumber(value) ? `${value > 0 ? '+' : ''}${digits == null ? value : value.toFixed(digits)}` : MISSING

/** A rank label: `rank(3)` → `#3`. */
export const rank = (value: unknown) => isNumber(value) ? `#${value}` : MISSING

/** A change in a 0–1 rate, in percentage points: `pointLift(0.034)` → `+3.4 pp`. */
export const pointLift = (value: unknown) => isNumber(value) ? `${signed(value * 100, 1)} pp` : MISSING
