import { useEffect, useState } from 'react'

/** Keep typing immediate while coalescing server-side searches. */
export function useDebounced(value: string, delay = 200) {
  const [settled, setSettled] = useState(value)
  useEffect(() => {
    const timeout = setTimeout(() => setSettled(value), delay)
    return () => clearTimeout(timeout)
  }, [value, delay])
  return settled
}
