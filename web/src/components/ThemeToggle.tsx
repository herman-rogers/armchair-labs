import { useState } from 'react'

export function ThemeToggle() {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light')
  const nextTheme = theme === 'light' ? 'dark' : 'light'

  function toggleTheme() {
    document.documentElement.dataset.theme = nextTheme
    setTheme(nextTheme)
    try {
      localStorage.setItem('sweaty-plays-theme', nextTheme)
    } catch {
      // The toggle still works for this visit when storage is unavailable.
    }
  }

  return <button type="button" className="theme-toggle" onClick={toggleTheme}
    aria-label={`Switch to ${nextTheme} mode`} title={`Switch to ${nextTheme} mode`}>
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {nextTheme === 'dark'
        ? <path d="M20.9 13.2A9 9 0 0 1 10.8 3.1 9 9 0 1 0 20.9 13.2Z" />
        : <><circle cx="12" cy="12" r="4" />
          <path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>}
    </svg>
    {nextTheme === 'dark' ? 'Dark mode' : 'Light mode'}
  </button>
}
