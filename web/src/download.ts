import { useState } from 'react'

/** Save a blob as a file through the browser's download flow. */
export function saveFile(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

/** CSV text from rows of cells, quoting every value. */
export const toCsv = (rows: unknown[][]) =>
  rows.map(row => row.map(value => `"${String(value ?? '').replaceAll('"', '""')}"`).join(',')).join('\n')

/** An export action with its error message; `download` builds the file, then saves it. */
export function useDownload() {
  const [error, setError] = useState('')
  const download = async (filename: string, build: () => Promise<Blob>) => {
    setError('')
    try { saveFile(await build(), filename) }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Export failed') }
  }
  return { download, error }
}
