/** Complete small result sets with bounded concurrency; large datasets use server pages. */
export async function allPages<T extends { total: number }>(
  params: URLSearchParams, fetchPage: (params: URLSearchParams) => Promise<T>, key: NoInfer<keyof T>, pageSize = 500,
): Promise<T> {
  const request = (offset: number) => {
    const page = new URLSearchParams(params)
    page.set('offset', String(offset)); page.set('limit', String(pageSize))
    return fetchPage(page)
  }
  const first = await request(0)
  const rows = [...first[key] as unknown[]]
  const changed = () => new Error('Results changed while loading. Please refresh the view.')
  if (rows.length !== Math.min(pageSize, first.total)) throw changed()
  for (let offset = rows.length; offset < first.total; offset += pageSize * 3) {
    const offsets = [offset, offset + pageSize, offset + pageSize * 2].filter(value => value < first.total)
    const pages = await Promise.all(offsets.map(request))
    pages.forEach((next, i) => {
      const batch = next[key] as unknown[]
      if (next.total !== first.total || batch.length !== Math.min(pageSize, first.total - offsets[i])) throw changed()
      rows.push(...batch)
    })
  }
  return { ...first, [key]: rows }
}
