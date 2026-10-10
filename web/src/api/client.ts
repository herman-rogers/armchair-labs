import type { DataCatalog, Status } from './types'

type RequestOptions = { dataCatalog?: string; signal?: AbortSignal; method?: 'GET' | 'POST' }

/**
 * Every API call goes through here: it pins the published table catalog, signals a
 * stale catalog, and turns failures into the API's `detail` message.
 */
async function request(path: string, { dataCatalog, signal, method = 'GET' }: RequestOptions = {}) {
  const headers: Record<string, string> = {}
  if (dataCatalog) headers['X-Data-Catalog'] = dataCatalog
  const response = await fetch(path, { method, headers, signal })
  if (!response.ok) {
    if (response.headers.get('X-Data-Catalog-Stale') === 'true') {
      window.dispatchEvent(new Event('data-catalog-changed'))
    }
    if (response.headers.get('X-Table-Catalog-Stale') === 'true') {
      window.dispatchEvent(new Event('table-catalog-changed'))
    }
    const body = (await response.json().catch(() => ({}))) as { detail?: string }
    throw new Error(body.detail ?? `${response.status} ${response.statusText}`)
  }
  return response
}

export const getResponse = (path: string, dataCatalog?: string, signal?: AbortSignal) =>
  request(path, { dataCatalog, signal })

export async function get<T>(path: string, dataCatalog?: string, signal?: AbortSignal): Promise<T> {
  return (await getResponse(path, dataCatalog, signal)).json() as Promise<T>
}

export async function post<T>(path: string, options: Pick<RequestOptions, 'dataCatalog'> = {}): Promise<T> {
  return (await request(path, { ...options, method: 'POST' })).json() as Promise<T>
}

export const fetchStatus = () => get<Status>('/api/status')
export const fetchDataCatalog = () => get<DataCatalog>('/api/tables/catalog')
