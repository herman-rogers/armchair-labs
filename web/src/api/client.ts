import type { BoardResponse, Status } from './types'

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: string }
    throw new Error(body.detail ?? `${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export const fetchStatus = () => get<Status>('/api/status')

export const fetchBoard = (limit = 400) => get<BoardResponse>(`/api/board?limit=${limit}`)
