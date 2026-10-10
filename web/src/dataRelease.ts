import { createContext, useContext } from 'react'
import type { DataCatalog } from './api/types'

/** The published table catalog; its checksum keys every observation query. */
export const ReleaseContext = createContext<{ token?: string; catalog?: DataCatalog }>({})
export const useDataRelease = () => useContext(ReleaseContext)
