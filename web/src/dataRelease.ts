import { createContext, useContext } from 'react'
import type { DataCatalog } from './api/types'

export const ReleaseContext = createContext<{ token?: string; catalog?: DataCatalog }>({})
export const useDataRelease = () => useContext(ReleaseContext)
