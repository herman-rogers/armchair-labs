import { useQuery } from '@tanstack/react-query'
import { useDataRelease } from '../../dataRelease'
import { catalogQuery } from '.'

/** The current analysis catalog for the verified data release. */
export const useCatalog = () => useQuery(catalogQuery(useDataRelease().token))
