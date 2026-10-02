# Shared data references

`profiles.json` specifies which local datasets belong in each download profile.
Published `RELEASE.json` files pin the store URI, release-manifest SHA-256 and GCS
generation. `current.json` selects the default release after a verified publication.
These files are small and safe to version; data bytes live in GCS and the local cache.

See [the shared-data workflow](../../docs/operations/shared-data.md) for snapshot,
upload, fetch, offline recovery, clean notebook versioning and rebuild commands.
If no `current.json` exists yet, the initial cloud publication has not completed.
