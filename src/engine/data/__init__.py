"""Source acquisition and validated build inputs for the analytical table system.

Production entry point: `engine data refresh --upload` (pipeline.py → refresh.py).
Raw captures and enrichment are provenance, not alternative query stores.
Named tables, recipe builds, SQL, and GCS distribution live in engine.tables.
The research/ command wrappers delegate here; archived product recipes retain
historical serialized references, read through releases.py compatibility adapters.
"""
