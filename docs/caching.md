# Caching and invalidation

Current cache: in-process retrieval cache keyed by `(workspace_id, normalized_query)`.

Invalidation rules:
- on successful ingestion (new content)
- on re-indexing existing documents
- on document deletion

TTL is 60 seconds to bound stale data during concurrent writes.
