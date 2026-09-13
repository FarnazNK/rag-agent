# ADR-002: PostgreSQL + pgvector
We store chunks, metadata, and embeddings in PostgreSQL with pgvector to keep transactional consistency and tenant-scoped filtering in one system.
