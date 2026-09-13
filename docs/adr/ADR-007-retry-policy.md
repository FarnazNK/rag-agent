# ADR-007: Retry policy
Retries are enabled only for idempotent embedding requests with exponential backoff. Mutating operations and user queries are not retried automatically to prevent duplicate writes and hidden costs.
