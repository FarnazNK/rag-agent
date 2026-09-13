# Deployment runbook

## Environments
- dev: local docker compose
- staging: pre-production image tags and isolated DB
- prod: promoted image tags, strict secrets and monitoring

## Safety mechanisms
- readiness: `GET /health`
- schema migration: `alembic upgrade head` before traffic shift
- rollback: redeploy previous image tag and run backward-compatible downgrade if needed
- image versioning: immutable tags (`rag-agent:<git-sha>`)
- staged flow: dev -> staging smoke -> prod
- post-deploy smoke test:
  1. register/login
  2. create org/workspace
  3. ingest document
  4. run query and verify citation snippets
