# Repository Standards

Every service repository must include:
- README with setup instructions
- CONTRIBUTING.md with the local dev workflow
- CODEOWNERS file
- Working `make test` target
- Working `make lint` target
- Dockerfile (if deployable)
- CI configuration that runs lint, type checks, and tests on every PR

Repos missing these are flagged in the quarterly tech-debt review.
