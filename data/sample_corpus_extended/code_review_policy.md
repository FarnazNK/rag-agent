# Code Review Policy

## Requirements
- All production code requires at least 1 approving review.
- Critical-path code (auth, billing, data export) requires 2 approving reviews.
- Reviews must be from someone other than the author.

## Turnaround
- Reviewers should respond within 1 business day.
- Review-blocked PRs older than 3 days should be raised in the team channel.

## Style
- Mechanical issues (formatting, naming) belong in linters, not review
  comments. If your linter doesn't catch it, file a ticket to fix the linter.
- Prefer concrete suggestions over abstract criticism.
- Disagreements escalate to the tech lead, not to a longer comment thread.
