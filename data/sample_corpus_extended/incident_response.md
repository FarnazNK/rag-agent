# Security Incident Response

## Reporting
- Suspected incident: page the on-call security engineer in PagerDuty
  ("security-oncall" service).
- Phishing: forward the email to phishing@company.example as an attachment.
- Lost device: notify IT immediately and file a ticket in #it-help.

## Severity levels
- SEV1: active breach, customer data exposure. Response: full team paged.
- SEV2: confirmed unauthorized access without exfiltration. Response: team
  paged, public disclosure within 72 hours.
- SEV3: phishing attempt or policy violation. Response: investigated within
  one business day.

## Post-incident
- Blameless postmortem within 5 business days.
- Action items tracked in the security backlog with explicit owners.
