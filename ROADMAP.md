# Roadmap

What ships in each version. Design details are in [`docs/DESIGN.md`](docs/DESIGN.md).

## v1 — Expiry tracking and reminders

A Paperless user adds one container and gets:

- expiry dates pulled automatically from IDs, written back to Paperless
- reminders that reach their phone even when the laptop is off
- a verify queue for anything the add-on isn't sure about

**Scope**

- `PaperlessSource`: authenticate, paginate, read OCR text
- MRZ parser with check-digit validation
- Date regex patterns
- Ollama fallback with JSON schema output
- Write custom fields and the verify tag back to Paperless
- Eval script: accuracy on a labelled synthetic test set, including OCR-noised samples, reported at each confidence level
- Incremental sync with cursor and backoff
- Reminder scheduler, `.ics` feed, ntfy and email push
- Catch-up check on wake
- HTMX pages: expiring documents, reminder settings, verify queue
- Docker image and example Docker Compose file
- README quick start, security notes and a CONTRIBUTING guide

**Not in v1:** application packs, mobile app, cloud hosting, document storage of its own (Paperless owns storage), multi-user support, legal or immigration advice.

## v1.1 — Application packs

Check a YAML template of required documents against the archive and export what's found.

- YAML loader with schema validation
- Matching engine for all rules
- Three shipped templates
- Packs page and PDF/ZIP export with a cover checklist

**Matching rules:** `kinds`, `count`, `must_be_valid`, `max_age_days`, `tax_years`, `covers_days` and `manual`.

**Templates.** Each template is a YAML file that users can copy and edit, and each links to an official or lender checklist because requirements change. The lists below are typical starting points to verify before release, not advice.

```yaml
id: ontario-rental-application
name: Ontario rental application
source_note: Landlords vary; confirm with the listing.
requirements:
  - id: photo-id
    label: Government photo ID
    kinds: [drivers_licence, passport, pr_card]
    must_be_valid: true
  - id: payslips
    label: Two most recent payslips
    kinds: [payslip]
    count: 2
    max_age_days: 60
  - id: employment-letter
    label: Employment letter
    kinds: [employment_letter]
    max_age_days: 90
  - id: credit-report
    label: Credit report
    kinds: [credit_report]
    max_age_days: 30
```

```yaml
id: mortgage-pre-approval
name: Mortgage pre-approval (Canada)
source_note: Lenders vary; confirm with your lender or broker.
requirements:
  - id: photo-id
    label: Government photo ID
    kinds: [drivers_licence, passport, pr_card]
    must_be_valid: true
  - id: payslips
    label: Recent payslips
    kinds: [payslip]
    count: 2
    max_age_days: 60
  - id: employment-letter
    label: Employment letter
    kinds: [employment_letter]
    max_age_days: 90
  - id: t4s
    label: T4 slips, last 2 years
    kinds: [t4]
    tax_years: 2
  - id: noas
    label: Notices of Assessment, last 2 years
    kinds: [noa]
    tax_years: 2
  - id: down-payment
    label: Bank statements showing down payment (90 days)
    kinds: [bank_statement]
    covers_days: 90
```

```yaml
id: canadian-passport-renewal
name: Canadian passport renewal
source_url: https://www.canada.ca/en/immigration-refugees-citizenship/services/canadian-passports.html
requirements:
  - id: current-passport
    label: Current or most recent passport
    kinds: [passport]
  - id: photos
    label: Passport photos
    manual: true   # a physical item; user ticks it off
```

## v1.2 — MCP server

- Expose expiring documents and pack status as MCP tools
- Example: "Do I have everything for my mortgage pre-approval?"

## Later

- Folder mode for users without Paperless
- More templates contributed by users
