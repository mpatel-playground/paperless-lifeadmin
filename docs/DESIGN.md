# Paperless Life Admin — Design

Paperless Life Admin is a self-hosted add-on for Paperless-ngx. It tracks when personal documents expire and reminds you before each renewal. It runs fully offline on a personal laptop.

**Problem.** Paperless-ngx stores and searches documents well, but it does nothing with what's in them. People still miss passport renewals and scramble to find two payslips and a Notice of Assessment when a landlord or lender asks.

**Who it's for.** Self-hosters who already run Paperless-ngx, starting with people in Canada who juggle IDs, tax slips and pay documents.

See [`ROADMAP.md`](../ROADMAP.md) for what ships in each version.

## Architecture

Three containers run side by side on the laptop; Paperless keeps the files and the add-on only reads and annotates them.

```
                         Laptop
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   ┌──────────────┐   REST API    ┌──────────────────┐    │
│   │ Paperless-ngx│ ◄───────────► │ Life Admin add-on│    │
│   │ (files, OCR) │  read OCR /   │ FastAPI + SQLite │    │
│   └──────────────┘  write fields └────────┬─────────┘    │
│                                           │ fallback     │
│                                  ┌────────▼─────────┐    │
│                                  │ Ollama (optional)│    │
│                                  └──────────────────┘    │
└───────────────────────────────────────────┬──────────────┘
                                            │ reminder text only
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                     .ics calendar feed            ntfy / email push
                       (phone calendar)              (phone)
```

The add-on reads OCR text over Paperless's REST API, writes expiry fields back, and calls Ollama only when MRZ and regex fail. Only reminder text leaves the machine.

## Repo structure

One Python package (FastAPI + SQLite), one Docker image, and templates kept as plain YAML so contributors can add packs without touching code.

```
paperless-lifeadmin/
├── src/lifeadmin/
│   ├── main.py              # FastAPI app + scheduler startup
│   ├── config.py            # env vars (PAPERLESS_URL, TOKEN, OLLAMA_URL)
│   ├── sources/
│   │   ├── base.py          # DocumentSource interface
│   │   ├── paperless.py     # v1: Paperless-ngx REST API
│   │   └── folder.py        # later: watched folder mode
│   ├── extract/
│   │   ├── mrz.py           # passport MRZ parser (deterministic)
│   │   ├── patterns.py      # regex for dates, card formats
│   │   └── llm.py           # Ollama fallback, structured JSON output
│   ├── reminders/
│   │   ├── ics.py           # calendar feed
│   │   ├── ntfy.py          # push to phone
│   │   └── email.py         # SMTP
│   ├── packs/
│   │   ├── engine.py        # v1.1: match requirements to documents
│   │   └── export.py        # v1.1: merged PDF / ZIP
│   ├── mcp/server.py        # v1.2: MCP tools
│   ├── db.py                # SQLite models
│   └── ui/                  # HTMX templates
├── packs/                   # v1.1: YAML application-pack templates
├── evals/                   # labelled sample docs + accuracy script
├── tests/
├── docker-compose.example.yml
├── Dockerfile
└── README.md
```

**Stack choices.** FastAPI and SQLite keep it light enough for a laptop. HTMX gives a usable UI without a separate front-end build. Ollama is optional: with it off, extraction still runs on MRZ and regex.

## Data model

Three SQLite tables in v1 (plus two for packs in v1.1) hold only metadata and state; the documents themselves never leave Paperless.

| Table | Key fields | Purpose |
| --- | --- | --- |
| `tracked_document` | paperless\_id, doc\_kind (passport, pr\_card, drivers\_licence, health\_card, payslip, t4, noa, employment\_letter, credit\_report, bank\_statement), holder, issue\_date, expiry\_date, extracted\_by (mrz, regex, llm, manual), confidence, verified | One row per Paperless document the add-on understands |
| `reminder` | tracked\_document\_id, remind\_on, channel (ics, ntfy, email), sent\_at | Scheduled alerts; `sent_at` makes sending idempotent |
| `pack_run` | template\_id, holder, created\_at, status (ready, missing\_items) | v1.1: one check of a template against the archive |
| `pack_item` | pack\_run\_id, requirement\_id, paperless\_id, state (found, missing, expired, too\_old) | v1.1: how each requirement was satisfied |
| `sync_state` | last\_synced\_at, last\_document\_id | Incremental sync cursor |

**Written back to Paperless.** Custom fields `expiry_date` and `doc_kind`, plus a `lifeadmin:verify` tag when confidence is low, so users fix data in the Paperless UI they already know.

**Lead times** (editable defaults): passport 8 months, PR card 6 months, driver's licence 2 months, health card 2 months.

## Core flows

Every flow is safe to re-run: sync uses a cursor, extraction skips verified documents, and reminders record when they were sent.

### 1. Sync (every 10 minutes, and on startup)

1. Fetch documents changed since `sync_state.last_synced_at` from `GET /api/documents/`, following pagination.
2. Keep documents whose type or tags mark them as IDs, pay or tax documents.
3. Queue new or changed ones for extraction; advance the cursor only after the batch succeeds.
4. On API errors, retry with exponential backoff (max 3), then log and wait for the next run.

### 2. Extraction (cheapest reliable method first)

1. **MRZ:** if the OCR text holds passport or PR card machine-readable lines, parse them and verify their check digits. A pass gives confidence 1.0.
2. **Patterns:** regex for labelled dates ("Expiry", "Exp", "Valid until") and known card formats.
3. **LLM fallback:** send the OCR text to Ollama with a JSON schema (doc\_kind, holder, issue\_date, expiry\_date, confidence).
4. Write results to Paperless custom fields. Below 0.8 confidence, add the `lifeadmin:verify` tag instead of trusting it.

### 3. Reminders

1. Daily job computes `remind_on` = expiry date minus lead time, plus follow-ups at 3 months, 1 month and 1 week.
2. Publish all upcoming reminders as an `.ics` feed; users subscribe once from their phone calendar, so alerts fire even with the laptop off.
3. Optionally push through ntfy or email; `sent_at` prevents duplicates.
4. On wake, a catch-up check sends anything missed while the laptop was asleep.

### 4. Application packs (v1.1)

1. User picks a template and a holder (for example "Ontario rental application" for themselves).
2. The engine resolves each requirement against tracked documents: newest first, within the allowed age, not expired.
3. The UI shows each requirement as found, missing, expired or too old, with a link to the document in Paperless.
4. Export merges found documents into one PDF or a ZIP, with a cover checklist page.

Templates and matching rules are described in [`ROADMAP.md`](../ROADMAP.md#v11--application-packs).

## Security and privacy

The add-on handles passports and payslips, so the default is that nothing leaves the user's machine and the add-on stores as little as possible.

- **Local by default.** Extraction uses Ollama on the same machine. Cloud LLMs are off unless the user sets an API key, and the README says plainly what that sends.
- **No document copies.** Only metadata (dates, kinds, states) lives in SQLite; files stay in Paperless. Exports are written to a temp folder and deleted after download.
- **Least-privilege token.** The README shows how to create a Paperless user limited to the documents the add-on needs.
- **Bound to localhost.** The UI listens on `127.0.0.1` by default; exposing it on the network is an explicit setting with a warning. Add a simple login before anyone runs it beyond localhost.
- **Minimal reminder content.** Calendar and push messages say "Passport expires in 6 months", never document numbers or names. The `.ics` feed URL carries a long random token.
- **No telemetry.** Nothing is sent home.
- **Test data only.** The repo, demo GIF and evals use synthetic sample documents, never real IDs.
