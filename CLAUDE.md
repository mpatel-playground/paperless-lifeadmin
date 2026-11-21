# CLAUDE.md

Guidance for Claude Code working in this repo.

## Before any task

1. **Read `docs/DESIGN.md` and `.local/PLAN.md` first.** `docs/DESIGN.md` is the source of truth for architecture, repo structure, data model, core flows and security. `.local/PLAN.md` holds the build plan and is gitignored, so it may be missing on other machines; if it is, carry on without it and never commit it. `ROADMAP.md` defines what belongs in each version.
2. **v1 scope only.** Do not build anything `ROADMAP.md` puts in v1.1 or later. That means no application packs (`packs/` YAML, `src/lifeadmin/packs/engine.py`, `export.py`, `pack_run` / `pack_item` tables), no MCP server (`src/lifeadmin/mcp/`) and no folder mode (`sources/folder.py`). If a task seems to need them, stop and ask.
3. **Show a plan before writing code.** Give a short plan (files to touch, approach, tests to add) and wait for approval before editing.

## Stack

- Python 3.12
- FastAPI (web app + HTMX templates in `src/lifeadmin/ui/`)
- SQLite (metadata and state only, never document copies)
- httpx for all HTTP calls (Paperless REST API, Ollama, ntfy)
- pytest for tests
- ruff for linting and formatting

Don't add other frameworks or major dependencies without asking.

## Commands

```bash
ruff check .          # lint
ruff format .         # format
pytest                # run tests
```

Run `ruff check`, `ruff format --check` and `pytest` before every commit; all must pass.

## Testing rules

- **Every module gets unit tests** in `tests/`, mirroring the `src/lifeadmin/` layout (e.g. `extract/mrz.py` → `tests/extract/test_mrz.py`).
- **Mock all HTTP calls.** Never hit real services in tests: no real Paperless, Ollama, ntfy or SMTP. Use `httpx.MockTransport` (or `respx`) for httpx, and a fake for SMTP.
- Use a temporary or in-memory SQLite database per test.
- Cover the idempotency guarantees from the spec: the sync cursor advances only on success, verified documents are skipped, and `sent_at` prevents duplicate reminders.

## Data and secrets

- **Never commit real personal documents or tokens.** No real passports, IDs, payslips, tax slips or Paperless/API tokens, ever.
- **Test data is synthetic only.** Fixtures and `evals/` samples use made-up names, numbers and dates (MRZ samples need valid check digits computed on fake data).
- Config comes from environment variables (`PAPERLESS_URL`, `PAPERLESS_TOKEN`, `OLLAMA_URL`, ...) via `config.py`. Keep `.env` files out of git and use placeholder values in examples.
- Follow the spec's privacy rules: bind to `127.0.0.1` by default, no telemetry, and reminder text never includes document numbers or names.

## Commits

- Small, focused commits, one logical change each.
- Clear messages in the imperative mood, e.g. `Add MRZ check-digit validation`.
- Don't commit unless asked, and check `git diff --staged` for secrets or real data before committing.
