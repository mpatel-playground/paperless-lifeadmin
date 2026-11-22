# Paperless Life Admin

A self-hosted add-on for [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) that pulls expiry dates out of your IDs and reminds you before each renewal, even when your laptop is off.

> **Status: work in progress.** Not ready for use yet. v1 (expiry tracking and reminders) is being built now.

- Reads OCR text from Paperless over its REST API; your documents never leave Paperless.
- Extracts dates with passport MRZ parsing, then regex, then an optional local LLM (Ollama).
- Flags anything it isn't sure about for you to check, instead of guessing.
- Sends reminders through a calendar feed your phone subscribes to, with optional ntfy or email.
- Runs fully offline. No telemetry.

## Docs

- [Design](docs/DESIGN.md): architecture, data model, core flows, security and privacy
- [Roadmap](ROADMAP.md): what ships in each version

## Licence

[MIT](LICENSE)
