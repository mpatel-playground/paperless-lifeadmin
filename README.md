# Paperless Life Admin

A self-hosted add-on for [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) that pulls expiry dates out of your IDs and reminds you before each renewal, even when your laptop is off.

> **Status: work in progress.** Not ready for use yet. v1 (expiry tracking and reminders) is being built now.

- Reads OCR text from Paperless over its REST API; your documents never leave Paperless.
- Extracts dates with passport MRZ parsing, then regex, then an optional local LLM (Ollama).
- Flags anything it isn't sure about for you to check, instead of guessing.
- Sends reminders through a calendar feed your phone subscribes to, with optional ntfy or email.
- Runs fully offline. No telemetry.

## Extraction accuracy

Measured with `make eval` on 17 synthetic documents (passports, PR cards, Ontario licences and health cards, payslips, T4s, NOAs) run through real Paperless OCR. Each document is scored clean and as 6 OCR-noised copies, 119 cases in all. The noise copies errors seen on real Paperless output, and the heavy level also misreads digits as other digits, which produces valid-looking wrong dates. The LLM is `qwen3.5:9b` on Ollama.

The confidence is computed by the add-on, never asked of the model. Anything below 0.8 goes to the verify queue.

| Confidence | Cases | Right |
| --- | --- | --- |
| 0.95–1.00 | 49 | 100% |
| 0.80–0.95 | 53 | 100% |
| 0.50–0.80 | 15 | 87% |
| below 0.50 | 2 | 50% |

- **Accepted without review:** 102 of 119 (86%), with **none of them wrong**.
- **Expiry dates:** 55 of 56 right, and no expiry date was invented for a document that has none (0 of 63).

"Right" means the document kind and the expiry date (for payslips, T4s and NOAs, the issue date) are both correct. The set is small and synthetic, so read these numbers as a regression check, not a promise about your scans.

## Docs

- [Design](docs/DESIGN.md): architecture, data model, core flows, security and privacy
- [Roadmap](ROADMAP.md): what ships in each version

## Licence

[MIT](LICENSE)
