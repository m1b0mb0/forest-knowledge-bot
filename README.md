# Forest Archive

A Telegram bot that answers Ukrainian questions about a public Google Drive
archive. Groq selects documents by filename, reads their full texts and returns
an answer with source names. Documents persist in SQLite and a Docker volume.

## Requirements

- Docker Engine with Docker Compose; on Windows, Docker Desktop with Linux containers.
- A Telegram bot token from [@BotFather](https://t.me/BotFather).
- A public Google Drive folder and a Google API key.
- A free [Groq API key](https://console.groq.com/keys).

## Setup and run

1. Copy `.env.example` to `.env`.
2. In [Google Cloud Console](https://console.cloud.google.com/), enable
   **Google Drive API**, create an API key and restrict it to **Google Drive API**.
   The archive folder must allow viewing by anyone with its link.
3. Create a Groq API key and fill in `.env`:

   | Variable | Value |
   |---|---|
   | `BOT_TOKEN` | Telegram token from BotFather |
   | `GROQ_API_KEY` | Groq API key |
   | `GROQ_MODEL` | `openai/gpt-oss-20b` (default) or `openai/gpt-oss-120b` |
   | `GOOGLE_API_KEY` | Google Drive API key |
   | `DRIVE_FOLDER_ID` | Folder ID after `/folders/` in its URL |
   | `DATA_DIR` | Keep `/app/data` for Docker |

4. Start the application:

   ```sh
   docker compose up --build
   ```

Open the bot in Telegram and send `/start`, then ask a question in Ukrainian.
Include the place name when relevant; the bot asks for clarification if needed.
Run one instance per bot token. Press `Ctrl+C` to stop it; run
`docker compose down` to remove the stopped container.
Rebuild after changing code or configuration.

Keep secrets in the local `.env`, which is excluded from Git and the Docker image.

## Documents and storage

- Imports PDF, DOCX and UTF-8 TXT files directly inside the configured folder.
  Google Drive access is read-only; nested folders are not imported.
- Limits: 10 MiB per file and 100 pages per PDF.
- PyMuPDF extracts PDF text and uses Ukrainian and English OCR data for pages
  without embedded text. OCR accuracy depends on scan quality; images on pages
  that also contain embedded text are not recognized.
- Startup checks Drive metadata and reuses unchanged local documents.
  The first import can take time; an unreadable supported file stops startup.
- SQLite and original files persist in the `bot_data` Docker volume.
  Removing the volume deletes the stored data.
- Groq chooses up to three active documents using their filenames, so filenames
  should describe their topic and place. Full selected texts are sent to Groq;
  oversized requests require a narrower question. Sources use stored filenames.
- [Groq rate limits](https://console.groq.com/docs/rate-limits) apply. The bot
  reports temporary API errors; it does not retry automatically.

## Tests

For local testing, use Python 3.12 and run from the repository root:

```sh
python -m venv .venv
```

Activate `.venv` with `.\.venv\Scripts\Activate.ps1` on PowerShell,
or `source .venv/bin/activate` on Linux/macOS, then run:

```sh
python -m pip install -r requirements.txt
pytest
```

Tests need no tokens, private documents or OCR language data; OCR is mocked.

Alternatively, run tests in the built Docker image:

```sh
docker compose run --rm --no-deps -T --volume "./tests:/app/tests:ro" bot python -m pytest -q -p no:cacheprovider --basetemp=/tmp/forest-archive-tests /app/tests
```
