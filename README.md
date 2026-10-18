# Forest Archive

A Telegram bot built with Python 3.12 and aiogram 3. It imports documents from
a public Google Drive folder, responds to `/start` and echoes text messages.
The application runs with Docker Compose.

## Requirements

- Docker Engine with Docker Compose. On Windows, use Docker Desktop
  with Linux containers.
- A Telegram bot token from [@BotFather](https://t.me/BotFather).
- A public Google Drive archive folder and a Google API key with Drive API enabled.

## Run

1. Clone the repository and open its directory.
2. Copy `.env.example` to `.env`:

   ```powershell
   # PowerShell
   Copy-Item .env.example .env
   ```

   ```sh
   # Linux / macOS
   cp .env.example .env
   ```

3. Open `.env` and set `BOT_TOKEN`, `GOOGLE_API_KEY` and `DRIVE_FOLDER_ID`.
   See the archive access instructions below. Keep `DATA_DIR=/app/data` for Docker.
4. Start the application:

   ```sh
   docker compose up --build
   ```

Python and dependencies are installed inside the Docker image.
Run only one instance with a given bot token. Press `Ctrl+C` to stop the bot.
Run `docker compose down` to remove the stopped container and Compose network.

After changing the code or `.env`, stop the application and run
`docker compose up --build` again to apply the changes.

## Configuration

| Variable | Description |
|---|---|
| `BOT_TOKEN` | Required Telegram bot token |
| `GOOGLE_API_KEY` | Required Google API key restricted to Google Drive API |
| `DRIVE_FOLDER_ID` | Required ID of the public archive folder |
| `DATA_DIR` | Data directory; `/app/data` in Docker Compose |

Compose reads `.env` and passes its values as environment variables to the
container. A missing or malformed token stops the application with an error.
Telegram checks the token's authenticity when the bot connects.

Keep tokens and API keys in your local `.env` file. `.env` and local credentials
are excluded from Git and are not copied into the Docker image.

## Archive access

The archive folder must allow viewing by anyone with its link. The importer
uses an API key to read public files; the key does not grant access to private
folders. All Drive requests are GET requests. The application does not change
files, permissions or folder contents.

1. Select or create a project in [Google Cloud Console](https://console.cloud.google.com/).
2. Open **APIs & Services > Library**, find **Google Drive API** and enable it.
3. Open **Credentials > Create credentials > API key**.
4. Restrict the key to **Google Drive API** under **API restrictions**.
   Website/referrer restrictions do not apply to this backend application.
5. Save the key as `GOOGLE_API_KEY` in your local `.env`.
6. Set `DRIVE_FOLDER_ID` to the ID after `/folders/` in the archive URL.

The importer reads supported files directly inside the configured folder,
following every API pagination token. Import runs before Telegram polling and
runs again on each application start. The same Drive file ID updates its
existing local record. Unsupported formats are skipped. If a supported file
cannot be read or the folder is inaccessible, startup stops with an error.
Download errors identify the filename and HTTP status without logging API keys
or request URLs. Google can reject automated downloads even when the same file
can be downloaded in a browser. In that case, check API restrictions and usage
in Google Cloud Console and try again later; changing archive permissions is
not necessarily required.

See Google's [public-folder API documentation](https://developers.google.com/workspace/drive/api/guides/search-files#list_files_in_a_public_folder).

## Project structure

- `bot/__main__.py` - application entry point, router registration and polling.
- `bot/config.py` - environment configuration and token validation.
- `bot/handlers.py` - Telegram message handlers.
- `bot/services/documents.py` - PDF text extraction and OCR with PyMuPDF,
  plus DOCX and TXT parsing.
- `bot/services/drive.py` - read-only public archive import.
- `bot/storage.py` - SQLite metadata and original document storage.
- `requirements.txt` - application dependencies.
- `pytest.ini` - test discovery and application import path.
- `Dockerfile` - application image definition.
- `docker-compose.yaml` - single-service configuration.

The bot uses long polling, so it does not require a web server, exposed ports
or a webhook. The `Bot` context manager closes the network session when polling
stops.

## Manual verification

With the application running and a valid token configured:

1. Open your bot in Telegram and send `/start`. Expect one start response.
2. Send a text message. Expect one reply containing the same text.
3. Send a photo or sticker. Expect no response from the text handler.
4. Press `Ctrl+C` in the terminal. Polling should stop cleanly.
5. Start the application again. The import should update existing documents
   without duplicating them.

## Document processing

The document service accepts PDF, DOCX and UTF-8 TXT files up to 10 MiB.
PDF documents are limited to 100 pages. DOCX paragraphs and tables are read
in document order. Encrypted PDFs, empty files and unsupported formats are rejected.

For each PDF page, PyMuPDF extracts embedded text first. Pages without embedded
text are rendered and recognized in memory through PyMuPDF's integrated OCR
using Ukrainian and English language data (`ukr+eng`). PyMuPDF includes the
Tesseract OCR engine; the application does not launch external OCR programs.
Pages are kept in their original order.
Text inside images on a page that also contains selectable text is not recognized.
OCR accuracy depends on scan quality, layout and orientation; recognition errors
are possible, especially in names and numbers. No document content is sent to
an external OCR service, and original PDF files are preserved unchanged.

OCR processes one page at a time, at up to 300 DPI. Resolution is reduced for
large pages to keep the rendered image within 3500 pixels on its longest side.
No temporary PDF or image files are written. PyMuPDF does not expose a hard OCR
timeout; processing time depends on document complexity and page count.
An OCR error stops import with an explanation. Import and OCR run again on each
application start, so scanning a large archive can delay Telegram polling.

The storage service keeps extracted text and metadata in `archive.sqlite3`
and original file bytes under `documents/` within its configured data directory.
Original filenames are stored as metadata; generated filenames prevent uploaded
names from being used as local paths. Reimporting the same Drive file ID updates
the existing record instead of creating a duplicate.

Docker Compose mounts the `bot_data` named volume at `/app/data`. Both the
database and originals survive container restarts, rebuilds and
`docker compose down`. Removing this volume also removes the stored data.

## Tests

The Docker image includes PyMuPDF and both OCR language files. Poppler and
Tesseract command-line programs are not required.

To run OCR outside Docker, download `eng.traineddata` and `ukr.traineddata` from
the official [Tesseract language data repository](https://github.com/tesseract-ocr/tessdata_fast)
into one directory. Set `TESSDATA_PREFIX` to that directory in your terminal:

```powershell
$env:TESSDATA_PREFIX = "C:\path\to\tessdata"
```

On Linux / macOS:

```sh
export TESSDATA_PREFIX=/path/to/tessdata
```

Docker sets this variable automatically. Text-based PDF, DOCX and TXT parsing
does not require OCR language files.

For local development, create a Python 3.12 virtual environment and install
the dependencies:

```sh
python -m venv .venv
```

On PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
pytest
```

On Linux / macOS:

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest
```

Run these commands from the repository root. Activate the environment in each
new terminal. `pytest.ini` makes the `bot` package importable for both `pytest`
and `python -m pytest`; `python -m pytest` explicitly uses the selected Python
interpreter.

Activation is optional when using the environment's Python directly. This also
works if PowerShell blocks `Activate.ps1`:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

On Linux / macOS, the equivalent command is `.venv/bin/python -m pytest`.

Tests use generated documents and temporary directories. They do not require
Telegram tokens, Google credentials or private archive files.
The real OCR test is skipped when the OCR language files are unavailable locally.
To run every test, including real OCR, in the built Docker image:

```sh
docker compose run --rm --no-deps -T --volume "./tests:/app/tests:ro" bot python -m pytest -q -p no:cacheprovider /app/tests
```
