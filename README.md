# Forest Archive

A Telegram bot built with Python 3.12 and aiogram 3. It responds to `/start`
and echoes text messages. The application runs with Docker Compose.

## Requirements

- Docker Engine with Docker Compose. On Windows, use Docker Desktop
  with Linux containers.
- A Telegram bot token from [@BotFather](https://t.me/BotFather).

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

3. Open `.env` and set `BOT_TOKEN` to your bot token.
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

Compose reads `.env` and passes its values as environment variables to the
container. A missing or malformed token stops the application with an error.
Telegram checks the token's authenticity when the bot connects.

Keep the real token in your local `.env` file. `.env` and local credentials
are excluded from Git and are not copied into the Docker image.

## Project structure

- `bot/__main__.py` - application entry point, router registration and polling.
- `bot/config.py` - environment configuration and token validation.
- `bot/handlers.py` - Telegram message handlers.
- `requirements.txt` - application dependencies.
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
