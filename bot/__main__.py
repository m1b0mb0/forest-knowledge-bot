"""Run the Telegram bot with long polling: python -m bot."""

import asyncio
import logging

from aiogram import Bot, Dispatcher

from bot.config import load_archive_settings, load_bot_token
from bot.handlers import router
from bot.services.drive import DriveImportError, import_archive
from bot.storage import Store


async def main() -> None:
    """Configure the bot and poll Telegram until a shutdown signal arrives."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        token = load_bot_token()
        archive_settings = load_archive_settings()
    except ValueError as error:
        logging.error("%s", error)
        raise SystemExit(1) from None

    store = Store(archive_settings.data_dir)
    await asyncio.to_thread(store.initialize)
    try:
        imported = await import_archive(archive_settings, store)
    except DriveImportError as error:
        logging.error("%s", error)
        raise SystemExit(1) from None
    logging.info("Archive import complete: %s documents", imported)

    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    async with Bot(token=token) as bot:
        await dispatcher.start_polling(bot, close_bot_session=False)


if __name__ == "__main__":
    asyncio.run(main())
