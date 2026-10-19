"""Run the Telegram bot with long polling: python -m bot."""

import asyncio
import logging

from aiogram import Bot, Dispatcher
from groq import AsyncGroq

from bot.config import load_archive_settings, load_bot_token, load_llm_settings
from bot.handlers import router
from bot.services.drive import DriveImportError, import_archive
from bot.services.llm import ArchiveAssistant
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
        llm_settings = load_llm_settings()
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

    pending_questions: dict[tuple[int, int], str] = {}
    logging.info("Groq model: %s", llm_settings.model)
    async with (
        AsyncGroq(
            api_key=llm_settings.api_key,
            max_retries=0,
            timeout=45.0,
        ) as client,
        Bot(token=token) as bot,
    ):
        assistant = ArchiveAssistant(client, llm_settings.model)
        await dispatcher.start_polling(
            bot,
            close_bot_session=False,
            assistant=assistant,
            store=store,
            pending_questions=pending_questions,
        )


if __name__ == "__main__":
    asyncio.run(main())
