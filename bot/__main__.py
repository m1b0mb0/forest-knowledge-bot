"""Run the Telegram bot with long polling: python -m bot."""

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramAPIError
from aiogram.types import (
    BotCommand,
    BotCommandScopeDefault,
    MenuButtonCommands,
)
from groq import AsyncGroq

from bot.config import load_archive_settings, load_bot_token, load_llm_settings
from bot.handlers import router
from bot.services.drive import DriveImportError, import_archive
from bot.services.llm import ArchiveAssistant
from bot.services.uploads import UploadService
from bot.storage import Store


async def configure_bot(bot: Bot) -> None:
    """Publish the command menu and the description shown in an empty chat."""
    await bot.set_my_commands(
        commands=[
            BotCommand(command="start", description="Як користуватися ботом"),
            BotCommand(
                command="my_documents", description="Мої завантажені документи"
            ),
            BotCommand(
                command="delete", description="Видалити свій документ за ID"
            ),
        ],
        scope=BotCommandScopeDefault(),
    )
    await bot.set_my_description(
        description=(
            "Допомагаю знаходити відповіді в «Лісовому архіві» \n\n"
            "Постав запитання або надішли PDF, DOCX чи TXT, "
            "щоб поповнити спільний архів.\n\n"
            "Команда /start підкаже, з чого почати."
        )
    )
    await bot.set_chat_menu_button(menu_button=MenuButtonCommands())


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
    with ProcessPoolExecutor(
        max_workers=1, mp_context=get_context("spawn"),
    ) as parser_pool:
        async with (
            AsyncGroq(
                api_key=llm_settings.api_key,
                max_retries=0,
                timeout=45.0,
            ) as client,
            Bot(token=token) as bot,
        ):
            assistant = ArchiveAssistant(client, llm_settings.model)
            uploads = UploadService(store, parser_pool)
            try:
                await configure_bot(bot)
                # Clear the backlog after imports and setup are complete.
                # Later updates belong to the new polling run.
                await bot.delete_webhook(drop_pending_updates=True)
            except (TelegramAPIError, TimeoutError):
                logging.error(
                    "Could not prepare Telegram or clear pending updates. "
                    "Polling was not started."
                )
                raise SystemExit(1) from None
            await dispatcher.start_polling(
                bot,
                close_bot_session=False,
                assistant=assistant,
                uploads=uploads,
                store=store,
                pending_questions=pending_questions,
            )


if __name__ == "__main__":
    asyncio.run(main())
