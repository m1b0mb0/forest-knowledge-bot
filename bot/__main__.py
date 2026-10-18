"""Run the Telegram bot with long polling: python -m bot."""

import asyncio
import logging

from aiogram import Bot, Dispatcher

from bot.config import load_bot_token
from bot.handlers import router


async def main() -> None:
    """Configure the bot and poll Telegram until a shutdown signal arrives."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        token = load_bot_token()
    except ValueError as error:
        logging.error("%s", error)
        raise SystemExit(1) from None

    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    async with Bot(token=token) as bot:
        await dispatcher.start_polling(bot, close_bot_session=False)


if __name__ == "__main__":
    asyncio.run(main())
