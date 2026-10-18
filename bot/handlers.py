"""Telegram message handlers."""

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message

router = Router(name=__name__)


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    await message.answer("Вітаю! Я — «Лісовий архів».")


@router.message(F.text)
async def text_handler(message: Message) -> None:
    await message.answer(message.text)
