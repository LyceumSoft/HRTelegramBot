import json
import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
AIAPI_KEY = ""
TOKEN = ""
with open('config.json', 'r', encoding='utf-8') as f:
    text = json.load(f)
    AIAPI_KEY = text['AI_API_KEY']
    TOKEN = text['BOT_TOKEN']

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Я ваш бот. Отправьте мне сообщение, и я его отвечу.')

def echo(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    update.message.reply_text(f'Вы написали: {user_message}')

def main() -> None:
    updater = Updater(token=TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()