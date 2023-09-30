import json
import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler, CallbackContext
AIAPI_KEY = ""
TOKEN = ""
with open('config.json', 'r', encoding='utf-8') as f:
    text = json.load(f)
    AIAPI_KEY = text['AI_API_KEY']
    TOKEN = text['BOT_TOKEN']

def main() -> None:
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher
    start_handler = CommandHandler('start')
    dispatcher.add_handler(start_handler)
if __name__ == '__main__':
    main()
