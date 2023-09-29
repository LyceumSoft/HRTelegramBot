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
    updater = Updater(TOKEN)

    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            RESUME: [MessageHandler(Filters.text & ~Filters.command, process_resume)],
            RESUME_TEXT: [MessageHandler(Filters.text & ~Filters.command, text)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)

    updater.start_polling()

    updater.idle()

if __name__ == '__main__':
    main()
