import json
import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext
TOKEN = ""
with open('config.json', 'r', encoding='utf-8') as f:
    text = json.load(f)
    TOKEN = text['BOT_TOKEN']
print(TOKEN)
keyword_responses = {
    "привет": "Привет!",
    "компания": "Smart Consulting — это самостоятельная компания.",
    "ответственность": "Ответственность — важная составляющая успеха для «Смартов».",
}