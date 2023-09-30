import json
import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext
AIAPI_KEY = ""
TOKEN = ""
with open('config.json', 'r', encoding='utf-8') as f:
    text = json.load(f)
    AIAPI_KEY = text['AI_API_KEY']
    TOKEN = text['BOT_TOKEN']