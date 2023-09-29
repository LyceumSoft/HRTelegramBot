import json
AIAPI_KEY = ""
BOT_TOKEN = ""
with open('config.json', 'r', encoding='utf-8') as f:
    text = json.load(f)
    AIAPI_KEY = text['AI_API_KEY']
    BOT_TOKEN = text['BOT_TOKEN']