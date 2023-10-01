import random
import json
import torch
import telebot
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from autocorrect import Speller
spell = Speller(lang='ru')

bot = telebot.TeleBot("6602521080:AAHzVn3CtxPzxBR7QEI8mneNJWfibhnqX7c")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "HR"
@bot.message_handler(content_types=['text'])
def chat_bot_generate_message(message):
    print(spell(message.text.strip()))
    v1 = tokenize(spell(message.text.strip()))
    X = bag_of_words(v1, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                text = f"{random.choice(intent['responses'])}"
                bot.send_message(message.chat.id, f"Ваш запрос: {spell(message.text.strip())} \n\n" + text)
    else:
        text = f"Простите, у меня недостаточно информации. Перефразируйте свой вопрос"
        bot.send_message(message.chat.id, text)
print("starting bot")
bot.infinity_polling()