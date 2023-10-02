import random  # Импорт модуля random для генерации случайных чисел
import json  # Импорт модуля json для работы с JSON-файлами
import torch  # Импорт библиотеки PyTorch
import telebot  # Импорт библиотеки Telebot для создания бота в Telegram
from model import NeuralNet  # Импорт определенной вами модели нейронной сети из другого файла
from nltk_utils import bag_of_words, tokenize  # Импорт необходимых функций для обработки текста из других файлов
from autocorrect import Speller  # Импорт библиотеки для автокоррекции текста
spell = Speller(lang='ru')  # Создание экземпляра автокорректора для русского языка
# Инициализация бота с использованием токена
bot = telebot.TeleBot("6602521080:AAHzVn3CtxPzxBR7QEI8mneNJWfibhnqX7c")

# Определение устройства для обучения (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка намерений (intentions) из файла JSON
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

# Создание экземпляра модели нейронной сети и загрузка её состояния
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Имя вашего бота
bot_name = "HR"

# Обработчик сообщений от пользователя
@bot.message_handler(content_types=['text'])
def chat_bot_generate_message(message):
    print(spell(message.text.strip()))  # Вывод на экран автокорректированного текста
    v1 = tokenize(spell(message.text.strip()))  # Токенизация и автокоррекция текста сообщения
    X = bag_of_words(v1, all_words)  # Преобразование токенизированного текста в мешок слов
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.85:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                text = f"{random.choice(intent['responses'])}"
                bot.send_message(message.chat.id, f"Ваш запрос: {spell(message.text.strip())} \n\n" + text)
                return
    else:
        text = f"Простите, у меня недостаточно информации. Перефразируйте свой вопрос"
        bot.send_message(message.chat.id, text)
        return

# Запуск бота
print("starting bot")
bot.infinity_polling()
