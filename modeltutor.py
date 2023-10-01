import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Загрузка данных из файла intents.json
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# Проходим по каждому предложению в шаблонах намерений
for intent in intents['intents']:
    tag = intent['tag']
    # Добавляем в список тегов
    tags.append(tag)
    for pattern in intent['patterns']:
        # Токенизируем каждое слово в предложении
        w = tokenize(pattern)
        # Добавляем в список слов
        all_words.extend(w)
        # Добавляем в пару xy
        xy.append((w, tag))

# Проводим стемминг и приводим к нижнему регистру каждое слово
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Удаляем дубликаты и сортируем
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "шаблонов")
print(len(tags), "тегов:", tags)
print(len(all_words), "уникальных слов:", all_words)

# Создаем обучающие данные
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: мешок слов для каждого предложения
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss требует только метки классов, не one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Гиперпараметры
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # Поддерживаем индексацию, чтобы можно было получить i-й образец dataset[i]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # Можем вызвать len(dataset), чтобы получить размер
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Обучение модели
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Прямой проход
        outputs = model(words)
        # если бы y было one-hot, мы должны были бы применить
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Обратный проход и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Эпоха [{epoch+1}/{num_epochs}], Потеря: {loss.item():.4f}')


print(f'Финальная потеря: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Обучение завершено. Файл сохранен как {FILE}')
