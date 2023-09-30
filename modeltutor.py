import torch
import torch.nn as nn
import torch.optim as optim
import json

# Загрузка данных из JSON-файла
with open('learnbase.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

questions = data["Q"]
answers = data["A"]

# Создание словаря токенов и индекса
vocab = set()
for text in questions + answers:
    vocab.update(text.split())
print(len(vocab), vocab)
word2index = {word: index for index, word in enumerate(vocab)}
index2word = {index: word for word, index in word2index.items()}

# Преобразование текста в индексы
def text_to_indices(text, word2index):
    return [word2index[word] for word in text.split()]

# Создание DataLoader и Batch
from torch.utils.data import DataLoader, TensorDataset

max_sequence_length = max(len(text.split()) for text in answers)
question_indices = [text_to_indices(question, word2index) + [0] * (max_sequence_length - len(question.split())) for question in questions]
answer_indices = [text_to_indices(answer, word2index) + [0] * (max_sequence_length - len(answer.split())) for answer in answers]

question_indices = torch.LongTensor(question_indices)
answer_indices = torch.LongTensor(answer_indices)

dataset = TensorDataset(question_indices, answer_indices)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Создание модели seq2seq
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.h = torch.zeros(1, 1, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output.view(1, -1))
        return output, hidden

# Инициализация модели и оптимизатора
hidden_size = 256
input_size = hidden_size
output_size = len(vocab)
model = Seq2Seq(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Обучение модели
def train(input_tensor, target_tensor, model, optimizer, criterion):
    target_length = target_tensor.size(0)
    optimizer.zero_grad()
    loss = 0
    hidden = torch.zeros(1, input_tensor.size(1), hidden_size)  # Исправьте размерность hidden
    for i in range(input_tensor.size(0)):
        output, hidden = model(input_tensor[i], hidden)
        loss += criterion(output.squeeze(0), target_tensor[i])
    loss.backward()
    optimizer.step()
    return loss.item() / target_length

# Процесс обучения
n_epochs = 100
for epoch in range(n_epochs):
    total_loss = 0
    for batch in dataloader:
        input_tensor, target_tensor = batch
        loss = train(input_tensor, target_tensor, model, optimizer, criterion)
        total_loss += loss
    average_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {average_loss:.4f}")

# Генерация ответа на вопрос
def generate_answer(question, model, word2index, index2word):
    question_indices = torch.LongTensor(text_to_indices(question, word2index))
    hidden = torch.zeros(1, 1, hidden_size)
    output, _ = model(question_indices, hidden)
    output_indices = output.squeeze(0).argmax(dim=1).tolist()
    return indices_to_text(output_indices, index2word)

# Тестирование модели
test_question = input("Введите вопрос--")
generated_answer = generate_answer(test_question, model, word2index, index2word)
print(f"Question: {test_question}")
print(f"Answer: {generated_answer}")