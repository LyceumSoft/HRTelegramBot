import torch
import torch.nn as nn
import torch.optim as optim
word_to_index = {}
index_to_word = {}

with open('base.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    paragraphs = text.split('\n\n')

dataset = []
for paragraph in paragraphs:
    sentences = paragraph.split('. ')
    if len(sentences) > 1:
        question = sentences[0]
        answer = '. '.join(sentences[1:])
        dataset.append((question, answer))
class SimpleQA(nn.Module):
    def __init__(self):
        super(SimpleQA, self).__init__()
        self.embedding = nn.Embedding(10000, 128)
        self.lstm = nn.LSTM(128, 128)
        self.fc = nn.Linear(128, 10000)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
model = SimpleQA()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    for question, answer in dataset:
        question_indices = [word_to_index[word] for word in question.split()]
        answer_indices = [word_to_index[word] for word in answer.split()]
        question_tensor = torch.LongTensor(question_indices)
        answer_tensor = torch.LongTensor(answer_indices)
        optimizer.zero_grad()
        outputs = model(question_tensor)
        loss = criterion(outputs.view(-1, outputs.size(-1)), answer_tensor)
        loss.backward()
        optimizer.step()
while True:
    user_question = input('Введите ваш вопрос (или "выход" для завершения): ')
    if user_question.lower() == 'выход':
        break

    user_question_indices = [word_to_index[word] for word in user_question.split()]
    user_question_tensor = torch.LongTensor(user_question_indices)

    model.eval()
    with torch.no_grad():
        output_indices = model(user_question_tensor).argmax(dim=-1).numpy()
        predicted_answer = ' '.join([index_to_word[index] for index in output_indices])

    print('Ответ:', predicted_answer)
