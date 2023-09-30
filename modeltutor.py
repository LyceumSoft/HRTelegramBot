import torch
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from torch.utils.data import DataLoader, TensorDataset

# Ваш исходный текст
text = """
Основная роль - = Роль, выполнением обязательств которой сотрудник занимается наибольшую часть времени в течение длительного периода. Основная Роль сотрудника может меняться.
Круг = это объединение в группу нескольких Ролей (собрание Ролей) для реализации своего предназначения;
Член круга, участник круга = это сотрудник, назначенный на Роль внутри Круга или на Роль Лид-линка Дочернего круга;
Дочерний круг = это Круг, входящий (вложенный) в другой Круг;
Родительский круг = вышестоящий Круг по отношению к Дочернему кругу;
Сообщество, Небюджетируемый круг = временное или постоянное собрание Ролей, предназначенное для развития каких-либо идей, общих интересов и проводимое в свободное время;
"""

# Ваши вопросы и ответы
questions = [
    "Что такое основная роль?",
    "Что представляет собой Круг?",
    "Какие роли могут быть в Дочернем круге?",
    "Что такое Родительский круг?",
]

answers = [
    "Основная Роль - это Роль, выполнением обязательств которой сотрудник занимается наибольшую часть времени в течение длительного периода. Основная Роль сотрудника может меняться.",
    "Круг - это объединение в группу нескольких Ролей (собрание Ролей) для реализации своего предназначения;",
    "В Дочернем круге могут быть различные роли, назначенные участникам этого Круга;",
    "Родительский круг - это вышестоящий Круг по отношению к Дочернему кругу;",
]

# Загрузка предобученной модели BERT и токенизатора
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Токенизация и предобработка текста и ответов
text_tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
question_tokens = tokenizer(questions, return_tensors="pt", truncation=True, padding=True)
answer_tokens = tokenizer(answers, return_tensors="pt", truncation=True, padding=True)

# Подготовка обучающих данных
input_ids = text_tokens["input_ids"].repeat(len(questions), 1)
attention_mask = text_tokens["attention_mask"].repeat(len(questions), 1)
start_positions = answer_tokens["input_ids"][:, 1:].clone()  # Начальные позиции ответов
end_positions = answer_tokens["input_ids"][:, :-1].clone()   # Конечные позиции ответов

# Создание датасета
dataset = TensorDataset(input_ids, attention_mask, start_positions, end_positions)

# Определение функции потерь и оптимизатора
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

# Создание загрузчика данных
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Обучение модели
num_epochs = 5
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, start_positions, end_positions = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        start_logits, end_logits = outputs.logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = criterion(start_logits, start_positions) + criterion(end_logits, end_positions)
        loss.backward()
        optimizer.step()

# Сохранение обученной модели
model.save_pretrained("trained_bert_model")
tokenizer.save_pretrained("trained_bert_model")
