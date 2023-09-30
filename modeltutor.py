import torch
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from torch.utils.data import DataLoader, TensorDataset

texts = [
    "Новый вопрос 1?",
    "Новый вопрос 2?",
    "Новый вопрос 3?",
]
labels = [
    "Правильный ответ 1",
    "Правильный ответ 2",
    "Правильный ответ 3",
]
# Загрузка предобученной модели BERT и токенизатора
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Чтение новых данных из файла
with open("base.md", "r", encoding="utf-8") as file:
    text = file.read()

# Здесь предполагается, что вы подготовите новые вопросы и ответы из файла base.md
# Замените texts и labels на ваши новые данные

# Токенизация и предобработка новых данных
input_ids = []
attention_masks = []

for text in texts:
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )
    input_ids.append(encoded_text['input_ids'])
    attention_masks.append(encoded_text['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Создание нового датасета и загрузчика данных
dataset = TensorDataset(input_ids, attention_masks, labels)
batch_size = 4  # Выберите размер пакета, подходящий для ваших ресурсов
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Определение функции потерь и оптимизатора
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

# Обучение модели на новых данных
num_epochs = 5  # Выберите количество эпох
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids_batch, attention_masks_batch, labels_batch = batch
        optimizer.zero_grad()
        outputs = model(input_ids_batch, attention_mask=attention_masks_batch)
        loss = criterion(outputs.logits, labels_batch)
        loss.backward()
        optimizer.step()

# Сохранение новой обученной модели
model.save_pretrained("trained_bert_model")
tokenizer.save_pretrained("trained_bert_model")
