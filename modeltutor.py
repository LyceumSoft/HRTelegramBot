import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset

# Загрузка предобученной модели и токенизатора
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Загрузка данных для обучения
with open("base.md", "r", encoding="utf-8") as file:
    text = file.read()

# Подготовка данных для обучения
questions = ["Расскажи о компании",   
             "Кто мы?",
             "Я только что устроился и это моя первая неделя",
]
answers = ["Когда-то давным-давно, году эдак в 2005-м, мы решили, что хотим что-то изменить в этой огромной и такой разнообразной стране. Хотим создавать улучшенную версию будущего благодаря IT-продуктам для государства и бизнеса. И… слава великой силе, Ктулху и Одину! У нас это получилось! И продолжает получаться! В те времена многие из нас еще не были знакомы и работали в разных компаниях. Но в глубине души нас объединяло желание сделать этот мир хоть чуточку, но удобнее.",
           "МЫ —это сотрудники Smart Consulting! И если ты новичок в «Смартах»_(да-да, именно так называют нас наши клиенты)_, то добро пожаловать в компанию! Главное достояние Smart Consulting — это люди и их вклад в развитие нашей с тобой компании!",
            "*Поздравляем! Ты прошел собеседование! Заполнил все необходимые бумаги, подписал кучу документов, каких-то согласий и стал частью**  **«**Смартов**»!** **Добро пожаловать в нашу команду!** _Если ты прошел собеседование лет так 10 назад, то твои действия__ —_ _встать, станцевать, улыбнуться, вспомнить первый день, офигеть от количества изменений и продолжить менять мир",
] 

input_ids = []
attention_masks = []
start_positions = []
end_positions = []

for question, answer in zip(questions, answers):
    encoded = tokenizer(question, text, return_tensors="pt", max_length=90000, truncation="longest_first", padding=True)
    input_ids.append(encoded["input_ids"])
    attention_masks.append(encoded["attention_mask"])
    answer_start = text.find(answer)
    answer_end = answer_start + len(answer)
    start_positions.append(answer_start)
    end_positions.append(answer_end)

input_ids = torch.stack(input_ids)
attention_masks = torch.stack(attention_masks)
start_positions = torch.tensor(start_positions)
end_positions = torch.tensor(end_positions)

# Создание DataLoader для обучения
dataset = TensorDataset(input_ids, attention_masks, start_positions, end_positions)
batch_size = 8  # Размер пакета, который можно настроить
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Определение параметров обучения
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader))

# Обучение модели
num_epochs = 3  # Количество эпох, которое можно настроить
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, start_positions, end_positions = batch
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.2f}")

# Сохранение обученной модели
model.save_pretrained("trained_hr_model")
