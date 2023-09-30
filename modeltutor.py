import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Загрузка предобученной модели BERT
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Загрузка текста из файла base.md
with open("base.md", "r", encoding="utf-8") as file:
    text = file.read()

# Ваши вопросы и ответы
questions = [
    "Расскажи о компании",
    "Кто мы?",
    "Я только что устроился и это моя первая неделя",
]

answers = [
    "Когда-то давным-давно, году эдак в 2005-м, мы решили, что хотим что-то изменить в этой огромной и такой разнообразной стране. Хотим создавать улучшенную версию будущего благодаря IT-продуктам для государства и бизнеса. И… слава великой силе, Ктулху и Одину! У нас это получилось! И продолжает получаться! В те времена многие из нас еще не были знакомы и работали в разных компаниях. Но в глубине души нас объединяло желание сделать этот мир хоть чуточку, но удобнее.",
    "МЫ —это сотрудники Smart Consulting! И если ты новичок в «Смартах»_(да-да, именно так называют нас наши клиенты)_, то добро пожаловать в компанию! Главное достояние Smart Consulting — это люди и их вклад в развитие нашей с тобой компании!",
    "*Поздравляем! Ты прошел собеседование! Заполнил все необходимые бумаги, подписал кучу документов, каких-то согласий и стал частью**  **«**Смартов**»!** **Добро пожаловать в нашу команду!** _Если ты прошел собеседование лет так 10 назад, то твои действия__ —_ _встать, станцевать, улыбнуться, вспомнить первый день, офигеть от количества изменений и продолжить менять мир",
]

while True:
    user_question = input("Задайте ваш вопрос (или 'exit' для выхода): ")
    if user_question.lower() == "exit":
        break

    # Токенизация вопроса и текста
    inputs = tokenizer(user_question, text, return_tensors="pt")

    # Получение оценок начала и конца ответа от модели
    start_scores, end_scores = model(**inputs)

    # Нахождение начала и конца ответа на основе оценок
    answer_start = torch.argmax(start_scores, dim=1).item()
    answer_end = torch.argmax(end_scores, dim=1).item() + 1

    # Получение фактически обрезанных токенов
    input_ids = inputs["input_ids"][0].tolist()  # Преобразовать в список
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    truncated_tokens = tokens[answer_start:answer_end]

    # Преобразование и вывод ответа
    answer_text = tokenizer.convert_tokens_to_string(truncated_tokens)

    # Оценка уверенности модели в ответе
    confidence = (start_scores.max().item() + end_scores.max().item()) / 2

    print("Ответ:")
    if confidence < 0.5:
        print("Модель не уверена в ответе.")
    else:
        print(answer_text)
    print(f"Уверенность: {confidence:.2f}")
