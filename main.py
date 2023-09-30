import torch
from transformers import BertTokenizer, BertForQuestionAnswering

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

with open("base.md", "r", encoding="utf-8") as file:
    text = file.read()

while True:
    user_question = input("Задайте ваш вопрос (или 'exit' для выхода): ")
    if user_question.lower() == "exit":
        break

    # Токенизация вопроса и текста
    inputs = tokenizer(user_question, text, return_tensors="pt", max_length=512, truncation=True)

    # Получение оценок начала и конца ответа от модели
    start_scores, end_scores = model(**inputs)

    # Нахождение начала и конца ответа на основе оценок
    answer_start = torch.argmax(start_scores).item()
    answer_end = torch.argmax(end_scores).item() + 1

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
