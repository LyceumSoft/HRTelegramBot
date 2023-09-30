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
    inputs = tokenizer(user_question, text, return_tensors="pt", max_length=512, truncation="longest_first")
    start_scores, end_scores = model(**inputs)
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    print("Ответ:")
    print(answer)
