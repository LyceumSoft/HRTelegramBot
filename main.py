import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка предобученных данных
with open("base.md", "r", encoding="utf-8") as file:
    text = file.read()
sentences = nltk.sent_tokenize(text)

def preprocess_text(text):
    text = text.lower() 
    text = "".join([char for char in text if char not in string.punctuation])
    return text

# Предобработка и TF-IDF векторизация текста
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words="english")
tfidf_matrix = vectorizer.fit_transform([preprocess_text(sentence) for sentence in sentences])

# Функция для получения ответа с более длинным текстом
def get_response(question):
    question = preprocess_text(question)
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, tfidf_matrix)
    most_similar_sentence_indices = np.argsort(similarities[0])[-3:][::-1]  # Выберем три наиболее похожих предложения
    response = "\n".join([sentences[idx] for idx in most_similar_sentence_indices])
    return response

while True:
    user_question = input("Ваш вопрос (или 'exit' для выхода): ")
    if user_question.lower() == "exit":
        break
    response = get_response(user_question)
    print("Ответ:")
    print(response)
