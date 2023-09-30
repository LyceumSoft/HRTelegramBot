import os
import openai
def askss(v1):
    
    askss(input())
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Машина — это устройство, спроектированное для передвижения.",
    "Python - это высокоуровневый язык программирования.",
    "Нейронные сети используются для решения задач машинного обучения.",
    "Какой язык программирования легче всего изучить?",
]
questions = [
    "Что такое машина?",
    "Чем характерен Python?",
    "Какие задачи решают нейронные сети?",
    "Какой язык программирования легче всего изучить?",
]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
question_vectors = tfidf_vectorizer.transform(questions)
cosine_similarities = cosine_similarity(question_vectors, tfidf_matrix)
most_similar_indexes = np.argmax(cosine_similarities, axis=1)
for i, question in enumerate(questions):
    most_similar_doc = documents[most_similar_indexes[i]]
    print(f"Вопрос: {question}")
    print(f"Ответ: {most_similar_doc}")
    print("=" * 40)
