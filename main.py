import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("base.txt", "r", encoding="utf-8") as file:
    text = file.read()

sentences = nltk.sent_tokenize(text)

def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words="english")
tfidf_matrix = vectorizer.fit_transform([preprocess_text(sentence) for sentence in sentences])

def get_response(question):
    question = preprocess_text(question)
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, tfidf_matrix)
    most_similar_sentence_index = np.argmax(similarities)
    return sentences[most_similar_sentence_index]

user_question = input()
response = get_response(user_question)
print(response)
