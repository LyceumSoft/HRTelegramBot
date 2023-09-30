import tensorflow as tf
from tensorflow import keras
from tensorflow import Tokenizer
from tensorflow import pad_sequences

with open("base.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

questions = []
answers = []

for i in range(0, len(lines), 2):
    questions.append(lines[i].strip())
    answers.append(lines[i + 1].strip())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)

questions_sequences = tokenizer.texts_to_sequences(questions)
answers_sequences = tokenizer.texts_to_sequences(answers)

max_seq_length = max(len(seq) for seq in questions_sequences + answers_sequences)
questions_sequences = pad_sequences(questions_sequences, maxlen=max_seq_length, padding="post")
answers_sequences = pad_sequences(answers_sequences, maxlen=max_seq_length, padding="post")

model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_seq_length),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.Dense(len(tokenizer.word_index) + 1, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

answers_one_hot = keras.utils.to_categorical(answers_sequences, num_classes=len(tokenizer.word_index) + 1)

model.fit(questions_sequences, answers_one_hot, epochs=10, batch_size=64)

new_question = "Ваш новый вопрос?"
new_question_sequence = tokenizer.texts_to_sequences([new_question])
new_question_sequence = pad_sequences(new_question_sequence, maxlen=max_seq_length, padding="post")
predicted_answer_sequence = model.predict(new_question_sequence)
predicted_answer = tokenizer.sequences_to_texts([predicted_answer_sequence])[0]
print(predicted_answer)
