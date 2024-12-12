#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob
import re
from flask import Flask, request, render_template

# ----------- Étape 1 : Prétraitement des données -----------#
def preprocess_text(text):
    negations = {
        "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "haven't": "have not", "hasn't": "has not",
        "hadn't": "had not", "won't": "will not", "wouldn't": "would not",
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
        "mightn't": "might not", "mustn't": "must not", "not good": "bad"
    }
    for key, value in negations.items():
        text = text.replace(key, value)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Charger les données
csv_file = "c:/Users/Asus/OneDrive/Bureau/NLP/tripadvisor_hotel_reviews.csv"  # Assurez-vous que ce fichier est présent dans le même répertoire
df = pd.read_csv(csv_file)

# Ajouter une colonne pour les données prétraitées
df["processed_review"] = df["Review"].apply(preprocess_text)
df["sentiment_score"] = df["Review"].apply(lambda x: TextBlob(x).sentiment.polarity)

# ----------- Étape 2 : Préparer les données -----------#
vocab_size = 10000
max_sequence_length = 100

# Tokenization et padding
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(df["processed_review"])

sequences = tokenizer.texts_to_sequences(df["processed_review"])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="post")

# Préparation des données X et y
X = padded_sequences
y = df["Rating"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------- Étape 3 : Charger ou entraîner le modèle -----------#
model_path = "C:/Users/Asus/OneDrive/Bureau/NLP/sentiment_model.h5"
if not os.path.exists(model_path):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sequence_length),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="linear")
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # Sauvegarder le modèle
    model.save(model_path)
else:
    model = load_model(model_path)

# ----------- Étape 4 : Création de l'application Flask -----------#
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_review = request.form['review']
        processed_review = preprocess_text(user_review)
        sentiment_score = TextBlob(user_review).sentiment.polarity

        # Préparation des données
        sequence = tokenizer.texts_to_sequences([processed_review])
        padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding="post")

        # Prédiction
        predicted_score = model.predict(padded_sequence)
        result = {
            "processed_review": processed_review,
            "sentiment_score": round(sentiment_score, 2),
            "predicted_score": round(predicted_score[0][0], 2)
        }
        return render_template('result.html', result=result)

# Lancer l'application
if __name__ == '__main__':
    app.run(debug=True)

