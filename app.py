import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import json

model = load_model("pre_model1.h5", compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load tokenizer.json
with open("tokenizer.json", "r") as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

st.title("💬 Emotion Detection using NLP")

text = st.text_input("Enter text")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Enter something!")
    else:
        seq = tokenizer.texts_to_sequences([text.lower()])
        pad = pad_sequences(seq, maxlen=100, padding='post')

        pred = model.predict(pad)
        idx = np.argmax(pred)

        labels = ['anger 😡','fear 😱','joy 😊','love ❤️','sadness 😢','surprise 😮']

        st.success(f"Predicted emotion: {labels[idx]}")
