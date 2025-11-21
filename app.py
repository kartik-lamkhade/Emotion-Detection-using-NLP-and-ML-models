import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import json

model = load_model("pre_model1.h5", compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load saved tokenizer
with open("tokenizer.json") as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

st.set_page_config(page_title="Emotion Detection", page_icon="💬", layout="centered")

st.markdown("<h1 class='title'>💬 Emotion Detection using NLP</h1>", unsafe_allow_html=True)

text = st.text_input("✍️ Enter Text", placeholder="Type something like 'I am feeling great today!'")

if st.button("🔍 Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text!")
    else:
        seq = tokenizer.texts_to_sequences([text.lower()])
        padded = pad_sequences(seq, maxlen=100, padding='post')

        preds = model.predict(padded)
        pred_index = np.argmax(preds)

        labels = ['anger 😡','fear 😱','joy 😊','love ❤️','sadness 😢','surprise 😮']

        st.success(f"### Predicted emotion: {labels[pred_index]}")
