import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from keras.utils import pad_sequences

model = load_model("pre_model1.h5", compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tokenizer = joblib.load("tokenizer.pkl")   # <-- Load real tokenizer here

st.set_page_config(page_title="Emotion Detection", page_icon="💬", layout="centered")

st.markdown("<h1 class='title'>💬 Emotion Detection using NLP</h1>", unsafe_allow_html=True)

st.write("Enter a sentence below and let the model detect the emotion!")

text = st.text_input("✍️ Enter Text", placeholder="Type something like 'I am feeling great today!'")

if st.button("🔍 Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text!")
    else:
        text_seq = tokenizer.texts_to_sequences([text.lower()])
        text_pad = pad_sequences(text_seq, padding='post', maxlen=100)

        pre = model.predict(text_pad)
        pred_index = np.argmax(pre)

        labels = ['anger 😡','fear 😱','joy 😊','love ❤️','sadness 😢','surprise 😮']

        st.success(f"### Predicted emotion: {labels[pred_index]}")
