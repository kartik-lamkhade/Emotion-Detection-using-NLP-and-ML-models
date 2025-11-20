import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
# Load model & vectorizer
model = joblib.load("pre_model1.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Emotion Detection", page_icon="💬", layout="centered")

st.markdown("""
    <style>
        .main {
            background-color: #F5F7FA;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #4A90E2;
        }
        .title {
            text-align: center;
            color: #333333;
            font-size: 40px;
            font-weight: 700;
        }
        .footer {
            text-align:center;
            color: #777777;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>💬 Emotion Detection using NLP</h1>", unsafe_allow_html=True)

st.write("Enter a sentence below and let the model detect the emotion!")

text = st.text_input("✍️ Enter Text", placeholder="Type something like 'I am feeling great today!'")

if st.button("🔍 Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text!")
    else:
        text_list = [text.lower()]
        tokenizer = Tokenizer(num_words=5000) 
        tokenizer.fit_on_texts(text_list)
        text_seq = tokenizer.texts_to_sequences(text_list)
        text_pad = pad_sequences(text_seq, padding='post', maxlen=100)
        pre = model.predict(text_pad)
        pred_index = np.argmax(pre)
        labels = ['anger 😡','fear 😱','joy 😊','love ❤️','sadness 😢','surprise 😮']
        pred_label = labels[pred_index]
        st.success(f"### Predicted emotion: {(pred_label, 'Unknown')}")
st.markdown("<p class='footer'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
