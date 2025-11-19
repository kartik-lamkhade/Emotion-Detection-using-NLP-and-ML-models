import streamlit as st
import joblib

model = joblib.load("model_loge.pkl")
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

Text = st.text_input("✍️ Enter Text", placeholder="Type something like 'I am feeling great today!'")

if st.button("🔍 Predict Emotion"):
    if Text.strip() == "":
        st.warning("Please enter some text!")
    else:
        x_new = vectorizer.transform([Text])
        pre = model.predict(x_new)[0]

        emotions = {
            1: "😢 Sadness",
            2: "😡 Anger",
            3: "❤️ Love",
            4: "😮 Surprise",
            5: "😱 Fear",
            6: "😊 Joy"
        }

        st.success(f"### Predicted emotion: {emotions.get(pre, 'Unknown')}")


st.markdown("<p class='footer'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
