# app.py

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

# Load model and vectorizer from single file
@st.cache_resource
def load_model():
    data = joblib.load("news-predictor.pkl")
    return data["model"], data["vectorizer"]

model, vectorizer = load_model()

# Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english")).union({'claim', 'secret', 'secretli', 'alleg', 'report', 'say'})

def preprocess(text):
    text = re.sub(r"\W", " ", text.lower())
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="📰 News Predictor", page_icon="🧠")
st.title("🧠 News Predictor")
st.markdown("Enter any news headline or article to predict whether it's **Real** or **Fake**.")

user_input = st.text_area("📝 Enter News Text", height=200)

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter some news text.")
    else:
        clean_text = preprocess(user_input)
        vec = vectorizer.transform([clean_text])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        label = "🟢 Real News" if pred == 1 else "🔴 Fake News"
        confidence = prob[1]*100 if pred == 1 else prob[0]*100

        st.subheader("📣 Prediction:")
        st.markdown(f"**{label}** with **{confidence:.2f}%** confidence.")
        if confidence < 60:
            st.warning("⚠️ This prediction has low confidence.")
