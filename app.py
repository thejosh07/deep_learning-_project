import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stopwords (first time only)
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Text cleaning (SAME as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Load model and tokenizer
model = load_model("model/fake_news_model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200  # MUST match training

# UI
st.title("📰 Fake News Detection")

news = st.text_area("Enter News Article")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Enter some news text")
    else:
        cleaned = clean_text(news)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len)

        prediction = model.predict(padded)[0][0]
        confidence = round(float(prediction) * 100, 2)

        if prediction > 0.5:
            st.success(f"✅ Real News  \nConfidence: {confidence}%")
        else:
            st.error(f"❌ Fake News  \nConfidence: {100 - confidence}%")