import pickle
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

# Load the saved TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as tfidf_file:
    tfidf = pickle.load(tfidf_file)


# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_news(news_text):
    transformed_text = vectorizer.transform([news_text])
    prediction = model.predict(transformed_text)
    return "Fake News" if prediction[0] == 0 else "Real News"

# Streamlit App UI
st.title("📰 Fake News Detector")
st.subheader("Enter a news article below to check if it's real or fake")

# User input
user_input = st.text_area("📝 Paste the news article here:", height=200)

if st.button("Detect Fake News"):
    if user_input.strip():
        # Transform input text
        input_transformed = tfidf.transform([user_input])

        # Predict
        prediction = model.predict(input_transformed)[0]
        proba = model.predict_proba(input_transformed)[0]

        # Display result with confidence score
        if prediction == 1:
            st.success(f"🟢 This is **Real News**! (Confidence: {proba[1] * 100:.2f}%)")
        else:
            st.error(f"🔴 This is **Fake News**! (Confidence: {proba[0] * 100:.2f}%)")
    else:
        st.warning("⚠ Please enter some text.")

# Sidebar with App Info
st.sidebar.header("📌 About This App")
st.sidebar.write("This Fake News Detector is built using **Machine Learning** and **Natural Language Processing (NLP)**.")
st.sidebar.write("Model: **Random Forest + TF-IDF** with **99% accuracy**.")
st.sidebar.write("🛠 Built with **Scikit-learn, Streamlit, and Python**.")

# Footer
st.markdown(
    """
    ---
    🏆 **Developed by AkhilDanday ** |
    """
)
