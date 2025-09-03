import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Load preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

# ---------------- Preprocessing ----------------
def clean_text(text):
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    RE_TAGS = re.compile(r"<[^>]+>")
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE)
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)

    text = re.sub(RE_TAGS, " ", text)
    text = re.sub(RE_ASCII, " ", text)
    text = re.sub(RE_SINGLECHAR, " ", text)
    text = re.sub(RE_WSPACE, " ", text)

    word_tokens = word_tokenize(text.lower())
    words_filtered = [lemmatizer.lemmatize(word) for word in word_tokens if word not in stop_words]
    return " ".join(words_filtered)

# ---------------- Sentiment ----------------
def get_sentiment_vader(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "Positive 😃"
    elif compound <= -0.05:
        return "Negative 😡"
    else:
        return "Neutral 😐"

# ---------------- Load Model ----------------
svm_model = joblib.load("svm_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---------------- Streamlit App ----------------
st.title("📰 Fake News & Sentiment Analyzer")

user_input = st.text_area("Paste a headline or tweet:", height=150)

if st.button("Predict"):
    if user_input.strip() != "":
        # Fake / Real Prediction
        clean_input = clean_text(user_input)
        X_tfidf = tfidf_vectorizer.transform([clean_input])
        fake_real_pred = svm_model.predict(X_tfidf)[0]
        fake_real_label = "Fake 🟥" if fake_real_pred == 0 else "Real 🟩"

        # Sentiment
        sentiment_label = get_sentiment_vader(user_input)

        # Show Results
        st.subheader("Results:")
        st.write(f"**Fake/Real:** {fake_real_label}")
        st.write(f"**Sentiment:** {sentiment_label}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
