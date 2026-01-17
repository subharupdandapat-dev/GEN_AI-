import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')

st.title("📝 Text Tokenization App")

text = st.text_area("Enter your text:")

if st.button("Remove stopwords & punctuation"):
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    filtered_tokens = [
        word for word in tokens
        if word.lower() not in stop_words and word not in punctuation
    ]

    removed_count = len(tokens) - len(filtered_tokens)

    st.subheader("Filtered Tokens")
    st.write(filtered_tokens)

    st.info(f"Removed {removed_count} stopwords/punctuation tokens")