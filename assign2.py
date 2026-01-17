import streamlit as st
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download resources
nltk.download('punkt')
nltk.download('stopwords')

st.title("🧹 Clean My Text")

text = st.text_area("Enter your text:")

lower = st.checkbox("Convert to lowercase")
remove_punct = st.checkbox("Remove punctuation")
remove_num = st.checkbox("Remove numbers")
remove_stop = st.checkbox("Remove stopwords")

if st.button("Clean Text"):
    result = text

    if lower:
        result = result.lower()

    if remove_punct:
        result = ''.join(ch for ch in result if ch not in string.punctuation)

    if remove_num:
        result = ''.join(ch for ch in result if not ch.isdigit())

    if remove_stop:
        words = word_tokenize(result)
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
        result = ' '.join(words)

    st.subheader("✅ Cleaned Text:")
    st.write(result)