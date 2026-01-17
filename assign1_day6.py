import streamlit as st
import nltk
import spacy
import string
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit page setup
st.set_page_config(
    page_title="NLP preprocessing",
    layout="wide"
)

st.title("NLP Preprocessing App")
st.write("Tokenization, Text Cleaning, Stemming, Lemmatization, and Bag of Words")

# Text input
text = st.text_area("Enter text for NLP processing", height=150,
        placeholder="Example: Aman is the HOD of HIT and loves NLP")

# Sidebar options
option = st.sidebar.radio(
    "Select NLP Technique",
    [
        "Tokenization",
        "Text Cleaning",
        "Stemming",
        "Lemmatization",
        "Bag of Words"
    ]
)

# Process button
if st.button("Process Text"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        if option == "Tokenization":
            st.subheader("Tokenization Output")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("Sentence Tokenization")
                sentences = sent_tokenize(text)
                st.write(sentences)

            with col2:
                st.markdown("Word Tokenization")
                words = word_tokenize(text)
                st.write(words)

            with col3:
                st.markdown("Character Tokenization")
                characters = list(text)
                st.write(characters)

        elif option == "Text Cleaning":
            st.subheader("Text Cleaning Output")

            text_lower = text.lower()
            cleaned_text = "".join(ch for ch in text_lower if ch not in string.punctuation and not ch.isdigit())

            doc = nlp(cleaned_text)
            final_words = [token.text for token in doc if not token.is_stop and token.text.strip() != ""]

            st.markdown("Original Text")
            st.write(text)

            st.markdown("Cleaned Text")
            st.write(" ".join(final_words))

        elif option == "Stemming":
            st.subheader("Stemming Output")

            words = word_tokenize(text)

            porter = PorterStemmer()
            lancaster = LancasterStemmer()

            porter_stem = [porter.stem(word) for word in words]
            lancaster_stem = [lancaster.stem(word) for word in words]

            df = pd.DataFrame({
                "Original Word": words,
                "Porter Stemmer": porter_stem,
                "Lancaster Stemmer": lancaster_stem
            })

            st.dataframe(df, use_container_width=True)

        elif option == "Lemmatization":
            st.subheader("Lemmatization using spaCy")

            doc = nlp(text)
            data = [(token.text, token.pos_, token.lemma_) for token in doc]

            df = pd.DataFrame(data, columns=["Word", "POS", "Lemma"])
            st.dataframe(df, use_container_width=True)

        elif option == "Bag of Words":
            st.subheader("Bag of Words Representation")

            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform([text])

            vocab = vectorizer.get_feature_names_out()
            freq = X.toarray()[0]

            df = pd.DataFrame({
                "Word": vocab,
                "Frequency": freq
            }).sort_values(by="Frequency", ascending=False)

            st.markdown("BoW Frequency Table")
            st.dataframe(df, use_container_width=True)

            st.markdown("Word Frequency Distribution (Top 10)")

            top_n = 10
            df_top = df.head(top_n)

            fig, ax = plt.subplots()
            ax.pie(
                df_top["Frequency"],
                labels=df_top["Word"],
                autopct="%1.1f%%",
                startangle=90
            )
            ax.axis("equal")

            st.pyplot(fig)