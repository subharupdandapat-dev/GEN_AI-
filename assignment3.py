import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import string
import spacy
from nltk.stem import PorterStemmer

# Downloads
nltk.download('punkt')

# Load spacy model
nlp = spacy.load("en_core_web_sm")

st.title("🍕 Root Word Cloud Generator")

text = st.text_area("Enter your text : ")
method = st.radio("Choose method:", ["Stemming", "Lemmatization"])

if text:
    words = nltk.word_tokenize(text)
    ps = PorterStemmer()
    roots = []

    for word in words:
        word = word.lower()
        if word not in string.punctuation:
            if method == "Stemming":
                roots.append(ps.stem(word))
            else:
                roots.append(nlp(word)[0].lemma_)

    freq = " ".join(roots)

    wc = WordCloud(
        width=600,
        height=300,
        background_color="white"
    ).generate(freq)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)