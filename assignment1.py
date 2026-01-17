import streamlit as st
from nltk.tokenize import word_tokenize,sent_tokenize
import nltk


nltk.download("punkt")
st.title("text tokenization app")
text = st.text_area("enter tour text:")
if st.button("Analyze"):
    if(next.strip()):
        sentence= sent_tokenize(text)
        word= word_tokenize(text)
        st.write("###Result")
        st.write("sentence count:",len(sentence))
        st.write("first word",word[0] if word else "")
        st.write("last word",word[-1] if word else "")
        st.write("###sentence")
        for i,sent in enumerate(sentence, start=1):
            st.write(f"{i}. {sent}")
    else:
        st.warning("please enter some text analyze")