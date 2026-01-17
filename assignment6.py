import streamlit as st
import re
st.title("🔍 Regex Pattern Finder")
text = st.text_area("Enter your text:")
removal_url = st.checkbox("Remove URLs")
remove_email = st.checkbox("Remove Email Addresses")
if st.button("Process Text"):
    result = text
    if removal_url:
        result = re.sub(r'http\S+|www\S+|https\S+', '', result)
    if remove_email:
        result = re.sub(r'\S+@\S+', '', result)
    st.subheader("Processed Text")
    st.write(result)