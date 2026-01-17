import re
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    text = text.lower()
    return text


docs = [
    "Python is great for data science!",
    "Data science uses Python and machine learning.",
    "Regex helps clean text before TF-IDF."
]

clean_docs = [preprocess(doc) for doc in docs]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(clean_docs)

print("Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())