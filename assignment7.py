from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
corpus = ["dogs are cute and fluppy",
          "cats are cute too",
          "i love my cute dog",]
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X = vectorizer.fit_transform(corpus)
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
for i, row in df.iterrows():
    top_words = row.sort_values(ascending=False).head(2)
    print(top_words)
    print()