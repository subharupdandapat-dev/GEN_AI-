from sklearn.feature_extraction.text import TfidfVectorizer 
 
corpus = ["text one", "text two", "text three"] 
vector = TfidfVectorizer() 
ector = CountVectorizer() 
X = vector.fit_transform(corpus) 
print(X.toarray()) 
print(vector.get_feature_names_out()) 
Y = vector.fit_transform(corpus) 
print(Y.toarray()) 
print(ector.get_feature_names_out()) 