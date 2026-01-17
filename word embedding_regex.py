import re
import numpy as np
import pandas as pd

text = "Aman is the HOD of HIT and loves NLP. NLP is fun!"


tokens = re.findall(r"\b\w+\b", text.lower())
print("Tokens:", tokens)

vocab = sorted(set(tokens))
word_to_index = {word: idx for idx, word in enumerate(vocab)}
print("Vocabulary:", word_to_index)


def one_hot_encode(tokens, vocab):
    vectors = []
    for token in tokens:
        vector = np.zeros(len(vocab))
        vector[word_to_index[token]] = 1
        vectors.append(vector)
    return np.array(vectors)

embeddings = one_hot_encode(tokens, vocab)


df = pd.DataFrame(embeddings, columns=vocab, index=tokens)
print("\nWord Embedding Matrix:")
print(df)