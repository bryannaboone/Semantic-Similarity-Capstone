print("downloading")
import gensim.downloader as api

print("finish download")

# Download a pre-trained word2vec (trained on Google News data)
w2v_model = api.load("word2vec-google-news-300")

# Compute similarity between words
def compute_similarity(model, word1, word2):
    try:
        return model.similarity(word1, word2)
    except KeyError:
        return 0  # return 0 if either of the words is not in the vocabulary

print("computing similarity")
print(compute_similarity(w2v_model, "cat", "dog"))

