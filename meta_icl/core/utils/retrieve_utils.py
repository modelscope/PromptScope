import bm25s
import Stemmer  # optional: for stemming
#
# # Create your corpus here
# corpus = [
#     "a cat is a feline and likes to purr",
#     "a dog is the human's best friend and loves to play",
#     "a bird is a beautiful animal that can fly",
#     "a fish is a creature that lives in water and swims",
# ]
#
# # optional: create a stemmer
stemmer = Stemmer.Stemmer("english")
# stemmer = Stemmer.Stemmer()
#
# # Tokenize the corpus and only keep the ids (faster and saves memory)
# corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
#
# # Create the BM25 model and index the corpus
# retriever = bm25s.BM25()
# retriever.index(corpus_tokens)
#
# # Query the corpus
query = "does the fish purr like a cat?"
query_tokens = bm25s.tokenize(query, stemmer=stemmer)
#
# # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
# results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)
#
# for i in range(results.shape[1]):
#     doc, score = results[0, i], scores[0, i]
#     print(f"Rank {i+1} (score: {score:.2f}): {doc}")
#
# # You can save the arrays to a directory...
# retriever.save("animal_index_bm25")
#
# # You can save the corpus along with the model
# retriever.save("animal_index_bm25", corpus=corpus)

# ...and load them when you need them
import bm25s
reloaded_retriever = bm25s.BM25.load("data/icl_bm25_demo/animal_index_bm25", load_corpus=True)
results, scores = reloaded_retriever.retrieve(query_tokens, k=2)
for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")
# set load_corpus=False if you don't need the corpus