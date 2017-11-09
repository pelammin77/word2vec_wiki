import gensim.models.word2vec as w2v
import os

text2vec = w2v.Word2Vec.load(os.path.join("trained", "text2vec.w2v"))
print("Corpus size", text2vec.corpus_count)
print("Model iter size", text2vec.iter)

#print(text2vec.vocabulary)
print( text2vec.most_similar("Graph minors A survey"))
