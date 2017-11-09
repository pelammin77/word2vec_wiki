import logging
import os
from nltk import tokenize
import multiprocessing
from nltk.corpus import stopwords


def remove_stops(data):
    stop_words = set(stopwords.words("english"))
    list_without_stops = []
    for w in data:
        if w not in stop_words:
            list_without_stops.append(w)
    return list_without_stops


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
raw_text = open('wiki.test.raw', encoding="utf8").read()



words = [ ]
tok = tokenize.word_tokenize(raw_text)
tok = remove_stops(tok)
words.append(tok)
print(words)
#print(toks[:50])


from gensim.models import word2vec as w2v
#model = word2vec.Word2Vec(words, iter=10, min_count=1, size=300, workers=4)

num_features = 300
# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers =  multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1

text2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

text2vec.build_vocab(words)
#print(len(words))

print("Corpus size", text2vec.corpus_count)
print("Model iter size", text2vec.iter)



text2vec.train(words, total_examples=text2vec.corpus_count, epochs=text2vec.iter)




# print(model['The quick brown fox jumped over the lazy dog'])
print(text2vec.most_similar('actor'))
##Saving the model
if not os.path.exists("trained"):
    os.makedirs("trained")
text2vec.save(os.path.join("trained", "text2vec.w2v"))





