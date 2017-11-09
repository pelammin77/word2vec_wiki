from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import gensim.models.word2vec as w2v
from nltk.corpus import stopwords
import multiprocessing
import os
import re
raw_data = open('wiki.test.raw', encoding="utf8").read()
raw_data = raw_data.lower()


raw_sents = sent_tokenize(raw_data)



def make_sentences_to_word_list(raw):


    clean_text = " ".join(re.findall(r"[a-zA-Z]+", raw))
   # clean_text =  remove_stops(clean_text)

    return clean_text

def remove_stops(data):
    stop_words = set(stopwords.words("english"))
    text_without_stops = []
    for w in data:
        if w not in stop_words:
            text_without_stops.append(w)

    return text_without_stops


tok = set(word_tokenize(raw_data))
sentences = []
for raw_sentence in raw_sents:
    if len(raw_sentence) > 0:
        sentences.append(make_sentences_to_word_list(raw_sentence))

print(sentences)
words = remove_stops(tok)
#print( "word count:", len(words))
#print(len(raw_sents))
clean_text = ' '.join(words)
clean_text = sent_tokenize(clean_text)
#print(clean_text)
num_features = 300
# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

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

text2vec.build_vocab(sentences)
#print(len(words))

print("Corpus size", text2vec.corpus_count)
print("Model iter size", text2vec.iter)



text2vec.train(words, total_examples=text2vec.corpus_count, epochs=text2vec.iter)

##Saving the model
if not os.path.exists("trained"):
    os.makedirs("trained")
text2vec.save(os.path.join("trained", "text2vec.w2v"))





