import logging
import os
import codecs
import glob
import re
import warnings
import nltk
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import word2vec

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words




data_set_filenames = sorted(glob.glob("data/*.*"))
print("books file names", data_set_filenames)
data_raw = u""
for data_filename in data_set_filenames:
    print("Reading '{0}'...".format(data_filename))
    with codecs.open(data_filename, "r", "utf-8") as data_file:
        data_raw += data_file.read()
    print("Book is now {0} characters long".format(len(data_raw)))
    print()
    data_raw = data_raw.lower()

#configfiles = glob.glob(r'C:\Users\sam\Desktop\*\*.txt')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(data_raw)

print("data to sents")
data_sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        data_sentences.append(sentence_to_wordlist(raw_sentence))




logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


nltk_corpus_word2vec_model = word2vec.Word2Vec(data_sentences, min_count=5, size=200, workers=4)

#print(compine_sents)

if not os.path.exists("trained_model"):
    os.makedirs("trained_model")


nltk_corpus_word2vec_model.save(os.path.join("trained_model", "corpus2vec.w2v"))
print(nltk_corpus_word2vec_model.most_similar(["lord"], topn=20))




