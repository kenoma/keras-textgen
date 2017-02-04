import gensim
import glob
import codecs
import re
import nltk
import logging
import numpy as np
import random
import sys

def review_to_wordlist(review, remove_stopwords=False):
    review_text = re.sub("[^a-zA-Zа-яёА-ЯЁ0-9]"," ", review, re.U)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    return sentences

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = []
labels = []

for filename in glob.glob("D:\projects\TextGenPython\software\*.txt"):
    print(filename)
    with codecs.open(filename, "r", encoding='windows-1251', errors='ignore') as fdata:
        sentences += review_to_sentences(text, tokenizer)

print("Parsing sentences from training set")
print("Sentences got:", len(sentences))
print("d2v model starting")

num_features = 100    # Word vector dimensionality                      
min_word_count = 1   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

wmodel = gensim.models.Doc2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)
wmodel.init_sims(replace=True)
wmodel.save("d2v_%dfeatures_%dminwords_%dcontext.w2v"%(num_features,min_word_count,context))
print("Vocabula has %d words"%len(wmodel.index2word))

