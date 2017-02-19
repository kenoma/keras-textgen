# -*- coding: utf-8 -*-

import glob
import codecs
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import datetime
import nltk
import re,string

train_data = ''
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

for filename in glob.glob("D:\projects\TextGenPython\software\*.txt"):
    print(filename)
    with codecs.open(filename, "r",encoding='utf-8', errors='strict') as fdata:
        odata = fdata.read()
        train_data += odata

train_data = re.sub(u'[a-zA-Z]', u'', train_data)
entities = []
seq = nltk.word_tokenize(train_data);

for w in range(len(seq)):
    if seq[w][0].isupper() and seq[w-1] not in string.punctuation:
        entities.append(seq[w])

enchanters = list(set([re.escape(a) for a in entities]))
print()
print()

for filename in glob.glob("D:\projects\TextGenPython\TextGenPython\gen\*.txt"):
    print(filename)
    odata = ''
    with codecs.open(filename, "r",encoding='utf-8', errors='strict') as fdata:
        odata = fdata.read()

    for enh in enchanters:
        odata = re.sub('\b%s\b'%enh.lower(), enh,odata)
    
    with codecs.open(filename+'_', "w",encoding='utf-8', errors='strict') as fdata:
        fdata.write(odata)
        