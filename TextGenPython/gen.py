import gensim
import glob
import codecs
from bs4 import BeautifulSoup
import re
import nltk
import logging
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

wmodel = gensim.models.Word2Vec.load(u"D:\\projects\\TextGenPython\\TextGenPython\\100features_1minwords_10context.w2v")

model = Sequential()
maxlen = 10
chars = 100
model.add(LSTM(128, input_shape=(maxlen, chars)))
model.add(Dense(chars,activation="tanh"))
model.add(Activation('tanh'))
model.load_weights(u"D:\\projects\\TextGenPython\\TextGenPython\\model_00-14.9627.hdf5")
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

sentence = input("Please enter kwds: ").split()
generated = " ".join(sentence)
#print(u'----- Generating with seed: "' + generated + '"')
print()
for i in range(20):
    x = np.zeros((1, maxlen, chars))
    for t, word in enumerate(sentence):
        vec = wmodel[word]
        for h in range(0, chars):
            x[0, t, h] = (vec[h]+1)/2

    preds = model.predict(x, verbose=0)[0]
    new_word = wmodel.similar_by_vector(preds,topn = 1)[0][0]
    sentence.append(new_word)
    del sentence[0]
    generated +=" "+ new_word

print(generated)
