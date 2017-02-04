# -*- coding: utf-8 -*-
'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import glob
import codecs
import string
import re
import nltk

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer("russian")

def review_to_wordlist(review):
    #review_text = re.sub("[^a-zA-Zа-яёА-ЯЁ]"," ", review, re.U)
    words = nltk.word_tokenize(review)
    return words#[stemmer.stem(w) for w in words]

def review_to_sentences(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            s = review_to_wordlist(raw_sentence)
            sentences += s
    return sentences

def get_training_set(content, maxlen=40, step=4, sample_size=1000000):
    start_index = random.randint(0, len(content) - sample_size - 1)
    text = content[start_index: start_index + sample_size]
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))
    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return X,y

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

content = []
for filename in glob.glob("D:\projects\TextGenPython\software\*.txt"):
    print(filename)
    try:
        with codecs.open(filename, "r",encoding='utf-8', errors='strict') as fdata:
            odata = fdata.read()
            content.extend(review_to_sentences(odata))
    except:
        print("Failed")

text = " ".join(content)
with open("train_set.txt", "w", encoding="utf-8") as traindata:
    traindata.write(text)

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
np.save('char_indices.npy', char_indices)
np.save('indices_char.npy', indices_char)

maxlen = 50
print('Build model...')
model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, len(char_indices)), return_sequences=True))
model.add(LSTM(256))
model.add(Dense(len(char_indices)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

filepath = "preciese_{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    X,y = get_training_set(text, maxlen, 1, 1000000)
    model.fit(X, y, batch_size=120, nb_epoch=5, callbacks=callbacks_list)

    try:
        start_index = random.randint(0, len(text) - maxlen - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence + "|"
            for i in range(400):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]
                generated += next_char
                sentence = sentence[1:] + next_char
            
            with open("output.txt", "a", encoding="utf-8") as traindata:
                traindata.write(u'Epoch %d | %f |\r' % (iteration, diversity))
                traindata.write(generated)
                traindata.write('\r')
    except:
        print("Error happens")