from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import glob
import codecs
import string
import re, nltk
import textwrap

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer("russian")

def review_to_wordlist(review):
    review_text = re.sub("[^a-zA-Zа-яёА-ЯЁ0-9]"," ", review, re.U)
    words = nltk.word_tokenize(review_text)
    return words#[stemmer.stem(w) for w in words]

def review_to_skipgramm(review, chunk=2):
    review_text = re.sub("[^a-zA-Zа-яёА-ЯЁ0-9]"," ", review, re.U)
    words = nltk.word_tokenize(review_text)
    retval = textwrap.wrap("_".join(words), chunk)
    retval.append("LE");
    return [a.replace("_"," ") for a in retval]

def review_to_sentences(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            s = review_to_skipgramm(raw_sentence)
            sentences += s
    return sentences

def weightedChoice(weights):
    cs = np.cumsum(weights) 
    idx = sum(cs < np.random.rand()) 
    return idx

text = []

for filename in glob.glob("D:\projects\TextGenPython\software\*.txt"):
    print(filename)
    try:
        with codecs.open(filename, "r",encoding='windows-1251', errors='strict') as fdata:
            odata = fdata.read().lower()
            text.extend(review_to_sentences(odata))
    except:
        print("Failed")
    break

print('sentences:',  len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
np.save('char_indices.npy', char_indices)
np.save('indices_char.npy', indices_char)

maxlen = 20
step = 2
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

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

filepath="keras_charbychar_{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1, callbacks=callbacks_list)

    #try:
    start_index = random.randint(0, len(text) - maxlen - 1)

    print()
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += "".join(sentence)
    print()

    for i in range(100):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = weightedChoice(preds)
        next_char = indices_char[next_index]

        generated = generated + next_char
        del sentence[0]
        sentence.append(next_char)

    with open("output.txt", "a", encoding="utf-8") as myfile:
        myfile.write(u'Epoch %d'%iteration)
        myfile.write(generated)
        myfile.write('\r\n')
        
    #except:
    #    print("Error happens")