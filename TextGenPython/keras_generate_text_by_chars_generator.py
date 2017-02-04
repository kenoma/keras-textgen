# -*- coding: utf-8 -*-

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import glob
import codecs
import string
import matplotlib.pyplot as plt

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

text = u'';
for filename in glob.glob("D:\projects\TextGenPython\software\*.txt"):
    print(filename)
    try:
        with codecs.open(filename, "r",encoding='utf-8', errors='strict') as fdata:
            odata = fdata.read()
            text += odata.replace('\r',' ')\
                    .replace('\n',' ')\
                    .replace('«','')\
                    .replace('»','')\
                    .replace('"','')\
                    .replace('(','')\
                    .replace(')','')\
                    .replace('{','')\
                    .replace('}','')\
                    .replace('[','')\
                    .replace(']','')\
                    .replace('\\','')\
                    .replace('/','')\
                    .replace('|','')\
                    .replace('~','')\
                    .replace('*','')\
                    .replace('&','')\
                    .replace('^','')\
                    .replace('%','')\
                    .replace('$','')\
                    .replace('#','')\
                    .replace('@','')\
                    .replace('`','')
    except:
        print("Failed")

print('corpus length:', len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
BATCH_SIZE = 256
print('total chars:', vocab_size)

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
np.save('char_indices.npy', char_indices)
np.save('indices_char.npy', indices_char)
maxlen = 50
dropout = 0.3

def generate_batch_data(data, batch_size):
    while 1:
        p = 0
        while p < len(data) - maxlen - batch_size:
            x = np.zeros((batch_size, maxlen, vocab_size), dtype=np.bool)
            y = np.zeros((batch_size, vocab_size), dtype=np.bool)
            for n in range(batch_size):
                for i in range(maxlen):
                    x[n, i, char_indices[data[p + i]]] = 1
                y[n, char_indices[data[p + maxlen]]] = 1
                p += 1

            yield (x, y)

print('Build model...')
model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, len(char_indices)), return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(256))
model.add(Dropout(dropout))
model.add(Dense(len(char_indices)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

plt.ion()
plt.show()
filepath = "preciese_{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
class LossHistory(Callback):
    losses = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        plt.clf()
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('batch')
        plt.plot(self.losses)
        plt.show()
        plt.pause(0.001)


history = LossHistory()
callbacks_list = [checkpoint, history]

for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    #model.fit(X, y, batch_size=120, nb_epoch=5, callbacks=callbacks_list)
    my_generator = generate_batch_data(text, BATCH_SIZE)
    model.fit_generator(my_generator, samples_per_epoch = BATCH_SIZE * 10000, nb_epoch = 1, verbose=1, callbacks=callbacks_list, nb_worker=1)

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