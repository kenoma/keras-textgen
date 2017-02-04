#!/usr/bin/env python
from __future__ import print_function
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint, Callback
import numpy as np
import random,string
import glob
import codecs
import sys
import matplotlib.pyplot as plt

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text=''

for filename in glob.glob("D:\projects\TextGenPython\software\*.txt"):
    print(filename)
    with codecs.open(filename, "r",encoding='utf-8', errors='strict') as fdata:
        odata = fdata.read()
        text+=odata

print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 10 # might be much easier with 3 or 2...
nbatch = 32

print('Vectorization...')
X = np.zeros((len(text), len(chars)), dtype=np.bool)
for t, char in enumerate(text):
    X[t, char_indices[char]] = 1


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, stateful=True, return_sequences=False, batch_input_shape=(nbatch, maxlen, len(chars))))
model.add(Dense(256, activation='relu'))
model.add(RepeatVector(maxlen))
model.add(LSTM(512, stateful=True, return_sequences=True))
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# start with a small sample that increases each iteration
numsamps = len(X)/100
numsampinc = len(X)/100
plt.ion()
plt.show()
filepath = "stateful_{epoch:02d}-{loss:.4f}.hdf5"
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


# train the model, output generated text after each iteration
for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    # get consecutive sequences for each "lane" by breaking the dataset
    # into 'nbatch' regions
    # X[0] X[s] X[2*s] ... X[(nbatch-1)*s] X[1] X[s+1] X[2*s+1] ...
    numsamps = min(len(X), numsamps)
    numsamps += numsampinc

    stride = int((numsamps-maxlen)/nbatch)
    sampsperbatch = int(stride/maxlen)
    totalsamps = sampsperbatch*nbatch
    XXs = np.zeros((totalsamps, maxlen, len(chars)), dtype=np.bool)
    YYs = np.zeros((totalsamps, maxlen, len(chars)), dtype=np.bool)
    for i in range(0,sampsperbatch):
      for j in range(0,nbatch):
        ofs = j*stride+i*maxlen
        XX = X[ofs:ofs+maxlen]
        YY = X[ofs+maxlen:ofs+maxlen*2]
        XXs[i*nbatch+j] = XX
        YYs[i*nbatch+j] = YY
    
    model.reset_states()
    model.fit(XXs, YYs, batch_size=nbatch, nb_epoch=3, shuffle=False, callbacks=callbacks_list)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence+'|'
        print('----- Generating with seed: "' + sentence + '"')
        model.reset_states()
        for i in range(400):
            x = np.zeros((nbatch, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            # just get prediction from 1st batch
            preds_seq = model.predict(x, verbose=0)[0]
            
            # don't know if this is correct since each successive sample
            # doesn't take into account the prior...
            next_indices = [sample(preds, diversity) for preds in preds_seq]
            next_chars = ''.join([indices_char[next_index] for next_index in next_indices])

            generated += next_chars
            sentence = next_chars

        with open("output.txt", "a", encoding="utf-8") as traindata:
            traindata.write(u'Epoch %d | %f |\r' % (iteration, diversity))
            traindata.write(generated)
            traindata.write('\r')
        print()