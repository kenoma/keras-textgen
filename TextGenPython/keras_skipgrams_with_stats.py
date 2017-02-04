# -*- coding: utf-8 -*-

import glob
import codecs
#import logging
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Embedding, Merge, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import datetime
from RussianTextPreprocessing import RussianTextPreprocessing


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

proc = RussianTextPreprocessing()
sentences = []

for filename in glob.glob("D:\projects\TextGenPython\software\*.txt"):
    print(filename)
    with codecs.open(filename, "r",encoding='utf-8', errors='strict') as fdata:
        odata = fdata.read()
        sentences += proc.review_to_sentences(odata)

words = []
for sentence in sentences:
    words.extend(sentence)

vocabular = sorted(list(set([a[0] for a in words])))
word_to_indices = dict((c, i) for i, c in enumerate(vocabular))
indices_to_word = dict((i, c) for i, c in enumerate(vocabular))
np.save('words_to_indices.npy', word_to_indices)
np.save('indices_to_words.npy', indices_to_word)

print("Parsing sentences from training set")
print("Sentences got:", len(sentences))

vocab_size = len(vocabular)
print('Vocabular size:', vocab_size)
water_dim = max([a[1] for a in words])
senlen_dim = max([a[2] for a in words])
context = 25
BATCH_SIZE = 64
dropout = 0.3
hidden_variables = int(round(vocab_size + dropout * vocab_size)) + 1

def generate_batch_data(data, batch_size):
    while 1:       
        print()
        print('Full moon')
        print()
        p = 0
        while p < len(data) - context - batch_size:
            textual_x = np.zeros((batch_size, context, vocab_size), dtype=np.int)
            water_x = np.zeros((batch_size, water_dim), dtype=np.int)
            senlen_x = np.zeros((batch_size, senlen_dim), dtype=np.int)
            y = np.zeros((batch_size, vocab_size), dtype=np.int)
            for batch_i in range(batch_size):
                for context_i in range(context):
                    textual_x[batch_i, context_i, word_to_indices[data[p + context_i][0]]] = 1
                water_x[batch_i,  data[p + context - 1][1]] = 1
                senlen_x[batch_i, data[p + context - 1][2]] = 1
                    
                y[batch_i, word_to_indices[data[p + context][0]]] = 1
                p += 1

            yield ([textual_x, water_x, senlen_x], y)

#tup= [a for a in generate_batch_data(words,100)]
print('Build model...')
textual_branch = Sequential()
textual_branch.add(LSTM(vocab_size, input_shape=(context, vocab_size), return_sequences=True))
textual_branch.add(Dropout(dropout))
textual_branch.add(LSTM(hidden_variables, return_sequences=False))
textual_branch.add(Dropout(dropout))

water_branch = Sequential()
water_branch.add(Dense(hidden_variables, input_dim=water_dim))
water_branch.add(Dropout(dropout))
water_branch.add(Dense(vocab_size))

senlen_branch = Sequential()
senlen_branch.add(Dense(hidden_variables, input_dim=senlen_dim))
senlen_branch.add(Dropout(dropout))
senlen_branch.add(Dense(vocab_size))

merged = Merge([textual_branch, water_branch,senlen_branch], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

plt.ion()
plt.show()

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

filepath = "model_{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history = LossHistory()
callbacks_list = [checkpoint, history]

# train the model, output generated text after each iteration
for iteration in range(1, 46 * 2):
    print("Started:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print('-' * 50)
    print('Iteration', iteration)
    my_generator = generate_batch_data(words, BATCH_SIZE)
    history = model.fit_generator(my_generator, samples_per_epoch = BATCH_SIZE * 2500, nb_epoch = 50, verbose=1, callbacks=callbacks_list, nb_worker=1)
    print("Finished:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
   
    start_index = random.randint(0, len(words))
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("Divercity: ",diversity)
    
        sentence = [a[0] for a in words[start_index: start_index + context]]
        generated = ''.join(sentence)
        print()
        for i in range(500):
            tx = np.zeros((1, context, vocab_size))
            wx = np.zeros((1, water_dim))
            sx = np.zeros((1, senlen_dim))

            for t, word in enumerate(sentence):
                tx[0, t,word_to_indices[word]] = 1
            sw, sl = proc.count_stats(generated, water_dim - 1, senlen_dim - 1)
            wx[0, sw] = 1
            sx[0, sl] = 1
                
            preds = model.predict([tx,wx,sx], verbose=0)[0]
            new_token = indices_to_word[sample(preds,diversity)]
            sentence.append(new_token)
            del sentence[0]
            generated += new_token

        generated = generated.replace('_',u'')\
                            .replace('TD','...')\
                            .replace('EM','!')\
                            .replace('SE','.')\
                            .replace('CS',',')\
                            .replace('LS',':')\
                            .replace('DS',';')\
                            .replace('QM','?')\

        with open("output.txt", "a", encoding="utf-8") as log:
            log.write("Started: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
            log.write(u'Embedded Epoch %d\t' % iteration)
            log.write(generated)
            log.write('\r')
            log.write('\r')

