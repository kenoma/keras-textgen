# -*- coding: utf-8 -*-

import glob
import codecs
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Embedding, Merge, LSTM, GRU
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
    #with codecs.open(filename, "r",encoding='utf-8', errors='strict') as fdata:
    with codecs.open(filename, "r",encoding='utf-8', errors='strict') as fdata:
        odata = fdata.read()
        sentences += proc.review_to_sentences(odata, 2)

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
with open("vocabular.txt", "w", encoding="utf-8") as log:
    for voc in vocabular:
        log.write(voc)
        log.write('\r')

context = 30
BATCH_SIZE = 512
dropout = 0.1
num_features = 600
hidden_variables = num_features

def generate_batch_data(data, batch_size):
    while 1:       
        print()
        print('Full moon')
        print()
        p = 0
        while p < len(data) - context - batch_size:
            textual_x = np.zeros((batch_size, context), dtype=np.int)
            y = np.zeros((batch_size, vocab_size), dtype=np.int)
            for batch_i in range(batch_size):
                for i in range(context):
                    textual_x[batch_i, i] = word_to_indices[data[p + i][0]]
                    
                y[batch_i, word_to_indices[data[p + context][0]]] = 1
                p += 1

            yield (textual_x, y)

print('Build model...')
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=hidden_variables, input_length=context))
model.add(LSTM(hidden_variables, return_sequences=True))# input_shape=(context, vocab_size),
model.add(Dropout(dropout))
model.add(LSTM(hidden_variables, return_sequences=False))
model.add(Dropout(dropout))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

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


batch_generator = generate_batch_data(words, BATCH_SIZE)

for iteration in range(1, 46 * 2):
    print("Started:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print('-' * 50)
    print('Iteration', iteration)
    history = model.fit_generator(batch_generator, samples_per_epoch = BATCH_SIZE * 1000, nb_epoch = 10, verbose=1, callbacks=callbacks_list, nb_worker=1)
    print("Finished:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    model.save("epoch_%d_model.hdf5"%iteration)
    start_index = random.randint(0, len(words))
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("Divercity: ",diversity)
        sentence = [a[0] for a in words[start_index: start_index + context]]
        generated = ''.join(sentence)
        print()
        for i in range(500):
            tx = np.zeros((1, context), dtype=np.int)

            for t, word in enumerate(sentence):
                tx[0, t] = word_to_indices[word]         
                
            preds = model.predict(tx, verbose=0)[0]
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

