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
import text_processing

text_data = ''

for train_files in glob.glob("D:\projects\TextGenPython\software\*.txt"):
    print(train_files)

    with codecs.open(train_files, "r",encoding='utf-8', errors='strict') as fdata:
        odata = fdata.read()
        text_data += text_processing.encode(odata)

vocab_size = len(text_processing.char_indices)
print('Vocabular size:', vocab_size)
print('Text length:', len(text_data))

context = 80
BATCH_SIZE = 256
dropout = 0.3
hidden_variables = 512

def generate_batch_data(data, batch_size, context, vocab_size):
    while 1:       
        print()
        print('Restart')
        print()
        p = 0
        while p < len(data) - context - batch_size:
            textual_x = np.zeros((batch_size, context, vocab_size), dtype=np.int)
            y = np.zeros((batch_size, vocab_size), dtype=np.int)
            for batch_i in range(batch_size):
                for context_i in range(context):
                    textual_x[batch_i, context_i, text_processing.char_indices[data[p + context_i]]] = 1
                    
                y[batch_i, text_processing.char_indices[data[p + context]]] = 1
                p += 1

            yield (textual_x, y)

print('Build model...')
model = Sequential()
model.add(LSTM(hidden_variables, input_shape=(context, vocab_size), return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(hidden_variables, return_sequences=False))
model.add(Dropout(dropout))
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

filepath = "c2c_model_{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history = LossHistory()
callbacks_list = [checkpoint, history]

batch_generator = generate_batch_data(text_data, BATCH_SIZE, context, vocab_size)

for iteration in range(1, 1000000):
    print("Started:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print('-' * 50)
    print('Iteration', iteration)
    history = model.fit_generator(batch_generator, steps_per_epoch = 2000, epochs = 10, verbose=1, callbacks=callbacks_list, workers=1)
    print("Finished:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    start_index = random.randint(0, len(text_data))
    model.save("epoch_%d_c2c_model.hdf5"%iteration)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("Divercity: ",diversity)
        sentence = [a for a in text_data[start_index: start_index + context]]
        generated = ''.join(sentence)
        print()
        for i in range(500):
            tx = np.zeros((1, context, vocab_size))

            for t, word in enumerate(sentence):
                tx[0, t, text_processing.char_indices[word]] = 1         
                
            preds = model.predict(tx, verbose=0)[0]
            new_token = text_processing.indices_char[text_processing.sample(preds,diversity)]
            sentence.append(new_token)
            del sentence[0]
            generated += new_token

        generated =''.join(text_processing.decode(generated))

        with open("output.txt", "a", encoding="utf-8") as log:
            log.write("Started: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
            log.write(u'Char2Char Epoch %d\t' % iteration)
            log.write(generated)
            log.write('\r')
            log.write('\r')

