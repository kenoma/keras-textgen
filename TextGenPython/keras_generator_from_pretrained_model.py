import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import codecs

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


char_to_int = np.load('char_indices.npy').item()
int_to_char = np.load('indices_char.npy').item()
maxlen = 40
divercity = 0.6
print('Build model...')

model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, len(char_to_int)), return_sequences=True))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(len(char_to_int)))
model.add(Activation('softmax'))


filename = "keras_charbychar_00-1.4874.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

generated = ''

sentence = input("Enter seed string: ")
sentence = (sentence * maxlen)[:maxlen]

for i in range(5000):
    x = np.zeros((1, maxlen, len(char_to_int)))
    for t, char in enumerate(sentence):
        x[0, t, char_to_int[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, divercity)
    next_char = int_to_char[next_index]
    generated += next_char
    sentence = sentence[1:] + next_char
            
print(generated)