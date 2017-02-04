from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Embedding, Merge, LSTM, GRU
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils.data_utils import get_file
import datetime
import time
import numpy as np
import random
import sys
from RussianTextPreprocessing import RussianTextPreprocessing
from TextChecker import TextChecker
from spellcheck_preparation import spellcheck_preparation

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

proc = RussianTextPreprocessing()
txt_checker = TextChecker()
#prep = spellcheck_preparation();
#prep.prepare()
word_to_indices = np.load('words_to_indices.npy').item()
indices_to_word = np.load('indices_to_words.npy').item()

vocab_size = len(word_to_indices)
print('Vocabular size:', vocab_size)
context = 25
dropout = 0.2
num_features = 1000
hidden_variables = int(round(num_features + dropout / 4.0 * num_features)) + 1

print('Build model...')
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=hidden_variables, input_length=context))
model.add(LSTM(hidden_variables, return_sequences=True))
model.add(Dropout(dropout))
model.add(GRU(hidden_variables, return_sequences=False))
model.add(Dropout(dropout))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.load_weights(u"D:\\projects\\TextGenPython\\TextGenPython\\model_48-1.8696.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


inp = input("Please enter kwds: ")
sentence = [(a[0] if a[0] in word_to_indices else ' ') for a in proc.sentence_to_tokens(inp, 2)]
sentence = [' '] * (context - len(sentence)) + sentence[-context:]
print()
diversity = 0.7
print("Divercity: ", diversity)
generated = ''.join(sentence)
print()
start_time = time.time()
i = 0
proceed = True
while proceed:
    tx = np.zeros((1, context), dtype=np.int)

    for t, word in enumerate(sentence):
        tx[0, t] = word_to_indices[word]         
                
    preds = model.predict(tx, verbose=0)[0]
    new_token = indices_to_word[sample(preds,diversity)]
    sentence.append(new_token)
    del sentence[0]
    generated += new_token
    i+=1
    if i > 500 and new_token == 'SE':
        proceed = False


elapsed_time = time.time() - start_time

generated = generated.replace('_',u'')\
                    .replace('TD','...')\
                    .replace('EM','!')\
                    .replace('SE','.')\
                    .replace('CS',',')\
                    .replace('LS',':')\
                    .replace('DS',';')\
                    .replace('QM','?')

generated = generated + '\r\n' + txt_checker.correct_text(generated)

with open("output.txt", "a", encoding="utf-8") as log:
    log.write("Started: %s (elapsed %.2f sec) " % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),elapsed_time))
    log.write(u'Generation, divercity %f:\t' % diversity)
    log.write(generated)
    log.write('\r')
    log.write('\r')

print(generated)
