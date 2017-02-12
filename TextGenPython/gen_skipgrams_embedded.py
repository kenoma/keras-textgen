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
import codecs

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
context = 30
BATCH_SIZE = 1
dropout = 0.0
num_features = 1200
hidden_variables = num_features

print('Build model...')
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=hidden_variables, input_length=context))
model.add(LSTM(hidden_variables, return_sequences=True))# input_shape=(context, vocab_size),
model.add(Dropout(dropout))
model.add(LSTM(hidden_variables, return_sequences=False))
model.add(Dropout(dropout))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.load_weights(u"D:\\projects\\TextGenPython\\TextGenPython\\epoch_5_model.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

with codecs.open("programs.txt", "r",encoding='utf-8', errors='strict') as f:
    content = f.readlines()
content = [x.strip() for x in content] 

for inp in content:
    #inp = input("Please enter kwds: ")
    sentence = [(a[0] if a[0] in word_to_indices else ' ') for a in proc.sentence_to_tokens(inp, 2)]
    sentence = [' '] * (context - len(sentence)) + sentence[-context:]
    #print(inp)
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
        sys.stdout.write(new_token)
        i+=1
        if i > 1500 and new_token == '.' :
            proceed = False


    elapsed_time = time.time() - start_time


    generated = generated.replace('_',u'').replace('000newline','\r').replace(' 000anchor ','`')
    generated = txt_checker.correct_text(generated).strip()

    with open("output.txt", "a", encoding="utf-8") as log:
        log.write(generated)
        log.write('\r')
        log.write('\r')
        log.write('\r')

    print(generated)
