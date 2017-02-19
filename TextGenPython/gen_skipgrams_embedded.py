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
import itertools
import copy

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
context = 20
BATCH_SIZE = 128
dropout = 0.1
num_features = 800
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

model.load_weights(u"D:\\projects\\TextGenPython\\TextGenPython\\epoch_10_model.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

with codecs.open("programs.txt", "r",encoding='utf-8', errors='strict') as f:
    content = f.readlines()
content = [x.strip() for x in content] 

for inp in content:
    #inp = input("Please enter kwds: ")
    init_sentence = [(a[0] if a[0] in word_to_indices else ' ') for a in proc.sentence_to_tokens(inp, 2)]
    init_sentence = [' '] * (context - len(init_sentence)) + init_sentence[-context:]
    sentence  = []
    for b in range(BATCH_SIZE):
        sentence.append(copy.deepcopy(init_sentence))

    #print(inp)
    diversity = 0.6
    print("Divercity: ", diversity)
    bait = ''.join(init_sentence)
    generated = list(itertools.repeat(bait, BATCH_SIZE))
    print()
    start_time = time.time()
    i = 0
    proceed = True
    while proceed:
        tx = np.zeros((BATCH_SIZE, context), dtype=np.int)
        
        for b in range(BATCH_SIZE):
            for t, word in enumerate(sentence[b]):
                tx[b, t] = word_to_indices[word]         
                
        preds = model.predict(tx, verbose=0)
        
        for b in range(BATCH_SIZE):
            new_token = indices_to_word[sample(preds[b], diversity)]
            sentence[b].append(new_token)
            del sentence[b][0]
            generated[b] += new_token
            sys.stdout.write(new_token)

        i+=1
        if i > 1500 and new_token == '.' :
            proceed = False


    elapsed_time = time.time() - start_time

    sgenerated = '\r\n\r\n'.join(generated).replace('_',u'').replace('000newline','\r').replace('000anchor','')
    sgenerated = txt_checker.correct_text(sgenerated).strip()

    with open("gen\\sresult_%s.txt"%(''.join(e for e in inp if e.isalnum())), "a", encoding="utf-8") as log:
        log.write(sgenerated)
        log.write('\r')
        log.write('\r')
        log.write('\r')

    print(sgenerated)
