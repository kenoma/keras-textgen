from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout,Embedding, Merge, LSTM, GRU
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback
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
word_to_indices = np.load('cap_words_to_indices.npy').item()
indices_to_word = np.load('cap_indices_to_words.npy').item()

vocab_size = len(word_to_indices)
print('Vocabular size:', vocab_size)
context = 20
BATCH_SIZE = 1
dropout = 0.0
num_features = 600
hidden_variables = num_features

print('Build model...')

main_input = Input(shape=(context,), dtype='int32', name='main_input')
x = Embedding(input_dim=vocab_size, output_dim=hidden_variables, input_length=context)(main_input)
x = LSTM(hidden_variables, return_sequences=True, dropout_W=dropout, dropout_U=dropout)(x)
x = LSTM(hidden_variables, return_sequences=False, dropout_W=dropout, dropout_U=dropout)(x)

main_output = Dense(vocab_size,activation='softmax', name='m')(x)
capitalization_output = Dense(2, activation='softmax', name='c')(x)

model = Model(input=[main_input], output=[main_output, capitalization_output])
model.load_weights(u"D:\\projects\\TextGenPython\\TextGenPython\\cap_epoch_9_model.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')

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
                
        preds = model.predict(tx, verbose=0)
            
        new_token = indices_to_word[sample(preds[0][0], diversity)]
        sentence.append(new_token)
        del sentence[0]
        new_token = new_token.replace('_','')
        generated += new_token if preds[1][0][0]>0.9 else new_token.capitalize()
        sys.stdout.write(new_token)
        i+=1
        if i > 1500 and new_token == '.' :
            proceed = False


    elapsed_time = time.time() - start_time
    sys.stdout.write('Elapsed %s'%elapsed_time)

    generated = txt_checker.correct_text(generated).strip()
    
    with open("output.txt", "a", encoding="utf-8") as log:
        log.write(generated)
        log.write('\r')
        log.write('\r')
        log.write('\r')

    print(generated)
