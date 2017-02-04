import gensim
import glob
import codecs
from bs4 import BeautifulSoup
import re
import nltk
import logging
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import datetime

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_wordlist(review):
    review = review.replace(u'.',u' SE ').replace(',',' CS ').replace(':',' LS ').replace(';',' DS ')
    review = re.sub("\W+"," ", review, re.U)
    review = review.replace(' CS', ', ').replace(' LS',': ').replace(' DS','; ').replace(u' SE',u'. ')
    syllabies = []
    
    for i in range(len(review)):
        if(i < len(review) - 1):
            syllabies.append(u'%s%s' % (review[i],review[i + 1]))
    return syllabies

def review_to_sentences(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            s = review_to_wordlist(raw_sentence)
            sentences.append(s)
    return sentences

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

sentences = []

for filename in glob.glob("D:\projects\TextGenPython\software\*.txt"):
    print(filename)
    try:
        with codecs.open(filename, "r",encoding='utf-8', errors='strict') as fdata:
            odata = fdata.read().lower()
            sentences+= review_to_sentences(odata)
    except:
        print("Failed")
    #if len(sentences) > 1000:
    #    break
words = []
for sentence in sentences:
    words.extend(sentence)

vocabular = sorted(list(set(words)))
word_to_indices = dict((c, i) for i, c in enumerate(vocabular))
indices_to_word = dict((i, c) for i, c in enumerate(vocabular))
np.save('words_to_indices.npy', word_to_indices)
np.save('indices_to_words.npy', indices_to_word)

print("Parsing sentences from training set")
print("Sentences got:", len(sentences))
print("w2v model starting")

vocab_size = len(vocabular)
print('Vocabular size:', vocab_size)
num_features = vocab_size    # Word vector dimensionality
min_word_count = 1   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 50          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words
BATCH_SIZE = 128

wmodel = gensim.models.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)
wmodel.init_sims(replace=True)
wmodel.save("%dfeatures_%dminwords_%dcontext.w2v" % (num_features,min_word_count,context))
print("w2v model done")
print("Vocabula has %d words" % len(wmodel.index2word))

print("LSTM model")

print('Build model...')
model = Sequential()
model.add(LSTM(400, input_shape=(context, num_features), activation="tanh", return_sequences=True))
model.add(LSTM(256, activation="tanh"))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

#optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer="adagrad", metrics=['accuracy'])

filepath = "model_{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

def generate_batch_data(data, batch_size):
    while 1:       
        print()
        print('Full moon')
        print()
        p = 0
        while p < len(data) - context - batch_size:
            x = np.zeros((batch_size, context, num_features), dtype=np.float)
            y = np.zeros((batch_size, vocab_size), dtype=np.float)
            for n in range(batch_size):
                for i in range(context):
                    #vec = wmodel[data[p + i]]
                    #for i in range(maxlen):
                    x[n, i, word_to_indices[data[p + i]]] = 1
                    #for h in range(0, num_features):
                    #    x[n, i, h] = vec[h]

                y[n, word_to_indices[data[p + context]]] = 1
                p += 1

            yield (x, y)

# train the model, output generated text after each iteration
for iteration in range(1, 46*2):
    print("Started:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print('-' * 50)
    print('Iteration', iteration)
    my_generator = generate_batch_data(words, BATCH_SIZE)
    history = model.fit_generator(my_generator, samples_per_epoch = BATCH_SIZE * 100, nb_epoch = 46, verbose=1, callbacks=callbacks_list, nb_worker=1)
    print("Finished:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    # summarize history for accuracy
    #print(history.history.keys())
    #plt.plot(history.history['acc'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(['train'], loc='upper left')
    #plt.draw()

    start_index = random.randint(0, len(words))
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("Divercity: ",diversity)
    
        sentence = words[start_index: start_index + context]
        generated = "".join([a[1] for a in sentence]) + "|"
        print()
        for i in range(250):
            x = np.zeros((1, context, num_features))
            for t, word in enumerate(sentence):
                x[0, t, word_to_indices[word]] = 1
                #vec = wmodel[word]
                #for h in range(0, num_features):
                #    x[0, t, h] = vec[h]

            preds = model.predict(x, verbose=0)[0]
            new_word = indices_to_word[sample(preds,diversity)]
            sentence.append(new_word)
            del sentence[0]
            generated += new_word[1]

        
        with open("output.txt", "a", encoding="utf-8") as log:
            log.write("Started: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
            log.write(u'w2v+LTSM Epoch %d\t' % iteration)
            log.write(generated)
            log.write('\r')

