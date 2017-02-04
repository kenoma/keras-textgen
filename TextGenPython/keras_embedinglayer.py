import glob
import codecs
import re
import nltk
import logging
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Embedding,Merge,LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import datetime
from natasha import Combinator, DEFAULT_GRAMMARS


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer("russian")
combinator = Combinator(DEFAULT_GRAMMARS)

def ngrams(word, n):
    output = []
    tmp = ''
    for i in range(len(word)):
        tmp +=word[i]
        if (i+1) % n == 0:
            output.append(tmp)
            tmp=''
    if tmp is not '':
        output.append(tmp)
    return output

def review_to_tokens(review):
    #matches = combinator.extract(review)
    #for grammar, tokens in combinator.resolve_matches(matches):
        #print("Правило:", grammar)
    #    for token in tokens:
    #        review = review.replace('%s' % token.value,'%s' %
    #        (type(grammar).__name__.upper()))

    review = review.lower()\
    .replace('\r',' ')\
    .replace('\n',' ')\
    .replace('«','')\
    .replace('»','')\
    .replace('"','')\
    .replace('(','')\
    .replace(')','')\
    .replace('{','')\
    .replace('}','')\
    .replace('[','')\
    .replace(']','')\
    .replace('\\','')\
    .replace('/','')\
    .replace('|','')\
    .replace('~','')\
    .replace('*','')\
    .replace('&','')\
    .replace('^','')\
    .replace('%','')\
    .replace('$','')\
    .replace('#','')\
    .replace('@','')\
    .replace('`','')\
    .replace('...', ' TD ')\
    .replace('…',' TD ')\
    .replace('!',' EM ')\
    .replace(u'.',u' SE ')\
    .replace(',',' CS ')\
    .replace(':',' LS ')\
    .replace(';',' DS ')\
    .replace('?',' QM ')\
    #.replace('brand', 'BRAND')\
    #.replace('date', ' DATE')\
    #.replace('event', 'EVENT')\
    #.replace('geo', 'GEO')\
    #.replace('money', 'MONEY')\
    #.replace('organisation', 'ORGANIZATION')\
    #.replace('person', 'PERSON')\
    
    review = re.sub("\W+"," ", review, re.U)
    retval = []
    for a in list(filter(None, review.split(' '))):
        if a!='TD' and a!='EM' and a!='SE' and a!='CS' and a!='LS' and a!='DS' and a!='QM':
            retval.append(' ')
            retval.extend(ngrams(a, 2))
        else:
            retval.append(a)

    return retval#[stemmer.stem(w) for w in list(filter(None, review.split(' ')))]#
def review_to_sentences(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            s = review_to_tokens(raw_sentence)
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
    #try:
    with codecs.open(filename, "r",encoding='utf-8', errors='strict') as fdata:
        odata = fdata.read()
        sentences+= review_to_sentences(odata)
    #except:
    #    print("Failed")
    words = []
for sentence in sentences:
    words.extend(sentence)

with open("input.txt", "w", encoding="utf-8") as log:
    log.write(''.join(words))
    
vocabular = sorted(list(set(words)))
word_to_indices = dict((c, i) for i, c in enumerate(vocabular))
indices_to_word = dict((i, c) for i, c in enumerate(vocabular))
np.save('words_to_indices.npy', word_to_indices)
np.save('indices_to_words.npy', indices_to_word)

print("Parsing sentences from training set")
print("Sentences got:", len(sentences))

vocab_size = len(vocabular)
print('Vocabular size:', vocab_size)
num_features = vocab_size
context = 15
BATCH_SIZE = 128
dropout = 0.3
hidden_variables = int(round(num_features + dropout * num_features)) + 1

print("LSTM model")

print('Build model...')
#model = Sequential()
#model.add(Embedding(1+vocab_size, num_features, input_length=context))
#model.add(LSTM(200, activation="tanh",
#return_sequences=True))#input_shape=(context, num_features),
#model.add(LSTM(150, activation="tanh"))
#model.add(Dense(vocab_size))
#model.add(Activation('softmax'))
#optimizer = RMSprop(lr=0.01)
#model.compile(loss='categorical_crossentropy', optimizer="adagrad",
#metrics=['accuracy'])

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=num_features, input_length=context))
model.add(LSTM(hidden_variables, return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(hidden_variables, return_sequences=False))
model.add(Dropout(dropout))
model.add(Dense(vocab_size))
model.add(Activation('softmax')) 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#right_branch = Sequential()
#right_branch.add(Dense(vocab_size, input_dim=vocab_size))
#right_branch.add(Dropout(dropout))
#right_branch.add(Dense(hidden_variables))
#right_branch.add(Dropout(dropout))

#merged = Merge([left_branch, right_branch], mode='concat')

#model = Sequential()
#model.add(merged)
#model.add(Dense(vocab_size))
#model.add(Activation('softmax')) 
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#plt.legend(['train'], loc='upper left')
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

def generate_batch_data(data, batch_size):
    while 1:       
        print()
        print('Full moon')
        print()
        p = 0
        while p < len(data) - context - batch_size:
            lx = np.zeros((batch_size, context), dtype=np.int)
            #rx = np.zeros((batch_size, vocab_size), dtype=np.int)
            y = np.zeros((batch_size, vocab_size), dtype=np.float)
            for n in range(batch_size):

                for i in range(context):
                    lx[n, i] = word_to_indices[data[p + i]]
                    #rx[n, word_to_indices[data[p + i]]] = 1
                    
                y[n, word_to_indices[data[p + context]]] = 1
                p += 1

            yield (lx, y)


# train the model, output generated text after each iteration
for iteration in range(1, 46 * 2):
    print("Started:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print('-' * 50)
    print('Iteration', iteration)
    my_generator = generate_batch_data(words, BATCH_SIZE)
    history = model.fit_generator(my_generator, samples_per_epoch = BATCH_SIZE * 500, nb_epoch = 50, verbose=1, callbacks=callbacks_list, nb_worker=1)
    print("Finished:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
   
    start_index = random.randint(0, len(words))
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("Divercity: ",diversity)
    
        sentence = words[start_index: start_index + context]
        generated = ' '.join(sentence) + "|"
        print()
        for i in range(250):
            lx = np.zeros((1, context))
            #rx = np.zeros((1, vocab_size))
            for t, word in enumerate(sentence):
                lx[0, t] = word_to_indices[word]
                #rx[0,vocab_size] = 0
                
            preds = model.predict(lx, verbose=0)[0]
            new_word = indices_to_word[sample(preds,diversity)]
            sentence.append(new_word)
            del sentence[0]
            generated += new_word
            generated = generated.replace('BRAND',u'Google')\
                                .replace('DATE',u'вчера')\
                                .replace('EVENT',u'корпоратив')\
                                .replace('GEO',u'Москва')\
                                .replace('MONEY',u'10 т.р.')\
                                .replace('ORGANIZATION',u'Рога и копыта')\
                                .replace('PERSON',u'Вася')\
                                .replace(' TD','...')\
                                .replace(' EM','!')\
                                .replace(' SE','.')\
                                .replace(' CS',',')\
                                .replace(' LS',':')\
                                .replace(' DS',';')\
                                .replace(' QM','?')\

        with open("output.txt", "a", encoding="utf-8") as log:
            log.write("Started: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
            log.write(u'Embedded Epoch %d\t' % iteration)
            log.write(generated)
            log.write('\r')
            log.write('\r')

