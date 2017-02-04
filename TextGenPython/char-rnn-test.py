#coding:utf-8
import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict
import RNN, codecs, pickle, sys

file = u"D:\\projects\\TextGenPython\\TextGenPython\\train_set.txt"
data= u''
with codecs.open(file, "r",encoding='utf-8', errors='strict') as fdata:
    odata = fdata.read()
    data = odata

chars = list(set(data)) #char vocabulary

data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 0.005
iter = 50
save_freq = 5 #The step (counted by the number of iterations) at which the model is saved to hard disk.
len_of_sample = 100 #The number of characters by sample

print('Compile the model...')

training_x = T.ivector('x_data')
training_y = T.ivector('y_data')

h0 = theano.shared(value=np.zeros((1, hidden_size),dtype='float32'), name='h0')

gru = RNN.GRU(vocab_size, hidden_size)
Probs = gru.build_GRU(training_x, h0) #the t-th line of Probs denote probability distribution of vocabulary in t-time step

target_probs = T.diag(Probs.T[training_y]) #T.diag reture the diagonal of matrix
cost = -T.log(target_probs)
training_cost = T.sum(cost)

def sharedX(value, name=None, borrow=False, dtype=None):
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)

def compute_updates(training_cost, params): #adagrad update
    updates = []
    
    grads = T.grad(training_cost, params)
    grads = OrderedDict(zip(params, grads))
    
    for p, g in grads.items():
        m = sharedX(p.get_value() * 0.)
        m_t = m + g * g
        p_t = p - learning_rate * g / T.sqrt(m_t + 1e-8)
        updates.append((m, m_t))
        updates.append((p, p_t))
    
    return updates

params = gru.get_params()
updates = compute_updates(training_cost, params)

train_model = theano.function(inputs=[training_x, training_y], outputs=[training_cost], updates=updates, on_unused_input='ignore', name="train_fn")
sample_model = gru.sample()
print('Done!')
    
def loadModel(filename):
    load_file = open(filename,'rb')
    param_list = params
    for i in range(len(param_list)): 
        param_list[i].set_value(pickle.load(load_file), borrow = True)
    load_file.close()

def sample(seed_ix, n): #generate a text that contains n characters. 
    out = sample_model(seed_ix, n)
    out = out[0].tolist()
    print(', '.join([str(a) for a in out]))
    essay = ''.join([ix_to_char[i] for i in out])
    return essay

saved_model = u"D:\\projects\\TextGenPython\\TextGenPython\\model45"
loadModel(saved_model)
while True:
    seed = input("Enter seed:")
    out = sample(chars.index(seed[0]), len_of_sample)
    with open("output.txt", "a", encoding="utf-8") as traindata:
        traindata.write(seed)
        traindata.write("|")
        traindata.write(out)
        traindata.write('\r')