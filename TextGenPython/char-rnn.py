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
idx_of_begin = chars.index(u'В') #begin character
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
    

def dumpModel(filename):
    save_file = open(filename, 'wb')  # this will overwrite current contents
    for param in params:
        pickle.dump(param.get_value(borrow = True),save_file,-1)
    save_file.close()

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

p = 0
n = 0
loss = 0
i = 1

print('Begin training...')
while(i<=iter):
    if p+seq_length+1 >= len(data): 
        h0.set_value(np.zeros((1, hidden_size),dtype='float32'))
        p = 0 # go from start of data
        print('the iter is:',i)
        print('the loss is:',loss)
        print('average loss: ',loss/n)
        
        if i%save_freq == 0:
            print('save model:iter = %i' % i)
            dumpModel('model'+str(i)) #save the model
            out = sample(idx_of_begin, len_of_sample) #generate a text that contains len_of_sample characters
            with open("output.txt", "a", encoding="utf-8") as traindata:
                traindata.write(u'Iteration %d |\r' % i)
                traindata.write(out)
                traindata.write('\r')
            
        loss = 0
        n = 0
        i += 1
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
    
    loss_ = train_model(inputs,targets)
    loss += loss_[0]
    
    n += 1
    p += seq_length


saved_model = u"D:\\projects\\TextGenPython\\TextGenPython\\model5"
loadModel(saved_model)
out = sample(chars.index(u'В'), len_of_sample)