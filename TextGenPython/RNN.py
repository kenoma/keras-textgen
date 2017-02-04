#coding:utf-8
import theano
import theano.tensor as T
import numpy as np

def NormalInit(n, m):
    return np.float32(np.random.randn(n, m)*0.01)

def add_to_params(params, new_param):
    params.append(new_param)
    return new_param

class GRU():
    def init_params(self):
        self.Ws_in = add_to_params(self.params, theano.shared(value=NormalInit(self.input_dim, self.sdim), name='Ws_in'+self.name))
        self.Ws_hh = add_to_params(self.params, theano.shared(value=NormalInit(self.sdim, self.sdim), name='Ws_hh'+self.name))
        self.bs_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_hh'+self.name))

        self.Ws_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.input_dim, self.sdim), name='Ws_in_r'+self.name))
        self.Ws_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.input_dim, self.sdim), name='Ws_in_z'+self.name))
        self.Ws_hh_r = add_to_params(self.params, theano.shared(value=NormalInit(self.sdim, self.sdim), name='Ws_hh_r'+self.name))
        self.Ws_hh_z = add_to_params(self.params, theano.shared(value=NormalInit(self.sdim, self.sdim), name='Ws_hh_z'+self.name))
        self.bs_z = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_z'+self.name))
        self.bs_r = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_r'+self.name))
        self.Why = add_to_params(self.params, theano.shared(value=NormalInit(self.sdim, self.input_dim), name='Why'+self.name))
        self.by = add_to_params(self.params, theano.shared(value=np.zeros((self.input_dim,), dtype='float32'), name='by'+self.name))

    def recurrent_fn(self, idx, ht):
        xi = theano.shared(np.zeros((self.input_dim, 1), dtype='float32'),name='xi')
        x = T.set_subtensor(xi[idx], 1.0)
        x = x.T
        
        rs_t = T.nnet.sigmoid(T.dot(x, self.Ws_in_r) + T.dot(ht, self.Ws_hh_r) + self.bs_r)
        zs_t = T.nnet.sigmoid(T.dot(x, self.Ws_in_z) + T.dot(ht, self.Ws_hh_z) + self.bs_z)
        hs_tilde = T.tanh(T.dot(x, self.Ws_in) + T.dot(rs_t * ht, self.Ws_hh) + self.bs_hh)
        hs_update = (np.float32(1.) - zs_t) * ht + zs_t * hs_tilde
        
        ys = T.dot(hs_update, self.Why) + self.by
        ps = T.exp(ys)/T.sum(T.exp(ys))
        ps = ps.flatten()
        
        return hs_update, ps

    def build_GRU(self, training_x, h0):
        _res, _ = theano.scan(self.recurrent_fn, sequences=[training_x], outputs_info=[h0, None])

        Probs = _res[1]
        return Probs
    
    def get_params(self):
        return self.params
        
    def sample(self):
        def recurrent_gn(hs, idx):
            h_t, ps = self.recurrent_fn(idx, hs)
            
            y_i = T.argmax(ps) #using a greedy strategy

            return h_t, y_i

        h_0 = theano.shared(value=np.zeros((1, self.sdim),dtype='float32'), name='h0')

        sod = T.lscalar('sod')
        n = T.lscalar('n')
        [h, y_idx], _ = theano.scan(recurrent_gn,
                                               outputs_info = [h_0, sod],
                                               n_steps = n)

        sample_model = theano.function(inputs=[sod, n], outputs=[y_idx], on_unused_input='ignore')
        
        return sample_model
    
    def __init__(self, vocab_size, hidden_size):
        self.params = []
        self.input_dim = vocab_size
        self.sdim = hidden_size
        self.name = 'GRU'
        self.init_params()