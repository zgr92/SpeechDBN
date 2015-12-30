from rbm import RBM

import theano
import theano.tensor as T

class GBRBM(RBM):
    def __init__(self, input, n_visible=784, n_hidden=500, \
                 W=None, hbias=None, vbias=None, numpy_rng=None, transpose=False,
                 theano_rng=None, weight_decay=0.0002):
            RBM.__init__(self, input=input, n_visible=n_visible, n_hidden=n_hidden, \
                         W=W, hbias=hbias, vbias=vbias, numpy_rng=numpy_rng,
                         theano_rng=theano_rng, weight_decay=weight_decay)

    def free_energy(self, v_sample):            
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = 0.5 * T.dot((v_sample - self.vbias), (v_sample - self.vbias).T)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - T.diagonal(vbias_term)


    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)    
        v1_sample = self.theano_rng.normal(size=v1_mean.shape, avg=v1_mean, std=1.0, dtype=theano.config.floatX) + pre_sigmoid_v1
        return [pre_sigmoid_v1, v1_mean, v1_sample]

