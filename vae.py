import numpy as np
import cPickle
import os

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne


# ################# custom layer ###########


class GaussianSampleLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape = (self.input_shapes[0][0] or inputs[0].shape[0],
                 self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(0.5 * logsigma) * self.rng.normal(shape)


# ##################### build model #######################

class VAE(object):
    def __init__(self, input_var, L=2, binary=True, imgshape=(28, 28), channels=1, z_dim=2, n_hid=500):
        self.input_var = input_var
        self.L = L
        self.binary = binary
        self.x_dim = imgshape[0] * imgshape[1] * channels
        self.z_dim = z_dim
        self.n_hid = n_hid
        self.l_input = lasagne.layers.InputLayer(shape=(None, channels, imgshape[0], imgshape[1]),
                                                 input_var=self.input_var, name='input')
        self.l_enc_hid = lasagne.layers.DenseLayer(incoming=self.l_input, num_units=self.n_hid,
                                                   nonlinearity=lasagne.nonlinearities.tanh if binary else T.nnet.softplus,
                                                   name='enc_hid')
        self.l_enc_mu = lasagne.layers.DenseLayer(incoming=self.l_enc_hid, num_units=self.z_dim,
                                                  nonlinearity=None, name='enc_mu')
        self.l_enc_logsigma = lasagne.layers.DenseLayer(incoming=self.l_enc_hid, num_units=self.z_dim,
                                                        nonlinearity=None, name='enc_logsigma')

        self.l_dec_mu_list = []
        self.l_dec_logsigma_list = []
        self.l_output_list = []
        self.l_output = None

        # tie the weights of all L versions so they are the "same" layer
        W_dec_hid = None
        b_dec_hid = None
        W_dec_mu = None
        b_dec_mu = None
        W_dec_logsigma = None
        b_dec_logsigma = None

        for i in xrange(self.L):
            l_Z = GaussianSampleLayer(self.l_enc_mu, self.l_enc_logsigma, name='Z')
            l_dec_hid = lasagne.layers.DenseLayer(incoming=l_Z, num_units=self.n_hid,
                                                  nonlinearity=lasagne.nonlinearities.tanh if binary else T.nnet.softplus,
                                                  W=lasagne.init.GlorotUniform() if W_dec_hid is None else W_dec_hid,
                                                  b=lasagne.init.Constant(0.) if b_dec_hid is None else b_dec_hid,
                                                  name='dec_hid')
            if self.binary:
                l_output = lasagne.layers.DenseLayer(incoming=l_dec_hid, num_units=self.x_dim,
                                                     nonlinearity=lasagne.nonlinearities.sigmoid,
                                                     W=lasagne.init.GlorotUniform() if W_dec_mu is None else W_dec_mu,
                                                     b=lasagne.init.Constant(0.0) if b_dec_mu is None else b_dec_mu,
                                                     name='dec_output')
                self.l_output_list.append(l_output)
                if W_dec_hid is None:
                    W_dec_hid = l_dec_hid.W
                    b_dec_hid = l_dec_hid.b
                    W_dec_mu = l_output.W
                    b_dec_mu = l_output.b
            else:
                l_dec_mu = lasagne.layers.DenseLayer(incoming=l_dec_hid, num_units=self.x_dim,
                                                     nonlinearity=lasagne.nonlinearities.sigmoid,
                                                     W=lasagne.init.GlorotUniform() if W_dec_mu is None else W_dec_mu,
                                                     b=lasagne.init.Constant(0.0) if b_dec_mu is None else b_dec_mu,
                                                     name='dec_mu')
                # relu_shift is for numerical stability
                relu_shift = 10
                l_dec_logsigma = lasagne.layers.DenseLayer(l_dec_hid, num_units=self.x_dim,
                                                           W=lasagne.init.GlorotUniform() if W_dec_logsigma is None else W_dec_logsigma,
                                                           b=lasagne.init.Constant(0) if b_dec_logsigma is None else b_dec_logsigma,
                                                           nonlinearity=lambda a: T.nnet.relu(a + relu_shift) - relu_shift,
                                                           name='dec_logsigma')
                l_output = GaussianSampleLayer(l_dec_mu, l_dec_logsigma, name='dec_output')
                self.l_dec_mu_list.append(l_dec_mu)
                self.l_dec_logsigma_list.append(l_dec_logsigma)
                self.l_output_list.append(l_output)
                if W_dec_hid is None:
                    W_dec_hid = l_dec_hid.W
                    b_dec_hid = l_dec_hid.b
                    W_dec_mu = l_dec_mu.W
                    b_dec_mu = l_dec_mu.b
                    W_dec_logsigma = l_dec_logsigma.W
                    b_dec_logsigma = l_dec_logsigma.b
        self.l_output = lasagne.layers.ElemwiseSumLayer(incomings=self.l_output_list, coeffs=1.0/self.L, name='output')

    def compute_loss(self, deterministic):
        layer_outputs = lasagne.layers.get_output([self.l_enc_mu, self.l_enc_logsigma] + self.l_dec_mu_list +
                                                  self.l_dec_logsigma_list + self.l_output_list,
                                                  deterministic=deterministic)
        # unpacl layer_outputs
        enc_mu = layer_outputs[0]
        enc_logsigma = layer_outputs[1]
        dec_mu_list = [] if self.binary else layer_outputs[2:2+self.L]
        dec_logsigma_list = [] if self.binary else layer_outputs[2+self.L: 2 + 2*self.L]
        output_list = layer_outputs[2:] if self.binary else layer_outputs[2 + 2*self.L]

        KLD_divergence = - 0.5 * T.sum(1 + enc_logsigma - enc_mu**2 - T.exp(enc_logsigma), axis=1)
        if self.binary:
            reconstruction_error = - (1./self.L) * sum(T.sum(T.nnet.binary_crossentropy(out, self.input_var.flatten(2)),
                                                       axis=1) for out in output_list)
        else:
            reconstruction_error = (1./self.L) * sum(T.sum(- 0.5 * np.log(2 * np.pi) - 0.5 * dec_logsigma
                                                           - 0.5 * (self.input_var.flatten(2) - dec_mu) ** 2
                                                           / T.exp(dec_logsigma), axis=1)
                                                     for dec_mu, dec_logsigma in zip(dec_mu_list, dec_logsigma_list))

        loss = - T.mean(reconstruction_error - KLD_divergence)
        return loss

    def save_params(self, dir='model', epoch=None):
        params = lasagne.layers.get_all_param_values(self.l_output)
        filename = 'params.pkl' if epoch is None else 'params_{}.pkl'.format(epoch)
        filename = os.path.join(dir, filename)
        with open(filename, 'wb') as f:
            cPickle.dump(params, f)

    def load_params(self, dir='model', epoch=None):
        filename = 'params.pkl' if epoch is None else 'params_{}.pkl'.format(epoch)
        filename = os.path.join(dir, filename)
        with open(filename, 'rb') as f:
            params = cPickle.load(f)
        lasagne.layers.set_all_param_values(self.l_output, params)

    def generate(self):
        z_var = T.vector('z_var')
        if self.binary:
            generated_sample = lasagne.layers.get_output(self.l_output, {self.l_enc_mu: z_var},
                                                         deterministic=True)
        else:
            generated_sample = lasagne.layers.get_output(self.l_output_list[0],
                                                         {self.l_enc_mu: z_var}, deterministic=True)
        return theano.function(
            inputs=[z_var],
            outputs=generated_sample
        )


