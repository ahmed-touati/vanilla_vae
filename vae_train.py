import theano
from theano import tensor as T
import lasagne
from vae import VAE
from vae_input import load_mnist_dataset, iterate_minibatches, load_frey_face_dataset

from datetime import datetime
import numpy as np
import json
import sys

# from optparse import OptionParser


def main(L=2, z_dim=2, n_hid=500, binary=True,
         n_epochs=40, batch_size=100, valid_freq=1000,
         checkpoint_freq=5000, model_dir='face_model', **kwargs):
    print('loading data ..')
    if binary:
        X_train, X_val, _ = load_mnist_dataset()
    else:
        X_train, X_val = load_frey_face_dataset()

    channels, height, width = X_train.shape[1], X_train.shape[2], X_train.shape[3]
    input_var = T.tensor4('input')

    print('building vae model')
    model = VAE(input_var,
                L=L,
                binary=binary,
                imgshape=(width, height),
                channels=channels,
                z_dim=z_dim,
                n_hid=n_hid)
    loss = model.compute_loss(deterministic=False)
    test_loss = model.compute_loss(deterministic=True)
    params = lasagne.layers.get_all_params(model.l_output, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=1e-4)
    train_model = theano.function(
        inputs=[input_var],
        outputs=loss,
        updates=updates
    )
    valid_model = theano.function(
        inputs=[input_var],
        outputs=test_loss,
    )

    print('start training ..')
    n_batches = X_train.shape[0] / batch_size
    for epoch in range(1, n_epochs + 1):
        for (idx, batch) in enumerate(iterate_minibatches(X_train, batch_size, shuffle=True)):
            iter = (epoch - 1) * n_batches + idx
            train_loss = train_model(batch)
            if iter % 10 == 0:
                print('%s, epoch %i, minibatch %i / %i, train likelihood %f' % (datetime.now(),
                                                                                epoch,
                                                                                idx + 1,
                                                                                n_batches,
                                                                                train_loss))
            if (iter + 1) % valid_freq == 0:
                # compute likelihood on validation set
                print 'computing validation likelihood..'
                valid_loss = [valid_model(valid_batch)
                                         for valid_batch in iterate_minibatches(X_val, batch_size, shuffle=False)]
                mean_valid_loss = np.mean(valid_loss)
                print('%s, epoch %i, minibatch %i / %i, validation likelihood %f' % (datetime.now(),
                                                                                     epoch,
                                                                                     idx + 1,
                                                                                     n_batches,
                                                                                     mean_valid_loss)
                      )

            if (iter + 1) % checkpoint_freq == 0:
                print('saving model ..')
                model.save_params(dir=model_dir, epoch=epoch)
    print('saving final model ..')
    model.save_params(dir=model_dir, epoch=None)


if __name__ == '__main__':

    # parser = OptionParser()
    # parser.add_option('--L', default=2)
    # parser.add_option('--z_dim', default=2)
    # parser.add_option('--n_hid', default=200)
    # parser.add_option('--binary', default=False)
    # parser.add_option('--n_epochs', default=300)
    # parser.add_option('--batch_size', default=100)
    # parser.add_option('--valid_freq', default=1000)
    # parser.add_option('--checkpoint_freq', default=5000)
    # parser.add_option('--model_dir', default='face_model')
    # (options, _) = parser.parse_args()
    # main(** vars(options))

    config_dict = json.load(open('config.json'))

    if sys.argv[1] == 'mnist':
        config_dict['mnist'].update({"binary": True})
        main(**config_dict['mnist'])
    if sys.argv[1] == 'faces':
        config_dict['faces'].update({"binary": False})
        main(**config_dict['faces'])



