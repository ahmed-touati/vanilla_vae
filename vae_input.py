from PIL import Image
import gzip
import os
from urllib import urlretrieve
import numpy as np

# ############################# data laoding ############################


def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print('Downloading {}'.format(filename))
    urlretrieve(source + filename, filename)


def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename, source='http://yann.lecun.com/exdb/mnist/')
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
    return data / np.float32(255)


def load_mnist_dataset():
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    return X_train, X_val, X_test


def load_frey_face_images(filename):
    if not os.path.exists(filename):
        download(filename, source='http://www.cs.nyu.edu/~roweis/data/')
    import scipy.io as sio
    data = sio.loadmat(filename)['ff'].reshape(28, 20, 1, -1).transpose(3, 2, 1, 0)
    return data / np.float32(255)


def load_frey_face_dataset():
    X = load_frey_face_images('frey_rawface.mat')
    X_train, X_val = X[:-565], X[-565:]
    return X_train, X_val

# ############################## images preprocessing ##################


def get_image_array(X, index, shp=(28, 28), channels=1):
    ret = (X[index] * 255.).reshape(channels, shp[0], shp[1]).transpose(2, 1, 0).clip(0, 255).\
        astype(np.uint8)
    if channels == 1:
        ret = ret.reshape(shp[1], shp[0])
    return ret


def get_image_pair(X, Xpr, channels=1, idx=-1):
    mode = 'RGB' if channels == 3 else 'L'
    shp = X[0][0].shape
    i = np.random.randint(X.shape[0]) if idx == -1 else idx
    orig = Image.fromarray(get_image_array(X, i, shp, channels), mode=mode)
    ret = Image.new(mode, (orig.size[0], orig.size[1]*2))
    ret.paste(orig, (0, 0))
    new = Image.fromarray(get_image_array(Xpr, i, shp, channels), mode=mode)
    ret.paste(new, (0, orig.size[1]))
    return ret

# ########################## data batches streaming #####################


def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]
