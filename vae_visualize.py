from vae import VAE
import theano
from vae_input import get_image_array
from PIL import Image
import numpy as np
from scipy.stats import norm
import json
import sys


def main(L=2, z_dim=2, n_hid=500, binary=True, model_dir='face_model', width=20, height=28,
         channels=1, img_file='2D_faces_manifold.png', **kwargs):
    print('building model ..')
    model = VAE(None,
                L=L,
                binary=binary,
                imgshape=(width, height),
                channels=channels,
                z_dim=z_dim,
                n_hid=n_hid)
    model.load_params(dir=model_dir, epoch=None)
    generate_funct = model.generate()

    im = Image.new('L', (width*19, height*19))
    for (x, y), _ in np.ndenumerate(np.zeros((19, 19))):
        z = np.asarray([norm.ppf(0.05*(x+1)), norm.ppf(0.05*(y+1))], dtype=theano.config.floatX)
        generated_sample = generate_funct(z).reshape(-1, 1, width, height)
        im.paste(Image.fromarray(get_image_array(generated_sample, 0, shp=(width, height))), (width*x, height*y))
    im.save(img_file)


if __name__ == '__main__':
    config_dict = json.load(open('config.json'))

    if sys.argv[1] == 'mnist':
        config_dict['mnist'].update({"binary": True})
        main(**config_dict['mnist'])
    if sys.argv[1] == 'faces':
        config_dict['faces'].update({"binary": False})
        main(**config_dict['faces'])

