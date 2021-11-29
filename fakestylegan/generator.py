# Copyright (c) 2021, Jeremy Fix. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

'''
This module plays with the generator
'''

# Standard imports
import urllib.request
from pathlib import Path
import pickle
# External imports
import tqdm
import torch
from PIL import Image

import matplotlib.pyplot as plt

class Generator:

    def __init__(self, 
                 network="stylegan3-t-ffhq-1024x1024.pkl"):
                 #network="stylegan3-r-ffhqu-256x256.pkl"):
        p = Path(network) 
        if not p.exists():
            raise NotImplementedError
            #TODO
            # The following never completes while the wget/curl
            # commands NVlabs provide do actually download the network
            # with urllib.request.urlopen(f'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/{network}') as response:
            #     with open(p, 'wb') as f:
            #         f.write(response.read())

        # TODO: for the following to work
        # we need stylegan3/dnnlib and stylegan3/torch_utils to be in 
        # pythonpath
        with open(p, 'rb') as f:
                G = pickle.load(f)['G_ema']  # torch.nn.Module
        self.G = G

    @property
    def zdim(self):
        return self.G.z_dim

    def __call__(self, z=None):
        if z is None:
            z = torch.randn([1, self.G.z_dim])    # latent codes
        c = None            # class labels
        img = self.G(z, c)  # NCHW, float32, dynamic range [-1, +1], no truncation
        img = img.squeeze().permute(1, 2, 0)
        img = img.numpy()
        return (1.0 + img)/2.0, z


if __name__ == '__main__':
    torch.manual_seed(0)

    gen = Generator(network="stylegan3-r-ffhqu-256x256.pkl")

    def save_img(np_img, idx):
        pilimg = Image.fromarray( (np_img.clip(0, 1)*255).astype('uint8'))
        pilimg.save(f"img-{idx:04d}.jpg")


    idx = 0

    img, z0 = gen()
    save_img(img, idx)
    idx += 1

    # Random walk
    dz = torch.rand((gen.zdim, ))
    dz = dz / dz.norm()
    Nsteps = 10
    for i, amp in tqdm.tqdm(enumerate(torch.linspace(0, 1, Nsteps))):
        z = z0 + amp * dz
        img, _ = gen(z)
        save_img(img, idx)
        idx += 1
    

