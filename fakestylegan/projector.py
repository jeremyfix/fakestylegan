# Copyright (c) 2021, Jeremy Fix. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""
This module back projects images onto the latent space of stylegan
using a pretrained network
"""

# Standard imports
import logging

# import urllib.request
from pathlib import Path
import pickle

# External imports
import torch
import numpy as np

import matplotlib.pyplot as plt


class Projector:
    def __init__(self, encoder="e4e_ffhq_encode.pt"):
        # Take it from https://github.com/omertov/encoder4editing/blob/main/scripts/inference.py
        # It contains the decoder as well
        # network="stylegan3-r-ffhqu-256x256.pkl"):
        p = Path(decoder)
        if not p.exists():
            logging.error(
                f"I did not find the network file {network}. You may want to download it from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3/files"
            )
            logging.error(
                "For example, with : curl -LO 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl'"
            )
            raise NotImplementedError

            # TODO
            # The following never completes while the wget/curl
            # commands NVlabs provide do actually download the network
            # with urllib.request.urlopen(f'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/{network}') as response:
            #     with open(p, 'wb') as f:
            #         f.write(response.read())

        # TODO: for the following to work
        # we need stylegan3/dnnlib and stylegan3/torch_utils to be in
        # pythonpath
        with open(p, "rb") as f:
            G = pickle.load(f)["G_ema"]  # torch.nn.Module
        print(G)
        z = torch.randn([1, G.z_dim])  # latent codes
        c = None  # class labels
        img = G(z, c)  # NCHW, float32, dynamic range [-1, +1], no truncation
        img = img.squeeze().permute(1, 2, 0)
        img = img.numpy()
        img = (1.0 + img) / 2.0
        img = img.clip(0.0, 1.0)
        plt.figure()
        plt.imshow(img)
        plt.show()

    def backproject(self, img):
        if isinstance(img, str) or isinstance(img, Path):
            # Consider img is a path
            pass

        if not isinstance(img, np.array):
            pass


if __name__ == "__main__":
    p = Projector(network="stylegan3-t-ffhqu-256x256.pkl")
