# Copyright (c) 2021, Jeremy Fix. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""
This module plays with the generator
"""

# Standard imports
# import urllib.request
from pathlib import Path
import pickle
import logging

# External imports
import tqdm
import torch
from PIL import Image


class Generator:
    def __init__(
        self,
        # network="stylegan3-t-ffhq-1024x1024.pkl"):
        network="stylegan3-r-ffhqu-256x256.pkl",
    ):
        p = Path(network)
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
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.G = G
        self.G.to(self.device)

    @property
    def zdim(self):
        return self.G.z_dim

    def __call__(self, z=None):
        if z is None:
            z = torch.randn([1, self.G.z_dim]).to(self.device)  # latent codes
        c = None  # class labels
        img = self.G(z, c)  # NCHW, float32, dynamic range [-1, +1], no truncation
        img = img.detach().squeeze().permute(1, 2, 0)
        img = img.cpu().numpy()
        return (1.0 + img) / 2.0, z


if __name__ == "__main__":
    torch.manual_seed(3)

    gen = Generator()  # network="stylegan3-r-ffhqu-256x256.pkl")

    def save_img(np_img, idx):
        pilimg = Image.fromarray((np_img.clip(0, 1) * 255).astype("uint8"))
        pilimg.save(f"img-{idx:04d}.jpg")

    idx = 0

    z0 = None
    # z0 = torch.zeros((1, gen.zdim), device=gen.device)
    img, z00 = gen(z0)
    save_img(img, idx)
    idx += 1

    # Random walk
    z0 = z00
    Ntargets = 4
    for j in range(Ntargets):
        Nsteps = 50
        if j == Ntargets - 1:
            # loop back to the first image
            zf = z00
        else:
            zf = torch.randn([1, gen.zdim], device=gen.device)
        for i in tqdm.tqdm(range(1, Nsteps + 1)):
            z = z0 + i * (zf - z0) / Nsteps
            img, _ = gen(z)
            save_img(img, idx)
            idx += 1
        z0 = zf
