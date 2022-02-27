# Fakestylegan

Code to experiment with stylegan.

Expected target: 

- [ ] stylegan running as a server on a GPU node, 
- [x] waiting for input images and aligning it
- [ ] backproejcting onto the latent space
- [ ] playing around with the latent direction to generate new images

## Installation

    python3 -m pip install git+https://github.com/jeremyfix/fakestylegan

You also need to have the `dnnlib` and `torch_utils` directories from the original [stylegan3](https://github.com/NVlabs/stylegan3) repository.

Then, if you are using a GPU, you need to set the `CUDA_HOME` variable appropriately :  

```
export CUDA_HOME=/usr/local/cuda-11
```

and then have fun without our scripts, for example 

    python3 -m fakestylegan.generator

## Testing

### Alignment

For testing the alignment code, which is the one used by [NVlabs](https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py) for realigning their input face image, you can :

    python3 -m fakestylegan.align mysourceimage.jpg

### Interpolation in the latent space

The video [faces.avi](./examples/faces.avi) has been generated with `python3 -m fakestylegan.generator` and goes around in the
latent space to illustrate its topology and the incredible power of stylegan to generate faces.

[![See stylegan3 in action](https://img.youtube.com/vi/xNXCXO3LpEI/hqdefault.jpg)](https://youtu.be/xNXCXO3LpEI)

The interpolation scripts takes 30 seconds on a Geforce 3090, generating 200 images.
