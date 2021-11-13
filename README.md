# Fakestylegan

Code to experiment with stylegan.

Expected target: 

- [ ] stylegan running as a server on a GPU node, 
- [x] waiting for input images and aligning it
- [ ] backproejcting onto the latent space
- [ ] playing around with the latent direction to generate new images

## Installation

    python3 -m pip install git+https://github.com/jeremyfix/fakestylegan

## Testing

### Alignment

For testing the alignment code, which is the one used by [NVlabs](https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py) for realigning their input face image, you can :

    python3 -m fakestylegan.align mysourceimage.jpg


