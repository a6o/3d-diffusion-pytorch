# Novel View Synthesis with Diffusion Models

Unofficial PyTorch Implementation of [Novel View Synthesis with Diffusion Models](3d-diffusion.github.io).

## Data Preparation:

Visit [SRN repository](https://github.com/vsitzmann/scene-representation-networks) and download `chairs_train.zip` and `cars_train.zip` and extract the downloaded files in `/data/`. Here we use 90% of the training data for training and 10% as the validation set.

We include pickle file that contains available view-png files per object. 

## Changes:

As the JAX code given by the authors are not runnable, we fixed the original code to runnable JAX code, while following the authors intend described in the paper. Please compare [Author's code](original_code.py) and our [fixed version](original_code_fixed.py) to see the changes we made.

The PyTorch implementation is in [xunet.py](xunet.py). Feel free to put up an issue if the implementation is not consistent with the original JAX code. 

## GPU requirement

We have testing 8 x RTX3090 with batch size of 128 and image size of 64 x 64. Due to the memory constraints we were not able to test the original authors' configuration of image size 128 x 128.

## TODO:
1. add trained model

## Missing:

1. EMA decay not implemented.
