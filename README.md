# Novel View Synthesis with Diffusion Models

Unofficial PyTorch Implementation of [Novel View Synthesis with Diffusion Models](https://3d-diffusion.github.io/).

## Changes:

As the JAX code given by the authors are not runnable, we fixed the original code to runnable JAX code, while following the authors intend described in the paper. Please compare [Author's code](original_code.py) and our [fixed version](original_code_fixed.py) to see the changes we made.

The PyTorch implementation is in [xunet.py](xunet.py). Feel free to put up an issue if the implementation is not consistent with the original JAX code. 

## Data Preparation:

Visit [SRN repository](https://github.com/vsitzmann/scene-representation-networks) and download `chairs_train.zip` and `cars_train.zip` and extract the downloaded files in `/data/`. Here we use 90% of the training data for training and 10% as the validation set.

We include pickle file that contains available view-png files per object. 

## Training:

```
python train.py
```

To continue training, 

```
python train.py --transfer ./results/shapenet_SRN_car/1235678
```

## Sampling:

```
python sample.py --model trained_model.pt --target ./data/SRN/cars_train/a4d535e1b1d3c153ff23af07d9064736
```

We set the diffusion steps to be 256 during sampling procedure, which takes around 10 minutes per view. 

## Pre-trained Model Weights:

[Google Drive](https://drive.google.com/file/d/1GarX4DA2FNPHeAUbzSkV1RuJC0Ci-SE5/view?usp=sharing)

We trained SRN Car dataset for 101K steps for 120 hours. We have tested using 8 x RTX3090 with batch size of 128 and image size of 64 x 64. Due to the memory constraints, we were not able to test the original authors' configuration of image size 128 x 128.


## TODO:
1. ~~Add trained model~~
2. Add evaluation code.
3. Get similar performance as reported.

## Missing:
1. EMA decay not implemented.
