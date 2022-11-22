from xunet import XUNet

import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange
import time

from SRNdataset import dataset, MultiEpochsDataLoader
from tensorboardX import SummaryWriter
import os
import glob
from PIL import Image
import random


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str, default="./trained_model.pt")
parser.add_argument('--target',type=str, default="./data/SRN/cars_train/a4d535e1b1d3c153ff23af07d9064736")
args = parser.parse_args()


imgsize = 64

data_imgs = []
data_Rs = []
data_Ts = []

for img_filename in sorted(glob.glob(args.target + "/rgb/*.png")):
    img = Image.open(img_filename)
    img = img.resize((imgsize, imgsize))
    img = np.array(img) / 255 * 2 - 1

    img = img.transpose(2,0,1)[:3].astype(np.float32)
    data_imgs.append(img)
    
    pose_filename = os.path.join(args.target, 'pose', os.path.basename(img_filename)[:-4]+".txt")
    pose = np.array(open(pose_filename).read().strip().split()).astype(float).reshape((4,4))
    
    data_Rs.append(pose[:3, :3])
    data_Ts.append(pose[:3, 3])
    

data_K = np.array(open(os.path.join(args.target, 'intrinsics', os.path.basename(img_filename)[:-4]+".txt")).read().strip().split()).astype(float).reshape((3,3))
data_K = torch.tensor(data_K)


model = XUNet(H=imgsize, W=imgsize, ch=128)
model = torch.nn.DataParallel(model)
model.to('cuda')

ckpt = torch.load(args.model)
model.load_state_dict(ckpt['model'])


def logsnr_schedule_cosine(t, *, logsnr_min=-20., logsnr_max=20.):
    b = np.arctan(np.exp(-.5 * logsnr_max))
    a = np.arctan(np.exp(-.5 * logsnr_min)) - b
    
    return -2. * torch.log(torch.tan(a * t + b))

def xt2batch(x, logsnr, z, R, T, K):
    b = x.shape[0]
    

    return {
        'x': x.cuda(),
        'z': z.cuda(),
        'logsnr': torch.stack([logsnr_schedule_cosine(torch.zeros_like(logsnr)), logsnr], dim=1).cuda(),
        'R': R.cuda(),
        't': T.cuda(),
        'K':K[None].repeat(b,1,1).cuda(),
    }

@torch.no_grad()
def p_mean_variance(model, x, z, R, T, K, logsnr, logsnr_next, w):
    
    
    w = w[:, None, None, None]
    b = w.shape[0]
    
    c = - torch.special.expm1(logsnr - logsnr_next)
    

    squared_alpha, squared_alpha_next = logsnr.sigmoid(), logsnr_next.sigmoid()
    squared_sigma, squared_sigma_next = (-logsnr).sigmoid(), (-logsnr_next).sigmoid()
    
    alpha, sigma, alpha_next = map(lambda x: x.sqrt(), (squared_alpha, squared_sigma, squared_alpha_next))
    
    # batch = xt2batch(x, logsnr.repeat(b), z, R)
    batch = xt2batch(x, logsnr.repeat(b), z, R, T, K)
    
    
    pred_noise = model(batch, cond_mask= torch.tensor([True]*b)).detach().cpu()
    batch['x'] = torch.randn_like(x).cuda()
    pred_noise_unconditioned = model(batch, cond_mask= torch.tensor([False]*b)).detach().cpu()
    
    pred_noise_final = (1+w) * pred_noise - w * pred_noise_unconditioned
    
    z = z.detach().cpu()
    
    z_start = (z - sigma * pred_noise_final) / alpha
    z_start.clamp_(-1., 1.)
    
    model_mean = alpha_next * (z * (1 - c) / alpha + c * z_start)
    
    posterior_variance = squared_sigma_next * c
    
    return model_mean, posterior_variance

@torch.no_grad()
def p_sample(model, z, x, R, T, K, logsnr, logsnr_next, w):
    
    
    model_mean, model_variance = p_mean_variance(model, 
                                                 x=x, 
                                                 z=z, 
                                                 R=R, 
                                                 T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
    
    if logsnr_next==0:
        return model_mean
    
    return model_mean + model_variance.sqrt() * torch.randn_like(x).cpu()

@torch.no_grad()
def sample(model, record, target_R, target_T, K, w, timesteps=256):
    b = w.shape[0]
    img = torch.randn_like(torch.tensor(record[0][0]))
    
    logsnrs = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[:-1])
    logsnr_nexts = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[1:])
    
    for logsnr, logsnr_next in tqdm(zip(logsnrs, logsnr_nexts), total=len(logsnrs), desc='diffusion loop', position=1, leave=False): # [1, ..., 0] = size is 257
        condition_img, condition_R, condition_T = random.choice(record)
        condition_img = torch.tensor(condition_img)
        condition_R = torch.tensor(condition_R)
        condition_T = torch.tensor(condition_T)
        
        R = torch.stack([condition_R, target_R], 0)[None].repeat(b, 1, 1, 1)
        T = torch.stack([condition_T, target_T], 0)[None].repeat(b, 1, 1)
        condition_img = condition_img
        img = p_sample(model,
                       z=img,
                       x=condition_img, 
                       R=R,
                       T=T,
                       K=K,
                       logsnr=logsnr, logsnr_next=logsnr_next,
                       w=w)
        
    return img.cpu().numpy()


w = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
b = w.shape[0]
record = [[data_imgs[0][None].repeat(b, axis=0), 
           data_Rs[0],
           data_Ts[0]]]

os.makedirs(f'sampling/0', exist_ok=True)
Image.fromarray(((data_imgs[0].transpose(1,2,0)+1)*127.5).astype(np.uint8)).save('sampling/0/gt.png')

with torch.no_grad():
    step = 1
    for gt, R, T in tqdm(zip(data_imgs[1:], data_Rs[1:], data_Ts[1:]), total=len(data_imgs[1:]), desc='view loop', position=0):
        
        R = torch.tensor(R)
        T = torch.tensor(T)

        img = sample(model, record=record, target_R=R, target_T=T, K=data_K, w=w)
        
        record.append([img, R.cpu().numpy(), T.cpu().numpy()])
        
        
        os.makedirs(f'sampling/{step}', exist_ok=True)
        Image.fromarray(((gt.transpose(1,2,0)+1)*127.5).astype(np.uint8)).save(f'sampling/{step}/gt.png')
        for i in w:
            Image.fromarray(((img[i].transpose(1,2,0)+1)*127.5).astype(np.uint8)).save(f'sampling/{step}/{i}.png')
        
        step += 1