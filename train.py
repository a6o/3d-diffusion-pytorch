from xunet import XUNet

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange
import time

from SRNdataset import dataset, MultiEpochsDataLoader
from tensorboardX import SummaryWriter
import os

image_size = 64
batch_size = 128

d = dataset('train', path='./data/SRN/cars_train', imgsize=image_size)
d_val = dataset('val', path='./data/SRN/cars_train', imgsize=image_size)

loader = MultiEpochsDataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=40)
loader_val = DataLoader(d_val, batch_size=128, shuffle=True, drop_last=True, num_workers=16)

device = "cuda:0"

model = XUNet(H=image_size, W=image_size, ch=128)
model = torch.nn.DataParallel(model)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))




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
        'K':K.cuda(),
    }

def q_sample(z, logsnr, noise):
    
    # lambdas = logsnr_schedule_cosine(t)
    
    alpha = logsnr.sigmoid().sqrt()
    sigma = (-logsnr).sigmoid().sqrt()
    
    alpha = alpha[:,None, None, None]
    sigma = sigma[:,None, None, None]

    return alpha * z + sigma * noise

def p_losses(denoise_model, img, R, T, K, logsnr, noise=None, loss_type="l2", cond_prob=0.1):
        
    B = img.shape[0]
    x = img[:, 0]
    z = img[:, 1]
    if noise is None:
        noise = torch.randn_like(x)

    z_noisy = q_sample(z=z, logsnr=logsnr, noise=noise)
    
    
    cond_mask = (torch.rand((B,)) > cond_prob).cuda()
    
    x_condition = torch.where(cond_mask[:, None, None, None], x, torch.randn_like(x))
    
    batch = xt2batch(x=x_condition, logsnr=logsnr, z=z_noisy, R=R, T=T, K=K)
    
    predicted_noise = denoise_model(batch, cond_mask=cond_mask.cuda())

    if loss_type == 'l1':
        loss = F.l1_loss(noise.to(device), predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise.to(device), predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise.to(device), predicted_noise)
    else:
        raise NotImplementedError()

    return loss


@torch.no_grad()
def sample(model, img, R, T, K, w, timesteps=256):
    x = img[:, 0]
    img = torch.randn_like(x)
    imgs = []
    
    logsnrs = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[:-1])
    logsnr_nexts = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[1:])
    
    for logsnr, logsnr_next in tqdm(zip(logsnrs, logsnr_nexts)): # [1, ..., 0] = size is 257
        img = p_sample(model, x=x, z=img, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def p_sample(model, x, z, R, T, K, logsnr, logsnr_next, w):
    
    
    model_mean, model_variance = p_mean_variance(model, x=x, z=z, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
    
    if logsnr_next==0:
        return model_mean
    
    return model_mean + model_variance.sqrt() * torch.randn_like(x).cpu()


@torch.no_grad()
def p_mean_variance(model, x, z, R, T, K, logsnr, logsnr_next, w=2.0):
    
    
    strt = time.time()
    b = x.shape[0]
    w = w[:, None, None, None]
    
    c = - torch.special.expm1(logsnr - logsnr_next)
    

    squared_alpha, squared_alpha_next = logsnr.sigmoid(), logsnr_next.sigmoid()
    squared_sigma, squared_sigma_next = (-logsnr).sigmoid(), (-logsnr_next).sigmoid()
    
    alpha, sigma, alpha_next = map(lambda x: x.sqrt(), (squared_alpha, squared_sigma, squared_alpha_next))
    
    # batch = xt2batch(x, logsnr.repeat(b), z, R)
    batch = xt2batch(x, logsnr.repeat(b), z, R, T, K)
    
    strt = time.time()
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


def warmup(optimizer, step, last_step, last_lr):
    
    if step < last_step:
        optimizer.param_groups[0]['lr'] = step / last_step * last_lr
        
    else:
        optimizer.param_groups[0]['lr'] = last_lr
        

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('transfer',type=str, default="")
args = parser.parse_args()


if args.transfer == "":
    now = './results/shapenet_SRN_car/'+"original_"+str(int(time.time()))
    writer = SummaryWriter(now)
    step = 0
else:
    print('transfering from: ', args.transfer)
    
    ckpt = torch.load(os.path.join(args.transfer, 'latest.pt'))
    
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optim'])
    
    now = args.transfer
    writer = SummaryWriter(now)
    step = ckpt['step']
    
    
for e in range(100000):
    print(f'starting epoch {e}')
    
    lt = time.time()
    for img, R, T, K in tqdm(loader):
        
        warmup(optimizer, step, 10000000/batch_size, 0.0001)
        
        B = img.shape[0]
        
        
        optimizer.zero_grad()

        logsnr = logsnr_schedule_cosine(torch.rand((B,)))
        
        loss = p_losses(model, img=img.cuda(), R=R.cuda(), T=T.cuda(), K=K.cuda(), logsnr=logsnr.cuda(), loss_type="l2", cond_prob=0.1)
        loss.backward()
        optimizer.step()
        
        writer.add_scalar("train/loss", loss.item(), global_step=step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step=step)
        
        if step % 500 == 0:
            print("Loss:", loss.item())
                
            
        if step % 1000 == 900: 
            model.eval()
            with torch.no_grad():
                for oriimg, R, T, K in loader_val:
                    
                    w = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(16)
                    img = sample(model, img=oriimg, R=R, T=T, K=K, w=w)

                    img = rearrange(((img[-1].clip(-1,1) + 1) * 127.5).astype(np.uint8), "(b a) c h w -> a c h (b w)", a=8, b=16)

                    gt = rearrange(((oriimg[:,1] + 1) * 127.5).detach().cpu().numpy().astype(np.uint8), "(b a) c h w -> a c h (b w)", a=8, b=16)
                    cd = rearrange(((oriimg[:,0] + 1) * 127.5).detach().cpu().numpy().astype(np.uint8), "(b a) c h w -> a c h (b w)", a=8, b=16)

                    fi = np.concatenate([cd, gt, img], axis=2)
                    for i, ww in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
                        writer.add_image(f"train/{ww}", fi[i], step)
                    break
            print('image sampled!')
            writer.flush()
            model.train()
            
        
        if step == int(10000000/batch_size):
            torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step}, now+f"/after_warmup.pt")
        
        step += 1
        starttime = time.time()
        
    
    if e%20 == 0:
        torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step, 'epoch':e}, now+f"/latest.pt")