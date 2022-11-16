from torch.utils.data import Dataset
import glob
import os
import pickle
import torch
from PIL import Image
import numpy as np
import csv
import torch
import random

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class dataset(Dataset):
    
    def __init__(self, split, path='./data/SRN/cars_train', picklefile='./data/cars.pickle', imgsize=128):
        self.imgsize = imgsize
        self.path = path
        super().__init__()
        self.picklefile = pickle.load(open(picklefile, 'rb'))
        
        allthevid = sorted(list(self.picklefile.keys()))
        
        random.seed(0)
        random.shuffle(allthevid)
        if split == 'train':
            self.ids = allthevid[:int(len(allthevid)*0.9)]
        else:
            self.ids = allthevid[int(len(allthevid)*0.9):]
            
                
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):
        
        item = self.ids[idx]
        
        intrinsics_filename = os.path.join(self.path, item, 'intrinsics', self.picklefile[item][0][:-4] + ".txt")
        K = np.array(open(intrinsics_filename).read().strip().split()).astype(float).reshape((3,3))
        
        indices = random.sample(self.picklefile[item], k=2)
        
        imgs = []
        poses = []
        for i in indices:
            img_filename = os.path.join(self.path, item, 'rgb', i)
            img = Image.open(img_filename)
            if self.imgsize != 128:
                img = img.resize((self.imgsize, self.imgsize))
            img = np.array(img) / 255 * 2 - 1
            
            img = img.transpose(2,0,1)[:3].astype(np.float32)
            imgs.append(img)
            
            
            pose_filename = os.path.join(self.path, item, 'pose', i[:-4]+".txt")
            pose = np.array(open(pose_filename).read().strip().split()).astype(float).reshape((4,4))
            poses.append(pose)
            
        imgs = np.stack(imgs, 0)
        poses = np.stack(poses, 0)
        R = poses[:, :3, :3]
        T = poses[:, :3, 3]
        
        return imgs, R, T, K
    
if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    
    d = dataset('train')
    dd = d[0]
    
    for ddd in dd:
        print(ddd.shape)