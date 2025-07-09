from operator import index
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# import torchio as tio
import warnings
from glob import glob
import albumentations as A
import random

warnings.filterwarnings("ignore")

class VISADataset(Dataset):
    """ABIDE dataset."""

    def __init__(self, mode, object_class, rootdir='/home/farzadbz/VisA/split_csv', transform=None,  normal=True, image_size=256,  center_size=224, augment=False, center_crop=False, anomaly_class='all'):
        """
        Args:
            mode: 'train','val','test'
            root_dir (string): Directory with all the volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
            df_root_path (string): dataframe directory containing csv files
        """
        self.mode = mode
        self.center_size = center_size
        if mode == 'train' and normal==False:
            raise Exception('training data should be normal')
        object_cls_dict = {
        "candle":0,
        "cashew":1,
        "fryum":2,
        "macaroni2":3,
        "pcb2":4,
        "pcb4":5,
        "capsules":6,
        "chewinggum":7,
        "macaroni1":8,
        "pcb1":9,
        "pcb3":10,
        "pipe_fryum":11
        }
        self.normal = normal
        self.transform = transform
        self.object_class = object_class
        self.augment = augment
        self.center_crop = center_crop
        self.image_size = image_size

        if self.object_class == 'all':
            if mode=='train':
                df = pd.read_csv(os.path.join('.', 'splits', 'visa-split.csv')).query(f'split=="train"').reset_index(drop=True)
            else:
                if normal:
                    df = pd.read_csv(os.path.join('.', 'splits', 'visa-split.csv')).query(f'split=="test"').query('label=="normal"').reset_index(drop=True)
                else:
                    df = pd.read_csv(os.path.join('.', 'splits', 'visa-split.csv')).query(f'split=="test"').reset_index(drop=True)
        else:
            if mode=='train':
                df = pd.read_csv(os.path.join('.', 'splits', 'visa-split.csv')).query(f'split=="train"').query(f'object=="{self.object_class}"').reset_index(drop=True)
            else:
                if normal:
                    df = pd.read_csv(os.path.join('.', 'splits', 'visa-split.csv')).query(f'split=="test"').query('label=="normal"').query(f'object=="{self.object_class}"').reset_index(drop=True)
                else:
                    df = pd.read_csv(os.path.join('.', 'splits', 'visa-split.csv')).query(f'split=="test"').query(f'object=="{self.object_class}"').reset_index(drop=True) 
        self.images = []
        self.segs = []
        self.object_classes = []
        for i,row in df.iterrows():
            data_path = os.path.join(rootdir, row['image'])
            img = np.array(Image.open(data_path).convert('RGB').resize((self.image_size, self.image_size))).astype(np.float32)
            self.images.append(img)
            # object_class = data_path.split('/')[9]
            
            self.object_classes.append(object_cls_dict[row['object']])
            if row['label']!='normal':
                seg_path = os.path.join(rootdir, row['mask'])
                    
                seg = (np.array(Image.open(seg_path).convert('L').resize((self.image_size, self.image_size)))>0).astype(np.uint8)
                self.segs.append(seg)
            else:
                self.segs.append(np.zeros((image_size,image_size)).astype(np.int32))

        if self.augment:
           self.aug = A.Compose([A.Affine (rotate=3, p=0.3),
                A.Affine (translate_px=int(self.image_size/8 - self.center_size/8), p=0.8),
                A.RandomBrightnessContrast(
                brightness_limit=0.05,
                contrast_limit=0.05,
                p=0.3),
                A.CenterCrop(p=1, height=self.center_size, width=self.center_size)])
        else:
            self.aug = A.CenterCrop(p=1, height=self.center_size, width=self.center_size)


    def transform_volume(self, x):
        x = torch.from_numpy(x.transpose((-1, 0 , 1)))
        return x


    def __len__(self):
        return len(self.images)
        

    def __getitem__(self, index):
        img = self.images[index].astype(np.uint8)
        seg = self.segs[index].astype(np.int32)
        if self.center_crop:
            augmented = self.aug(image=img, mask=seg)
            img = augmented['image']
            seg = augmented['mask']
            
        if random.random()< 0.5 and self.mode=='train':
            img = np.rot90(img, 2).copy()
            seg = np.rot90(seg, 2).copy()
            
        img = img.astype(np.float32) / 255.0
        y = self.object_classes[index]
        if self.transform:
            img = self.transform(img)
        else:
            img = self.transform_volume(img)
            img = (img-0.5)/0.5
        return img, seg, int(y)