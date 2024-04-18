import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
transform = T.ToPILImage()
from PIL import Image, ImageDraw

import os, sys, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
import os
import random

import torch
from torch.utils.data import Dataset, Subset
import h5py
import skimage
import dxchange


class InpaintingTrain_ldm(Dataset):
    def __init__(self, size, data_root, config=None):
        self.size = size
        self.config = config or OmegaConf.create()
        self.data_root=data_root
        self.images = 20000
        #self.mask_list = []
        
        #for mask in os.listdir(data_root+"/masks"):
        #    self.mask_list.append(mask)

                    
    def __len__(self):
        return 20000
        #return len(self.mask_list)


    def __getitem__(self, i):
        
        im_size = 512

        mask = np.ones((im_size, im_size, 1), dtype=np.uint8)
        spr_fact = int(np.round(random.uniform(2, 20)))
        spr_ind = np.arange(0, im_size, spr_fact)
        mask[spr_ind, :, :] = 0

        image_address = self.data_root + '/sino/ds-simu-tr1-sino_20k_finetune.h5'
        image_al = h5py.File(image_address, 'r')
        image_all = image_al["sino"]
        image = image_all[i]
        image = skimage.color.gray2rgb(image)
        image = image.astype(np.float32)/65535.0
        image = torch.from_numpy(image)

        mask = torch.from_numpy(mask)
        masked_image = (1 - mask) * image        

        batch = {"image": np.squeeze(image,0), "mask": np.squeeze(mask,0), "masked_image": np.squeeze(masked_image,0)}
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0

        return batch
