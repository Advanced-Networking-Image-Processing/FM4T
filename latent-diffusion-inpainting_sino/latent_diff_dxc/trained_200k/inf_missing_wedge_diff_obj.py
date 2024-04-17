import argparse, os, sys, glob
sys.path.append(os.getcwd()+"/ldm")
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
import numpy as np
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import Dataset, DataLoader
import os, sys, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
import skimage
import os
import h5py
import dxchange
import torch
from torch.utils.data import Dataset, Subset
import random
import torchvision.transforms as T
transform_PIL = T.ToPILImage()
import uuid

# config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yaml_path = "../../ldm/models/ldm/inpainting_big/config.yaml"
model_path = "../../logs_20k_MW/checkpoints/epoch_580.ckpt"

## create model
def create_model(device):
    
    #load config and checkpoint
    config = OmegaConf.load(yaml_path)
    config.model['params']['ckpt_path']=model_path
    
    model = instantiate_from_config(config.model)
    sampler = DDIMSampler(model)
    model = model.to(device)

    return model,sampler


def generate_sino_mask(load_image_i, spr_sam_jj):
    
    print('spr_sam_jj', spr_sam_jj)
    im_size = load_image_i.shape
    #print('im_size', im_size)
    
    ind = int((spr_sam_jj/180)*512)
    print('ind', ind)
    mask = np.zeros((im_size[0], im_size[1]), dtype=np.uint8)
    mask[:ind, :] = 1
    mask[-ind:, :] = 1
    mask = mask.astype(np.float32)

    image = skimage.color.gray2rgb(load_image_i)
    image = image.astype(np.float32)/65535.0
    
    return image, mask


def process_data(image,mask):

    # creating a 3 dimensional mask
    mask = np.array(mask)
    mask = np.expand_dims(mask, axis=2)

    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = mask[None].transpose(0,3,1,2)
    mask = torch.from_numpy(mask)
    masked_image = (1 - mask) * image

    batch = {"image_tensor": image, "mask_tensor": mask, "masked_image_tensor": masked_image}
    for k in batch:
        batch[k] = batch[k] * 2.0 - 1.0

    return batch

model,sampler=create_model(device)
#print('model', model)
#print('sampler', sampler)

fr_smb = [16, 4, 0]
smb = ['small', 'medium', 'big']
spr_sam = [10, 20, 30]
data_pat = "../../sino_data/512x512/data/sino/diff_size_obj/ds-simu-test_sino_20_"

for ii in range(len(fr_smb)):
    data_path = data_pat + smb[ii] + '.h5'
    print('data_path', data_path)
    load_image = h5py.File(data_path, 'r')["sino"]
    load_image_i = load_image[fr_smb[ii], :, :]
    print('load_image_i', load_image_i.shape)

    for jj in range(len(spr_sam)):
        image, mask = generate_sino_mask(load_image_i, spr_sam[jj])

        ##Inference

        # convert PIL image into input Torch Tensor
        batch=process_data(image, mask)
        image_tensor=batch["image_tensor"]
        mask_tensor=batch["mask_tensor"]
        masked_image_tensor=batch["masked_image_tensor"]

        #print('image_tensor shape', image_tensor.shape)
        #print('mask_tensor shape', mask_tensor.shape)
        #print('masked_image_tensor shape', masked_image_tensor.shape)

        # encode masked image and concat downsampled mask
        c = model.cond_stage_model.encode(masked_image_tensor.to(device))
        #print('c shape', c.shape)

        # the mask is frst being downsampled
        cc = torch.nn.functional.interpolate(mask_tensor.to(device),size=c.shape[-2:])
        #print('cc shape', cc.shape)

        # concat the masked image and downsampled mask
        c = torch.cat((c, cc), dim=1)
        #print('c shape', c.shape)
        shape = (c.shape[1]-1,)+c.shape[2:]
        #print('shape', shape)

        # diffusion process
        samples_ddim, _ = sampler.sample(S=50, conditioning=c, batch_size=c.shape[0], shape=shape, verbose=False)

        # decode the latent vector (output)
        x_samples_ddim = model.decode_first_stage(samples_ddim)

        # denormalize the output
        predicted_image_clamped = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

        output_PIL=transform_PIL(predicted_image_clamped[0])

        image_dxc = skimage.color.rgb2gray(np.squeeze(np.array(image_tensor)).transpose(1,2,0))
        mi_dxc = skimage.color.rgb2gray(np.squeeze(np.array(masked_image_tensor)).transpose(1,2,0))
        o_dxc = skimage.color.rgb2gray(np.squeeze(np.array(output_PIL)))

        #print('dxc shapes', image_dxc.shape, mi_dxc.shape, o_dxc.shape)

        dxchange.write_tiff(image_dxc, 'fine_tuned_20k/Missing_Wedge/diff_size_obj/' + smb[ii] + '_obj/' + 'mask_' + str(spr_sam[jj]) + '/org_image' + str(fr_smb[ii]), dtype='float32', overwrite=True)
        dxchange.write_tiff(mi_dxc, 'fine_tuned_20k/Missing_Wedge/diff_size_obj/' + smb[ii] + '_obj/' + 'mask_' + str(spr_sam[jj]) + '/masked_org_image' + str(fr_smb[ii]), dtype='float32', overwrite=True)
        dxchange.write_tiff(o_dxc, 'fine_tuned_20k/Missing_Wedge/diff_size_obj/' + smb[ii] + '_obj/' + 'mask_' + str(spr_sam[jj]) + '/output_image' + str(fr_smb[ii]), dtype='float32', overwrite=True)
