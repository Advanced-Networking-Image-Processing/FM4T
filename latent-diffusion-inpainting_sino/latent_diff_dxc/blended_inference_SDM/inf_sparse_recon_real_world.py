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
from torchvision import models
from collections import namedtuple
import asyncio
from torch import optim
from DIB_utils import Vgg16, gram_matrix, MeanShift
import torch.nn.functional as F


# config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yaml_path = "/homes/sruban/latent-diffusion-inpainting_sino_trial/ldm/models/ldm/inpainting_big/config_Spr_Rec_rl.yaml"
model_path = '/homes/sruban/latent-diffusion-inpainting_sino_trial/logs_20k_Spr_Rec/Sept11_24/Org_phy123_w_gan/real_only_lr_1e-7/checkpoints/epoch=000153.ckpt'
bl_opt_steps = 40; bl_lr = 0.01; n_images = 100


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
    
    #print('spr_sam_jj', spr_sam_jj)
    im_size = load_image_i.shape
    #print('im_size', im_size)
    
    mask = np.ones((im_size[0], im_size[1]), dtype=np.uint8)
    sparse_ind = np.arange(0, 511, spr_sam_jj)
    #print('sparse_ind', sparse_ind)

    mask[sparse_ind, :] = 0
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
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0

    return batch


def blend_loss(model, fg_image: torch.Tensor, bg_image: torch.Tensor, curr_latent: torch.Tensor, mask: torch.Tensor, mean_shift, vgg, mse_t, preservation_ratio: float = 1e5, tv_weight: float = 1e-6, style_weight: float = 1e4):

    curr_reconstruction = model.decode_first_stage(curr_latent)
    image_comb = fg_image * mask + bg_image * (1-mask)

    target_features_style = vgg(mean_shift(image_comb))
    target_gram_style = [gram_matrix(y) for y in target_features_style]

    blend_features_style = vgg(mean_shift(curr_reconstruction))
    blend_gram_style = [gram_matrix(y) for y in blend_features_style]
    style_loss = 0

    for layer in range(len(blend_gram_style)):
        style_loss += mse_t(blend_gram_style[layer], target_gram_style[layer])

    style_loss /= len(blend_gram_style)
    style_loss *= style_weight

    tv_loss = torch.sum(torch.abs(curr_reconstruction[:, :, :, :-1] - curr_reconstruction[:, :, :, 1:])) + \
                torch.sum(torch.abs(curr_reconstruction[:, :, :-1, :] - curr_reconstruction[:, :, 1:, :]))

    tv_loss *= tv_weight

    loss = (style_loss + tv_loss + F.mse_loss(fg_image * mask, curr_reconstruction * mask)
        + F.mse_loss(bg_image * (1 - mask), curr_reconstruction * (1 - mask))
        * preservation_ratio
    )

    #for some testing purposes only
    #loss = F.mse_loss(fg_image * mask, curr_reconstruction * mask) + F.mse_loss(bg_image * (1 - mask), curr_reconstruction * (1 - mask)) * preservation_ratio

    # loss = self.lpips_model(fg_image * mask, curr_reconstruction * mask).sum() + \
    #     self.lpips_model(bg_image * (1 - mask), curr_reconstruction * (1 - mask)).sum()

    print('loss', loss)

    return loss



model,sampler=create_model(device)
spr_sam = [2, 4, 8, 16, 20]
test_dp = '/homes/sruban/latent-diffusion-inpainting_sino_trial/sino_data/512x512/Jiaze/data_test/test_100_tiff.tiff'
test_data = dxchange.read_tiff(test_dp)
print(test_data.shape)

mean_shift = MeanShift(device)
vgg = Vgg16().to(device)
mse_t = torch.nn.MSELoss()


for ii in range(len(spr_sam)):
    
    inp_data = np.zeros((n_images, 512, 512))
    mask_data = np.zeros((n_images, 512, 512))
    inpainted_data = np.zeros((n_images, 512, 512))
    pred_data = np.zeros((n_images, 512, 512))
    mask_img_data = np.zeros((n_images, 512, 512))
    blended_data = np.zeros((n_images, 512, 512))

    spr_sam_ii = spr_sam[ii]

    for jj in range(test_data.shape[0]):
        load_image_i = test_data[jj]
        image, mask = generate_sino_mask(load_image_i, spr_sam_ii)

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
        print('c shape', c.shape)

        # the mask is frst being downsampled
        cc = torch.nn.functional.interpolate(mask_tensor.to(device),size=c.shape[-2:])
        print('cc shape', cc.shape)

        # concat the masked image and downsampled mask
        c = torch.cat((c, cc), dim=1)
        print('c shape', c.shape)
        shape = (c.shape[1]-1,)+c.shape[2:]
        print('shape after c shape', shape)

        # diffusion process
        samples_ddim, _ = sampler.sample(S=50, conditioning=c, batch_size=c.shape[0], shape=shape, verbose=False)
        print('samples_ddim shape', samples_ddim.shape)

        # decode the latent vector (output)
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        print('x_samples_ddim shape', x_samples_ddim.shape)
        #print(torch.min(x_samples_ddim), torch.max(x_samples_ddim))
        
        # Cut-paste method
        inpainted = (1 - mask_tensor) * image_tensor + (mask_tensor * x_samples_ddim)

        # Actual blending process
        encoder_posterior = model.encode_first_stage(inpainted)
        initial_latent = model.get_first_stage_encoding(encoder_posterior)

        curr_latent = initial_latent.clone().detach().requires_grad_(True)
        #cr_lt = initial_latent.clone().detach()
        #curr_latent = cr_lt.requires_grad_(True)
        optimizer = optim.Adam([curr_latent], lr=bl_lr)
        #print('curr_latent.requires_grad_', curr_latent.requires_grad_)
        
        for i in tqdm(range(bl_opt_steps), desc="Reconstruction optimization"):

            optimizer.zero_grad()

            loss = blend_loss(model, fg_image=inpainted, bg_image=masked_image_tensor, curr_latent=curr_latent, mask=mask_tensor, mean_shift=mean_shift, vgg=vgg, mse_t = mse_t)

            print(f"Iteration {i}: Curr loss in {loss}")

            loss.backward()
            optimizer.step()

        bl_recons = model.decode_first_stage(curr_latent)
        recons_tensor = torch.clamp((bl_recons + 1.0) / 2.0, min=0.0, max=1.0)

        # denormalize the output
        predicted_image_clamped = torch.clamp((x_samples_ddim + 1.0)/2.0, min=0.0, max=1.0)
        image = torch.clamp((batch["image_tensor"] + 1.0) / 2.0, min=0.0, max=1.0)
        mask_tensor = torch.clamp((batch["mask_tensor"] + 1.0) / 2.0, min=0.0, max=1.0)
        mask_img_tensor = torch.clamp((batch["masked_image_tensor"] + 1.0) / 2.0, min=0.0, max=1.0)
        inpainted_tensor = torch.clamp((inpainted + 1.0) / 2.0, min=0.0, max=1.0).cpu()

        image_dxc = skimage.color.rgb2gray(np.squeeze(np.array(image.cpu())).transpose(1,2,0))
        mi_dxc = np.squeeze(np.array(mask_tensor.cpu()))
        o_dxc = skimage.color.rgb2gray(np.squeeze(np.array(inpainted_tensor)).transpose(1,2,0))
        pred_dxc = skimage.color.rgb2gray(np.squeeze(np.array(predicted_image_clamped.cpu())).transpose(1,2,0))
        mit_dxc = skimage.color.rgb2gray(np.squeeze(np.array(mask_img_tensor.cpu())).transpose(1,2,0))
        blended_dxc = skimage.color.rgb2gray(np.squeeze(np.array(recons_tensor.detach().cpu())).transpose(1,2,0))

        inp_data[jj, :, :] = image_dxc
        mask_data[jj, :, :] = mi_dxc
        inpainted_data[jj, :, :] = o_dxc
        pred_data[jj, :, :] = pred_dxc
        mask_img_data[jj, :, :] = mit_dxc
        blended_data[jj, :, :] = blended_dxc


    dxchange.write_tiff(pred_data, 'results/Jiaze_data/ldm_real/test_100/Spr_Rec/sp_' + str(spr_sam_ii) + '/pred_data', dtype='float32', overwrite=True)
    dxchange.write_tiff(inp_data, 'results/Jiaze_data/ldm_real/test_100/Spr_Rec/sp_' + str(spr_sam_ii) + '/input_data', dtype='float32', overwrite=True)
    dxchange.write_tiff(mask_data, 'results/Jiaze_data/ldm_real/test_100/Spr_Rec/sp_' + str(spr_sam_ii) + '/mask_data' , dtype='float32', overwrite=True)
    dxchange.write_tiff(inpainted_data, 'results/Jiaze_data/ldm_real/test_100/Spr_Rec/sp_' + str(spr_sam_ii) + '/inpainted_data'  , dtype='float32', overwrite=True)
    dxchange.write_tiff(mask_img_data, 'results/Jiaze_data/ldm_real/test_100/Spr_Rec/sp_' + str(spr_sam_ii) + '/masked_image_data', dtype='float32', overwrite=True)
    dxchange.write_tiff(blended_data, 'results/Jiaze_data/ldm_real/test_100/Spr_Rec/sp_' + str(spr_sam_ii) + '/blended_data', dtype='float32', overwrite=True)

