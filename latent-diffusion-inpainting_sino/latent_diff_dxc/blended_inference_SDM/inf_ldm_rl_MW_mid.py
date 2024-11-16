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
yaml_path = "/homes/sruban/latent-diffusion-inpainting_sino_trial/ldm/models/ldm/inpainting_big/config_MW_rl.yaml"
model_path = "/homes/sruban/latent-diffusion-inpainting_sino_trial/logs_20k_MW/Sept10_24/Org_phy123_w_gan/real_only/epch134_onw/checkpoints/epoch=000138.ckpt"
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


def generate_sino_mask(load_image_i, f_i):
    
    im_size = load_image_i.shape
    blck = f_i
    si1 = random.sample(range(im_size[0] - (blck + 10)), 1)
    si = si1[0]
    mask = np.zeros((im_size[0], im_size[1]), dtype=np.uint8)
    mask[si:(si + blck), :] = 1
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
f = [10, 20, 30]
test_dp = '/homes/sruban/latent-diffusion-inpainting_sino_trial/sino_data/512x512/Jiaze/data_test/test_100_tiff.tiff'
test_data = dxchange.read_tiff(test_dp)
print(test_data.shape)

mean_shift = MeanShift(device)
vgg = Vgg16().to(device)
mse_t = torch.nn.MSELoss()


for ff in range(len(f)):
    print('ff', ff)
    f_i = f[ff]

    inp_data = np.zeros((n_images, 512, 512))
    mask_data = np.zeros((n_images, 512, 512))
    inpainted_data = np.zeros((n_images, 512, 512))
    pred_data = np.zeros((n_images, 512, 512))
    mask_img_data = np.zeros((n_images, 512, 512))
    blended_data = np.zeros((n_images, 512, 512))

    for ii in range(test_data.shape[0]):
        load_image_i = test_data[ii, :, :]
        image, mask = generate_sino_mask(load_image_i, f_i)

        # convert PIL image into input Torch Tensor
        batch=process_data(image, mask)
        image_tensor=batch["image_tensor"]
        mask_tensor=batch["mask_tensor"]
        masked_image_tensor=batch["masked_image_tensor"]

        # encode masked image and concat downsampled mask
        c = model.cond_stage_model.encode(masked_image_tensor.to(device))

        # the mask is frst being downsampled
        cc = torch.nn.functional.interpolate(mask_tensor.to(device),size=c.shape[-2:])

        # concat the masked image and downsampled mask
        c = torch.cat((c, cc), dim=1)
        shape = (c.shape[1]-1,)+c.shape[2:]

        # diffusion process
        samples_ddim, _ = sampler.sample(S=50, conditioning=c, batch_size=c.shape[0], shape=shape, verbose=False)

        # decode the latent vector (output)
        x_samples_ddim = model.decode_first_stage(samples_ddim)

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

        inp_data[ii, :, :] = image_dxc
        mask_data[ii, :, :] = mi_dxc
        inpainted_data[ii, :, :] = o_dxc
        pred_data[ii, :, :] = pred_dxc
        mask_img_data[ii, :, :] = mit_dxc
        blended_data[ii, :, :] = blended_dxc

    dxchange.write_tiff(pred_data, 'results/Jiaze_data/ldm_real/Missing_Wedge_mid/maskr' + str(f_i) + '/pred_data', dtype='float32', overwrite=True)
    dxchange.write_tiff(inp_data, 'results/Jiaze_data/ldm_real/Missing_Wedge_mid/maskr' + str(f_i) + '/input_data', dtype='float32', overwrite=True) 
    dxchange.write_tiff(mask_data, 'results/Jiaze_data/ldm_real/Missing_Wedge_mid/maskr' + str(f_i) + '/mask_data', dtype='float32', overwrite=True)
    dxchange.write_tiff(inpainted_data, 'results/Jiaze_data/ldm_real/Missing_Wedge_mid/maskr' + str(f_i) + '/inpainted_data', dtype='float32', overwrite=True)
    dxchange.write_tiff(mask_img_data, 'results/Jiaze_data/ldm_real/Missing_Wedge_mid/maskr' + str(f_i) + '/masked_image_data', dtype='float32', overwrite=True)
    dxchange.write_tiff(blended_data, 'results/Jiaze_data/ldm_real/Missing_Wedge_mid/maskr' + str(f_i) + '/blended_data', dtype='float32', overwrite=True)
