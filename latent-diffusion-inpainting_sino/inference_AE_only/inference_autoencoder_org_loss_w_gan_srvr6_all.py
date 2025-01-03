import argparse, os, sys, glob
sys.path.append(os.getcwd()+"/ldm")
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import h5py
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import Dataset, DataLoader
import dxchange
import skimage
import torchvision.transforms as T
transform = T.ToPILImage()
from natsort import natsorted


def proc_data(image_ii):
    image_ii = skimage.color.gray2rgb(image_ii)
    image_ii = image_ii.astype(np.float32)/65535.0
    #print('image shape', image.shape)
    image_ii = image_ii[None].transpose(0,3,1,2)
    #print('image shape', image_ii.shape)
    image_ii = np.expand_dims(image_ii, axis=0)
    #print('image shape', image_ii.shape)

    image_ii = torch.from_numpy(image_ii)

    batch = {"image": np.squeeze(image_ii,0)}
    for k in batch:
        batch[k] = batch[k] * 2.0 - 1.0

    return batch

yaml_path = "/homes/sruban/latent-diffusion-inpainting_sino_trial/ldm/models/first_stage_models/vq-f4-noattn/config_org_loss_rl_gan.yaml"
data_path = "/homes/sruban/latent-diffusion-inpainting_sino_trial/sino_data/512x512/Jiaze/data_test/test_100.h5"
'''
yaml_path = "/homes/sruban/nicky_LDM_Torch_Radon/latent-diffusion-inpainting/ldm/models/first_stage_models/vq-f4-noattn/config_org_vo_rl_gan_fact0p005.yaml"
data_path = "/homes/sruban/latent-diffusion-inpainting_sino_trial/sino_data/512x512/Jiaze/data_test/test_124.h5"
'''
num_test_samples = 100

##load the config
config_auto = OmegaConf.load(yaml_path)

##generate the model from config
auto = instantiate_from_config(config_auto.model)
all_dirs = ckpt_path = "/homes/sruban/latent-diffusion-inpainting_sino_trial/logs_AE_trained_only/with_disc_loss/org_loss_only_lambda6_srvr/200k_data/checkpoints/"

#all_dirs = '/homes/sruban/nicky_LDM_Torch_Radon/latent-diffusion-inpainting/logs_AE_trained_org_vo_losses/rl_only/fact0p005/all_epochs/'
sub_dirs = []
for sub_fldr in os.scandir(all_dirs):
    sub_dirs.append(sub_fldr.path)

sub_dirsn = natsorted(sub_dirs)
print(sub_dirsn)
print(len(sub_dirsn))

epchs = np.array([5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 65])
#epchs = np.array([2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38])
print('epchs length', len(epchs))
#blabla
kk = 0

for sub_fldri in sub_dirsn:
    
    ckpt_path = sub_fldri
    print('ckpt_path', ckpt_path)
    ##load the state dict
    auto.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)

    image_data = h5py.File(data_path, 'r')["sino"]
    sino_op = np.zeros((num_test_samples, 512, 512))
    sino_ip = np.zeros((num_test_samples, 512, 512))

    for ii in range(num_test_samples):
        print('ii', ii)
        image_ii = image_data[ii, :, :]
        batch = proc_data(image_ii)

        ##using cpu, not gpu
        output=auto(batch['image'])
    
        ##output image
        original_image=np.array(transform(batch['image'][0]*0.5 + 0.5))
        original_image=skimage.color.rgb2gray(original_image) 
        sino_ip[ii, :, :] = original_image

        output_image = np.array(transform(output[0][0]*0.5 + 0.5))
        output_image = skimage.color.rgb2gray(output_image)
        sino_op[ii, :, :] = output_image

    dxchange.write_tiff(sino_ip, 'org_loss/w_disc/rl_only/srvr_6/epch' + str(epchs[kk]) + '/original/input_sino', dtype='float32', overwrite=True)
    dxchange.write_tiff(sino_op, 'org_loss/w_disc/rl_only/srvr_6/epch' + str(epchs[kk]) + '/output/output_sino', dtype='float32', overwrite=True)

    kk += 1



