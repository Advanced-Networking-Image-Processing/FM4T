import argparse, os, sys, glob
sys.path.append(os.getcwd()+"/ldm")
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import h5py
#from ldm.util import instantiate_from_config
#from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import Dataset, DataLoader
import dxchange
import skimage
import torchvision.transforms as T
transform = T.ToPILImage()
from natsort import natsorted

all_dirs = 'org_vo_losses/fact0p05/'
sub_dirs = []
for sub_fldr in os.scandir(all_dirs):
    sub_dirs.append(sub_fldr.path)

sub_dirsn = natsorted(sub_dirs)
print(sub_dirsn)
print(len(sub_dirsn))
epchs = np.transpose(np.array([[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 25, 26, 28, 29, 30, 35, 36, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 55]]))
print(epchs.shape)
SSIM_overall = np.zeros((len(sub_dirsn), 3))
PSNR_overall = np.zeros((len(sub_dirsn), 3))
jj = 0

for sub_fldri in sub_dirsn:
    print(sub_fldri)
    output_folder = sub_fldri + '/output/output_sino.tiff'
    original_folder = sub_fldri + '/original/input_sino.tiff'
    output = dxchange.read_tiff(output_folder).copy()
    original = dxchange.read_tiff(original_folder).copy()
    nums = 100
    SSIM = np.zeros(nums); PSNR = np.zeros(nums)
    
    for ii in range(nums):
        original_ii = original[ii, :, :]
        output_ii = output[ii, :, :]
        dr_org = np.max(original_ii) - np.min(original_ii)
        dr_op = np.max(output_ii) - np.min(output_ii)
        print('org and op data range', dr_org, dr_op)
        dr = max(dr_org, dr_op)

        SSIM[ii] = skimage.metrics.structural_similarity(original_ii, output_ii, data_range=dr)
        PSNR[ii] = skimage.metrics.peak_signal_noise_ratio(original_ii, output_ii, data_range=dr)
    
    print('Shapes + Real World Data')
    print('SSIM min, max, mean', np.min(SSIM), np.max(SSIM), np.mean(SSIM))
    print('PSNR min, max, mean', np.min(PSNR), np.max(PSNR), np.mean(PSNR))

    print('Shapes Data')
    SSIM_shapes = SSIM[15:65]; PSNR_shapes = PSNR[15:65]
    print('SSIM min, max, mean', np.min(SSIM_shapes), np.max(SSIM_shapes), np.mean(SSIM_shapes))
    print('PSNR min, max, mean', np.min(PSNR_shapes), np.max(PSNR_shapes), np.mean(PSNR_shapes))

    print('Real World Data')
    SSIM_RW1 = SSIM[:15]; SSIM_RW2 = SSIM[65:]; SSIM_RW = np.concatenate((SSIM_RW1, SSIM_RW2))
    PSNR_RW1 = PSNR[:15]; PSNR_RW2 = PSNR[65:]; PSNR_RW = np.concatenate((PSNR_RW1, PSNR_RW2))
    print(SSIM_RW.shape, PSNR_RW.shape)
    print('SSIM min, max, mean', np.min(SSIM_RW), np.max(SSIM_RW), np.mean(SSIM_RW))
    print('PSNR min, max, mean', np.min(PSNR_RW), np.max(PSNR_RW), np.mean(PSNR_RW))
    print('SSIM percentiles', np.percentile(SSIM_RW, [25, 50, 75]))
    print('PSNR percentiles', np.percentile(PSNR_RW, [25, 50, 75]))
    
    SSIM_overall[jj, 0] = np.min(SSIM_RW); SSIM_overall[jj, 1] = np.mean(SSIM_RW); SSIM_overall[jj, 2] = np.max(SSIM_RW)
    PSNR_overall[jj, 0] = np.min(PSNR_RW); PSNR_overall[jj, 1] = np.mean(PSNR_RW); PSNR_overall[jj, 2] = np.max(PSNR_RW)
    jj += 1

np.savetxt(all_dirs + 'SSIM_overall.txt', np.concatenate((SSIM_overall, epchs), axis=1), delimiter=',')
np.savetxt(all_dirs + 'PSNR_overall.txt', np.concatenate((PSNR_overall, epchs), axis=1), delimiter=',')

