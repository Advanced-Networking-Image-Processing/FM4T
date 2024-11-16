import argparse, os, sys, glob
sys.path.append(os.getcwd()+"/ldm")
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import h5py
from natsort import natsorted
#from ldm.util import instantiate_from_config
#from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import Dataset, DataLoader
import dxchange
import skimage
import torchvision.transforms as T
transform = T.ToPILImage()

directory = 'results/Jiaze_data/w_blending/ldm_real_only/random_masking/2D_sinograms'
subfolders1 = [f.path for f in os.scandir(directory) if f.is_dir()]
print(subfolders1)
subfolders= natsorted(subfolders1)[:-1]
print(subfolders)
print(len(subfolders))

SSIM_all_rw = np.zeros((len(subfolders), 3)); PSNR_all_rw = np.zeros((len(subfolders), 3))
SSIM_pr_all_rw = np.zeros((len(subfolders), 3)); PSNR_pr_all_rw = np.zeros((len(subfolders), 3))
SSIM_bl_all_rw = np.zeros((len(subfolders), 3)); PSNR_bl_all_rw = np.zeros((len(subfolders), 3))

SSIM_all_rw_sh = np.zeros((len(subfolders), 3)); PSNR_all_rw_sh = np.zeros((len(subfolders), 3))
SSIM_pr_all_rw_sh = np.zeros((len(subfolders), 3)); PSNR_pr_all_rw_sh = np.zeros((len(subfolders), 3))
SSIM_bl_all_rw_sh = np.zeros((len(subfolders), 3)); PSNR_bl_all_rw_sh = np.zeros((len(subfolders), 3))

SSIM_all_sh = np.zeros((len(subfolders), 3)); PSNR_all_sh = np.zeros((len(subfolders), 3))
SSIM_pr_all_sh = np.zeros((len(subfolders), 3)); PSNR_pr_all_sh = np.zeros((len(subfolders), 3))
SSIM_bl_all_sh = np.zeros((len(subfolders), 3)); PSNR_bl_all_sh = np.zeros((len(subfolders), 3))

ratio = np.transpose(np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]))

for ii in range(len(subfolders)):
    print('ii', ii)
    print('subfolders[ii]', subfolders[ii])
    subfolder_ii = subfolders[ii]
    
    output_folder = subfolder_ii + '/inpainted_data.tiff'
    original_folder = subfolder_ii +  '/input_data.tiff'
    mask_folder = subfolder_ii + '/mask_data.tiff'
    blended_folder = subfolder_ii + '/blended_data.tiff'
    pred_folder = subfolder_ii + '/pred_data.tiff'

    output = dxchange.read_tiff(output_folder).copy()
    original = dxchange.read_tiff(original_folder).copy()
    mask = dxchange.read_tiff(mask_folder).copy()
    blended = dxchange.read_tiff(blended_folder).copy()
    pred = dxchange.read_tiff(pred_folder).copy()
    
    print('output min and max', np.min(output), np.max(output))
    print('original min and max', np.min(original), np.max(original))
    print('mask min and max', np.min(mask), np.max(mask))
    print('blended min and max', np.min(blended), np.max(blended))
    print('pred min and max', np.min(pred), np.max(pred))

    nums = 100
    SSIM_ii = np.zeros(nums); PSNR_ii = np.zeros(nums)
    SSIM_bl_ii = np.zeros(nums); PSNR_bl_ii = np.zeros(nums)
    SSIM_pr_ii = np.zeros(nums); PSNR_pr_ii= np.zeros(nums)

    for iii in range(nums):

        print('iii', iii)
        original_iii = original[iii, :, :]
        output_iii = output[iii, :, :]
        blended_iii = blended[iii, :, :]
        pred_iii = pred[iii, :, :]

        dr_org = np.max(original_iii) - np.min(original_iii)
        dr_op = np.max(output_iii) - np.min(output_iii)
        dr_bl = np.max(blended_iii) - np.min(blended_iii)
        dr_pr = np.max(pred_iii) - np.min(pred_iii)

        print('org and op data range', dr_org, dr_op)
        dr = max(dr_org, dr_op)
        SSIM_ii[iii] = skimage.metrics.structural_similarity(original_iii, output_iii, data_range=dr)
        PSNR_ii[iii] = skimage.metrics.peak_signal_noise_ratio(original_iii, output_iii, data_range=dr)
        print('SSIM, PSNR', SSIM_ii[iii], PSNR_ii[iii])
        
        print('org and bl data range', dr_bl, dr_org)
        dr1 = max(dr_bl, dr_org)
        SSIM_bl_ii[iii] = skimage.metrics.structural_similarity(original_iii, blended_iii, data_range=dr1)
        PSNR_bl_ii[iii] = skimage.metrics.peak_signal_noise_ratio(original_iii, blended_iii, data_range=dr1)
        
        print('org and pr data range', dr_pr, dr_org)
        dr2 = max(dr_pr, dr_org)
        SSIM_pr_ii[iii] = skimage.metrics.structural_similarity(original_iii, pred_iii, data_range=dr2)
        PSNR_pr_ii[iii] = skimage.metrics.peak_signal_noise_ratio(original_iii, pred_iii, data_range=dr2)

    SSIM_sh = SSIM_ii[15:65]; PSNR_sh = PSNR_ii[15:65]
    SSIM_bl_sh = SSIM_bl_ii[15:65]; PSNR_bl_sh = PSNR_bl_ii[15:65]
    SSIM_pr_sh = SSIM_pr_ii[15:65]; PSNR_pr_sh = PSNR_pr_ii[15:65]

    SSIM_rw = np.concatenate((SSIM_ii[:15], SSIM_ii[65:])); PSNR_rw = np.concatenate((PSNR_ii[:15], PSNR_ii[65:]))
    SSIM_bl_rw = np.concatenate((SSIM_bl_ii[:15], SSIM_bl_ii[65:])); PSNR_bl_rw = np.concatenate((PSNR_bl_ii[:15], PSNR_bl_ii[65:]))
    SSIM_pr_rw = np.concatenate((SSIM_pr_ii[:15], SSIM_pr_ii[65:])); PSNR_pr_rw = np.concatenate((PSNR_pr_ii[:15], PSNR_pr_ii[65:]))
    
    SSIM_all_rw_sh[ii, 0] = np.min(SSIM_ii); SSIM_all_rw_sh[ii, 1] = np.mean(SSIM_ii); SSIM_all_rw_sh[ii, 2] = np.max(SSIM_ii) 
    PSNR_all_rw_sh[ii, 0] = np.min(PSNR_ii); PSNR_all_rw_sh[ii, 1] = np.mean(PSNR_ii); PSNR_all_rw_sh[ii, 2] = np.max(PSNR_ii)
    print('Real World + Shapes Data')
    print('SSIM min, max, mean', np.min(SSIM_ii), np.max(SSIM_ii), np.mean(SSIM_ii))
    print('PSNR min, max, mean', np.min(PSNR_ii), np.max(PSNR_ii), np.mean(PSNR_ii))
    
    SSIM_bl_all_rw_sh[ii, 0] = np.min(SSIM_bl_ii); SSIM_bl_all_rw_sh[ii, 1] = np.mean(SSIM_bl_ii); SSIM_bl_all_rw_sh[ii, 2] = np.max(SSIM_bl_ii) 
    PSNR_bl_all_rw_sh[ii, 0] = np.min(PSNR_bl_ii); PSNR_bl_all_rw_sh[ii, 1] = np.mean(PSNR_bl_ii); PSNR_bl_all_rw_sh[ii, 2] = np.max(PSNR_bl_ii)
    print('SSIM blended min, max, mean', np.min(SSIM_bl_ii), np.max(SSIM_bl_ii), np.mean(SSIM_bl_ii))
    print('PSNR blended min, max, mean', np.min(PSNR_bl_ii), np.max(PSNR_bl_ii), np.mean(PSNR_bl_ii))
  
    SSIM_pr_all_rw_sh[ii, 0] = np.min(SSIM_pr_ii); SSIM_pr_all_rw_sh[ii, 1] = np.mean(SSIM_pr_ii); SSIM_pr_all_rw_sh[ii, 2] = np.max(SSIM_pr_ii) 
    PSNR_pr_all_rw_sh[ii, 0] = np.min(PSNR_pr_ii); PSNR_pr_all_rw_sh[ii, 1] = np.mean(PSNR_pr_ii); PSNR_pr_all_rw_sh[ii, 2] = np.max(PSNR_pr_ii)
    print('SSIM predicted min, max, mean', np.min(SSIM_pr_ii), np.max(SSIM_pr_ii), np.mean(SSIM_pr_ii))
    print('PSNR predicted min, max, mean', np.min(PSNR_pr_ii), np.max(PSNR_pr_ii), np.mean(PSNR_pr_ii))


 
    SSIM_all_rw[ii, 0] = np.min(SSIM_rw); SSIM_all_rw[ii, 1] = np.mean(SSIM_rw); SSIM_all_rw[ii, 2] = np.max(SSIM_rw) 
    PSNR_all_rw[ii, 0] = np.min(PSNR_rw); PSNR_all_rw[ii, 1] = np.mean(PSNR_rw); PSNR_all_rw[ii, 2] = np.max(PSNR_rw)

    SSIM_bl_all_rw[ii, 0] = np.min(SSIM_bl_rw); SSIM_bl_all_rw[ii, 1] = np.mean(SSIM_bl_rw); SSIM_bl_all_rw[ii, 2] = np.max(SSIM_bl_rw) 
    PSNR_bl_all_rw[ii, 0] = np.min(PSNR_bl_rw); PSNR_bl_all_rw[ii, 1] = np.mean(PSNR_bl_rw); PSNR_bl_all_rw[ii, 2] = np.max(PSNR_bl_rw)
 
    SSIM_pr_all_rw[ii, 0] = np.min(SSIM_pr_rw); SSIM_pr_all_rw[ii, 1] = np.mean(SSIM_pr_rw); SSIM_pr_all_rw[ii, 2] = np.max(SSIM_pr_rw) 
    PSNR_pr_all_rw[ii, 0] = np.min(PSNR_pr_rw); PSNR_pr_all_rw[ii, 1] = np.mean(PSNR_pr_rw); PSNR_pr_all_rw[ii, 2] = np.max(PSNR_pr_rw)


 
    SSIM_all_sh[ii, 0] = np.min(SSIM_sh); SSIM_all_sh[ii, 1] = np.mean(SSIM_sh); SSIM_all_sh[ii, 2] = np.max(SSIM_sh) 
    PSNR_all_sh[ii, 0] = np.min(PSNR_sh); PSNR_all_sh[ii, 1] = np.mean(PSNR_sh); PSNR_all_sh[ii, 2] = np.max(PSNR_sh)

    SSIM_bl_all_sh[ii, 0] = np.min(SSIM_bl_sh); SSIM_bl_all_sh[ii, 1] = np.mean(SSIM_bl_sh); SSIM_bl_all_sh[ii, 2] = np.max(SSIM_bl_sh) 
    PSNR_bl_all_sh[ii, 0] = np.min(PSNR_bl_sh); PSNR_bl_all_sh[ii, 1] = np.mean(PSNR_bl_sh); PSNR_bl_all_sh[ii, 2] = np.max(PSNR_bl_sh)
 
    SSIM_pr_all_sh[ii, 0] = np.min(SSIM_pr_sh); SSIM_pr_all_sh[ii, 1] = np.mean(SSIM_pr_sh); SSIM_pr_all_sh[ii, 2] = np.max(SSIM_pr_sh) 
    PSNR_pr_all_sh[ii, 0] = np.min(PSNR_pr_sh); PSNR_pr_all_sh[ii, 1] = np.mean(PSNR_pr_sh); PSNR_pr_all_sh[ii, 2] = np.max(PSNR_pr_sh)


np.savetxt(directory + '/metrics/SSIM_rw_sh.csv', np.concatenate((SSIM_all_rw_sh, ratio), axis=1), delimiter=',')
np.savetxt(directory + '/metrics/PSNR_rw_sh.csv', np.concatenate((PSNR_all_rw_sh, ratio), axis=1), delimiter=',')

np.savetxt(directory + '/metrics/SSIM_bl_rw_sh.csv', np.concatenate((SSIM_bl_all_rw_sh, ratio), axis=1), delimiter=',')
np.savetxt(directory + '/metrics/PSNR_bl_rw_sh.csv', np.concatenate((PSNR_bl_all_rw_sh, ratio), axis=1), delimiter=',')

np.savetxt(directory + '/metrics/SSIM_pr_rw_sh.csv', np.concatenate((SSIM_pr_all_rw_sh, ratio), axis=1), delimiter=',')
np.savetxt(directory + '/metrics/PSNR_pr_rw_sh.csv', np.concatenate((PSNR_pr_all_rw_sh, ratio), axis=1), delimiter=',')



np.savetxt(directory + '/metrics/SSIM_rw.csv', np.concatenate((SSIM_all_rw, ratio), axis=1), delimiter=',')
np.savetxt(directory + '/metrics/PSNR_rw.csv', np.concatenate((PSNR_all_rw, ratio), axis=1), delimiter=',')

np.savetxt(directory + '/metrics/SSIM_bl_rw.csv', np.concatenate((SSIM_bl_all_rw, ratio), axis=1), delimiter=',')
np.savetxt(directory + '/metrics/PSNR_bl_rw.csv', np.concatenate((PSNR_bl_all_rw, ratio), axis=1), delimiter=',')

np.savetxt(directory + '/metrics/SSIM_pr_rw.csv', np.concatenate((SSIM_pr_all_rw, ratio), axis=1), delimiter=',')
np.savetxt(directory + '/metrics/PSNR_pr_rw.csv', np.concatenate((PSNR_pr_all_rw, ratio), axis=1), delimiter=',')



np.savetxt(directory + '/metrics/SSIM_sh.csv', np.concatenate((SSIM_all_sh, ratio), axis=1), delimiter=',')
np.savetxt(directory + '/metrics/PSNR_sh.csv', np.concatenate((PSNR_all_sh, ratio), axis=1), delimiter=',')

np.savetxt(directory + '/metrics/SSIM_bl_sh.csv', np.concatenate((SSIM_bl_all_sh, ratio), axis=1), delimiter=',')
np.savetxt(directory + '/metrics/PSNR_bl_sh.csv', np.concatenate((PSNR_bl_all_sh, ratio), axis=1), delimiter=',')

np.savetxt(directory + '/metrics/SSIM_pr_sh.csv', np.concatenate((SSIM_pr_all_sh, ratio), axis=1), delimiter=',')
np.savetxt(directory + '/metrics/PSNR_pr_sh.csv', np.concatenate((PSNR_pr_all_sh, ratio), axis=1), delimiter=',')

