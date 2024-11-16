import argparse, os, sys, glob
sys.path.append(os.getcwd()+"/ldm")
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import h5py
from natsort import natsorted
from ldm.util import instantiate_from_config
#from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import Dataset, DataLoader
import dxchange
import skimage
import torchvision.transforms as T
transform = T.ToPILImage()

directory_s = 'results/Jiaze_data/w_blending/ldm_real_only/random_masking/2D_sinograms'
directory_r = 'results/Jiaze_data/w_blending/ldm_real_only/random_masking/2D_recons'
subfolders1 = [f.path for f in os.scandir(directory_s) if f.is_dir()]
print(subfolders1)
subfolders= natsorted(subfolders1)[:-1]
print(subfolders)
print(len(subfolders))

SSIM_RW_sh_oo = np.zeros((len(subfolders), 3))
PSNR_RW_sh_oo = np.zeros((len(subfolders), 3))

SSIM_RW_sh_om = np.zeros((len(subfolders), 3))
PSNR_RW_sh_om = np.zeros((len(subfolders), 3))

SSIM_RW_sh_o_bl = np.zeros((len(subfolders), 3))
PSNR_RW_sh_o_bl = np.zeros((len(subfolders), 3))


SSIM_RW_oo = np.zeros((len(subfolders), 3))
PSNR_RW_oo = np.zeros((len(subfolders), 3))

SSIM_RW_om = np.zeros((len(subfolders), 3))
PSNR_RW_om = np.zeros((len(subfolders), 3))

SSIM_RW_o_bl = np.zeros((len(subfolders), 3))
PSNR_RW_o_bl = np.zeros((len(subfolders), 3))


SSIM_sh_oo = np.zeros((len(subfolders), 3))
PSNR_sh_oo = np.zeros((len(subfolders), 3))

SSIM_sh_om = np.zeros((len(subfolders), 3))
PSNR_sh_om = np.zeros((len(subfolders), 3))

SSIM_sh_o_bl = np.zeros((len(subfolders), 3))
PSNR_sh_o_bl = np.zeros((len(subfolders), 3))



ratio = np.transpose(np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]))

for ii in range(len(subfolders)):
    print('ii', ii)
    subfolder_ii = subfolders[ii]

    output_folder = subfolder_ii + '/inpainted_data.tiff'
    original_folder = subfolder_ii + '/input_data.tiff'
    mask_folder = subfolder_ii + '/mask_data.tiff'
    blended_folder = subfolder_ii + '/blended_data.tiff'
    
    blended = dxchange.read_tiff(blended_folder).copy()
    output = dxchange.read_tiff(output_folder).copy()
    original = dxchange.read_tiff(original_folder).copy()
    mask = dxchange.read_tiff(mask_folder).copy()
    mask_data = (1 - mask) * original

    nums = 100

    angs = np.linspace(0, 180, 512, endpoint=False)
    
    rec_op = np.zeros((nums, 512, 512)); rec_org = np.zeros((nums, 512, 512))
    rec_mask = np.zeros((nums, 512, 512)); rec_blend = np.zeros((nums, 512, 512))
    
    SSIM_ii_oo = np.zeros(nums); PSNR_ii_oo = np.zeros(nums)
    SSIM_ii_om = np.zeros(nums); PSNR_ii_om = np.zeros(nums)
    SSIM_ii_o_bl = np.zeros(nums); PSNR_ii_o_bl = np.zeros(nums)
    

    for iii in range(nums):

        print('iii', iii)
        original_iii = original[iii, :, :]
        output_iii = output[iii, :, :]
        mask_iii = mask_data[iii, :, :]
        blended_iii = blended[iii, :, :]
        
        '''
        print(np.min(original_iii), np.max(original_iii))
        print(np.min(output_iii), np.max(output_iii))
        print(np.min(mask_iii), np.max(mask_iii))
        print(np.min(blended_iii), np.max(blended_iii))
        '''

        rec_op_iii = skimage.transform.iradon(np.transpose(output_iii), angs)
        rec_op[iii, :, :] = rec_op_iii

        rec_org_iii = skimage.transform.iradon(np.transpose(original_iii), angs)
        rec_org[iii, :, :] = rec_org_iii

        rec_mask_iii = skimage.transform.iradon(np.transpose(mask_iii), angs)
        rec_mask[iii, :, :] = rec_mask_iii

        rec_bl_iii = skimage.transform.iradon(np.transpose(blended_iii), angs)
        rec_blend[iii, :, :] = rec_bl_iii


        dr_org = np.max(rec_org_iii) - np.min(rec_org_iii)
        dr_op = np.max(rec_op_iii) - np.min(rec_op_iii)
        dr_mask = np.max(rec_mask_iii) - np.min(rec_mask_iii)
        dr_blend = np.max(rec_bl_iii) - np.min(rec_bl_iii)
        print('org, op and mask data range', dr_org, dr_op, dr_mask, dr_blend)
        #dr = max(dr_org, dr_op, dr_mask, dr_blend)

        SSIM_ii_oo[iii] = skimage.metrics.structural_similarity(rec_org_iii, rec_op_iii, data_range=max(dr_org, dr_op))
        PSNR_ii_oo[iii] = skimage.metrics.peak_signal_noise_ratio(rec_org_iii, rec_op_iii, data_range=max(dr_org, dr_op))

        SSIM_ii_om[iii] = skimage.metrics.structural_similarity(rec_org_iii, rec_mask_iii, data_range=max(dr_org, dr_mask))
        PSNR_ii_om[iii] = skimage.metrics.peak_signal_noise_ratio(rec_org_iii, rec_mask_iii, data_range=max(dr_org, dr_mask))

        SSIM_ii_o_bl[iii] = skimage.metrics.structural_similarity(rec_org_iii, rec_bl_iii, data_range=max(dr_org, dr_blend))
        PSNR_ii_o_bl[iii] = skimage.metrics.peak_signal_noise_ratio(rec_org_iii, rec_bl_iii, data_range=max(dr_org, dr_blend))

        print('SSIM_ii_oo, PSNR_ii_oo', SSIM_ii_oo[iii], PSNR_ii_oo[iii])
        print('SSIM_ii_om, PSNR_ii_om', SSIM_ii_om[iii], PSNR_ii_om[iii])
        print('SSIM_ii_o_bl, PSNR_ii_o_bl', SSIM_ii_o_bl[iii], PSNR_ii_o_bl[iii])

    dxchange.write_tiff(rec_mask, directory_r + '/maskr' + str(ratio[ii]) + '/mask', dtype='float32', overwrite=True)
    dxchange.write_tiff(rec_org, directory_r + '/maskr' + str(ratio[ii]) + '/orig', dtype='float32', overwrite=True)
    dxchange.write_tiff(rec_op, directory_r + '/maskr' + str(ratio[ii]) + '/output', dtype='float32', overwrite=True)
    dxchange.write_tiff(rec_blend, directory_r + '/maskr' + str(ratio[ii]) + '/blend_lr_0p01', dtype='float32', overwrite=True)

    SSIM_ii_oo_sh = SSIM_ii_oo[15:65]; PSNR_ii_oo_sh = PSNR_ii_oo[15:65]
    SSIM_ii_om_sh = SSIM_ii_om[15:65]; PSNR_ii_om_sh = PSNR_ii_om[15:65]
    SSIM_ii_o_bl_sh = SSIM_ii_o_bl[15:65]; PSNR_ii_o_bl_sh = PSNR_ii_o_bl[15:65]


    SSIM_ii_oo_rw = np.concatenate((SSIM_ii_oo[:15], SSIM_ii_oo[65:])); PSNR_ii_oo_rw = np.concatenate((PSNR_ii_oo[:15], PSNR_ii_oo[65:]))
    SSIM_ii_om_rw = np.concatenate((SSIM_ii_om[:15], SSIM_ii_om[65:])); PSNR_ii_om_rw = np.concatenate((PSNR_ii_om[:15], PSNR_ii_om[65:]))
    SSIM_ii_o_bl_rw = np.concatenate((SSIM_ii_o_bl[:15], SSIM_ii_o_bl[65:])); PSNR_ii_o_bl_rw = np.concatenate((PSNR_ii_o_bl[:15], PSNR_ii_o_bl[65:]))


    print('Real World + Shapes Data')
    SSIM_RW_sh_oo[ii, 0] = np.min(SSIM_ii_oo); SSIM_RW_sh_oo[ii, 1] = np.mean(SSIM_ii_oo); SSIM_RW_sh_oo[ii, 2] = np.max(SSIM_ii_oo)
    PSNR_RW_sh_oo[ii, 0] = np.min(PSNR_ii_oo); PSNR_RW_sh_oo[ii, 1] = np.mean(PSNR_ii_oo); PSNR_RW_sh_oo[ii, 2] = np.max(PSNR_ii_oo)

    SSIM_RW_sh_om[ii, 0] = np.min(SSIM_ii_om); SSIM_RW_sh_om[ii, 1] = np.mean(SSIM_ii_om); SSIM_RW_sh_om[ii, 2] = np.max(SSIM_ii_om)
    PSNR_RW_sh_om[ii, 0] = np.min(PSNR_ii_om); PSNR_RW_sh_om[ii, 1] = np.mean(PSNR_ii_om); PSNR_RW_sh_om[ii, 2] = np.max(PSNR_ii_om)
 
    SSIM_RW_sh_o_bl[ii, 0] = np.min(SSIM_ii_o_bl); SSIM_RW_sh_o_bl[ii, 1] = np.mean(SSIM_ii_o_bl); SSIM_RW_sh_o_bl[ii, 2] = np.max(SSIM_ii_o_bl)
    PSNR_RW_sh_o_bl[ii, 0] = np.min(PSNR_ii_o_bl); PSNR_RW_sh_o_bl[ii, 1] = np.mean(PSNR_ii_o_bl); PSNR_RW_sh_o_bl[ii, 2] = np.max(PSNR_ii_o_bl)
    

    print('Real World Data')
    SSIM_RW_oo[ii, 0] = np.min(SSIM_ii_oo_rw); SSIM_RW_oo[ii, 1] = np.mean(SSIM_ii_oo_rw); SSIM_RW_oo[ii, 2] = np.max(SSIM_ii_oo_rw)
    PSNR_RW_oo[ii, 0] = np.min(PSNR_ii_oo_rw); PSNR_RW_oo[ii, 1] = np.mean(PSNR_ii_oo_rw); PSNR_RW_oo[ii, 2] = np.max(PSNR_ii_oo_rw)

    SSIM_RW_om[ii, 0] = np.min(SSIM_ii_om_rw); SSIM_RW_om[ii, 1] = np.mean(SSIM_ii_om_rw); SSIM_RW_om[ii, 2] = np.max(SSIM_ii_om_rw)
    PSNR_RW_om[ii, 0] = np.min(PSNR_ii_om_rw); PSNR_RW_om[ii, 1] = np.mean(PSNR_ii_om_rw); PSNR_RW_om[ii, 2] = np.max(PSNR_ii_om_rw)
 
    SSIM_RW_o_bl[ii, 0] = np.min(SSIM_ii_o_bl_rw); SSIM_RW_o_bl[ii, 1] = np.mean(SSIM_ii_o_bl_rw); SSIM_RW_o_bl[ii, 2] = np.max(SSIM_ii_o_bl_rw)
    PSNR_RW_o_bl[ii, 0] = np.min(PSNR_ii_o_bl_rw); PSNR_RW_o_bl[ii, 1] = np.mean(PSNR_ii_o_bl_rw); PSNR_RW_o_bl[ii, 2] = np.max(PSNR_ii_o_bl_rw)
 
    
    print('Shapes Data')
    SSIM_sh_oo[ii, 0] = np.min(SSIM_ii_oo_sh); SSIM_sh_oo[ii, 1] = np.mean(SSIM_ii_oo_sh); SSIM_sh_oo[ii, 2] = np.max(SSIM_ii_oo_sh)
    PSNR_sh_oo[ii, 0] = np.min(PSNR_ii_oo_sh); PSNR_sh_oo[ii, 1] = np.mean(PSNR_ii_oo_sh); PSNR_sh_oo[ii, 2] = np.max(PSNR_ii_oo_sh)

    SSIM_sh_om[ii, 0] = np.min(SSIM_ii_om_sh); SSIM_sh_om[ii, 1] = np.mean(SSIM_ii_om_sh); SSIM_sh_om[ii, 2] = np.max(SSIM_ii_om_sh)
    PSNR_sh_om[ii, 0] = np.min(PSNR_ii_om_sh); PSNR_sh_om[ii, 1] = np.mean(PSNR_ii_om_sh); PSNR_sh_om[ii, 2] = np.max(PSNR_ii_om_sh)
 
    SSIM_sh_o_bl[ii, 0] = np.min(SSIM_ii_o_bl_sh); SSIM_sh_o_bl[ii, 1] = np.mean(SSIM_ii_o_bl_sh); SSIM_sh_o_bl[ii, 2] = np.max(SSIM_ii_o_bl_sh)
    PSNR_sh_o_bl[ii, 0] = np.min(PSNR_ii_o_bl_sh); PSNR_sh_o_bl[ii, 1] = np.mean(PSNR_ii_o_bl_sh); PSNR_sh_o_bl[ii, 2] = np.max(PSNR_ii_o_bl_sh)
 

np.savetxt(directory_r + '/metrics/SSIM_oo_rw_sh.csv', np.concatenate((SSIM_RW_sh_oo, ratio), axis=1), delimiter=',')
np.savetxt(directory_r + '/metrics/PSNR_oo_rw_sh.csv', np.concatenate((PSNR_RW_sh_oo, ratio), axis=1), delimiter=',')

np.savetxt(directory_r + '/metrics/SSIM_om_rw_sh.csv', np.concatenate((SSIM_RW_sh_om, ratio), axis=1), delimiter=',')
np.savetxt(directory_r + '/metrics/PSNR_om_rw_sh.csv', np.concatenate((PSNR_RW_sh_om, ratio), axis=1), delimiter=',')

np.savetxt(directory_r + '/metrics/SSIM_o_bl_rw_sh.csv', np.concatenate((SSIM_RW_sh_o_bl, ratio), axis=1), delimiter=',')
np.savetxt(directory_r + '/metrics/PSNR_o_bl_rw_sh.csv', np.concatenate((PSNR_RW_sh_o_bl, ratio), axis=1), delimiter=',')



np.savetxt(directory_r + '/metrics/SSIM_oo_rw.csv', np.concatenate((SSIM_RW_oo, ratio), axis=1), delimiter=',')
np.savetxt(directory_r + '/metrics/PSNR_oo_rw.csv', np.concatenate((PSNR_RW_oo, ratio), axis=1), delimiter=',')

np.savetxt(directory_r + '/metrics/SSIM_om_rw.csv', np.concatenate((SSIM_RW_om, ratio), axis=1), delimiter=',')
np.savetxt(directory_r + '/metrics/PSNR_om_rw.csv', np.concatenate((PSNR_RW_om, ratio), axis=1), delimiter=',')

np.savetxt(directory_r + '/metrics/SSIM_o_bl_rw.csv', np.concatenate((SSIM_RW_o_bl, ratio), axis=1), delimiter=',')
np.savetxt(directory_r + '/metrics/PSNR_o_bl_rw.csv', np.concatenate((PSNR_RW_o_bl, ratio), axis=1), delimiter=',')


 
np.savetxt(directory_r + '/metrics/SSIM_oo_sh.csv', np.concatenate((SSIM_sh_oo, ratio), axis=1), delimiter=',')
np.savetxt(directory_r + '/metrics/PSNR_oo_sh.csv', np.concatenate((PSNR_sh_oo, ratio), axis=1), delimiter=',')

np.savetxt(directory_r + '/metrics/SSIM_om_sh.csv', np.concatenate((SSIM_sh_om, ratio), axis=1), delimiter=',')
np.savetxt(directory_r + '/metrics/PSNR_om_sh.csv', np.concatenate((PSNR_sh_om, ratio), axis=1), delimiter=',')

np.savetxt(directory_r + '/metrics/SSIM_o_bl_sh.csv', np.concatenate((SSIM_sh_o_bl, ratio), axis=1), delimiter=',')
np.savetxt(directory_r + '/metrics/PSNR_o_bl_sh.csv', np.concatenate((PSNR_sh_o_bl, ratio), axis=1), delimiter=',')
  
