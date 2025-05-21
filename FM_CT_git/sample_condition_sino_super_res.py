from ldm_inverse.condition_methods import get_conditioning_method
from ldm.models.diffusion.ddim_git_admm_match_sf_all import DDIMSampler
from data.dataloader import get_dataset, get_dataloader
from scripts.utils import clear_color, mask_generator
import matplotlib.pyplot as plt
from ldm_inverse.measurements import get_noise, get_operator
from functools import partial
import numpy as np
from model_loader import load_model_from_config, load_yaml
import os
import torch
import torchvision.transforms as transforms
import argparse
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from skimage.metrics import peak_signal_noise_ratio as psnr
import dxchange
import skimage


def get_model(args):
    config = OmegaConf.load(args.ldm_config)
    print('config', config)
    model = load_model_from_config(config, args.diffusion_config)

    return model


parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str)
parser.add_argument('--ldm_config', default="/homes/sruban/latent-diffusion-inpainting_sino_trial/ldm/models/ldm/inpainting_big/config_rl_no_mask.yaml", type=str)
parser.add_argument('--diffusion_config', default="/homes/sruban/latent-diffusion-inpainting_sino_trial/LDM_Sinogram_uncond/epoch=000205.ckpt", type=str)
parser.add_argument('--task_config', default="configs/tasks/super_resolution_config_512_4x.yaml", type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='./results_512/sinogram/six_samples_smc/super_res/4x')
parser.add_argument('--ddim_steps', default=500, type=int)
parser.add_argument('--ddim_eta', default=0.0, type=float)
parser.add_argument('--n_samples_per_class', default=1, type=int)
parser.add_argument('--ddim_scale', default=1.0, type=float)

args = parser.parse_args()


# Load configurations
task_config = load_yaml(args.task_config)

# Device setting
device_str = f"cuda:0" if torch.cuda.is_available() else 'cpu'
print(f"Device set to {device_str}.")
device = torch.device(device_str)  
print('I am here')

# Loading model
model = get_model(args)
print('I am here1')
sampler = DDIMSampler(model) # Sampling using DDIM
print('I am here2')

# Prepare Operator and noise
measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])
noiser = get_noise(**measure_config['noise'])
print(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

# Prepare conditioning method
cond_config = task_config['conditioning']
cond_method = get_conditioning_method(cond_config['method'], model, operator, noiser, **cond_config['params'])
measurement_cond_fn = cond_method.conditioning
print(f"Conditioning sampler : {task_config['conditioning']['main_sampler']}")

# Working directory
out_path = os.path.join(args.save_dir)
os.makedirs(out_path, exist_ok=True)
for img_dir in ['input', 'recon', 'progress', 'label']:
    os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

# Prepare dataloader
data_config = task_config['data']
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )
dataset = get_dataset(**data_config, transforms=transform)
loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

# Exception) In case of inpainting, we need to generate a mask 
if measure_config['operator']['name'] == 'inpainting':
  mask_gen = mask_generator(**measure_config['mask_opt'])

# Do inference
for i, ref_img in enumerate(loader):
    
    print(ref_img.shape)
    print(f"Inference for image {i}")

    fname = str(i).zfill(3)
    ref_img = ref_img.to(device)
    #plt.imsave('ref_img1.png', np.transpose(np.squeeze(ref_img.cpu().numpy()), (1, 2, 0))) 

    # Exception) In case of inpainting
    if measure_config['operator'] ['name'] == 'inpainting':
      mask = mask_gen(ref_img)
      mask = mask[:, 0, :, :].unsqueeze(dim=0)
       
    c = torch.zeros(1,4,128,128).to(device)

    # Instantiating sampler
    sample_fn = partial(sampler.posterior_sampler, measurement_cond_fn=measurement_cond_fn, operator_fn=operator.forward,
                                        S=args.ddim_steps,
                                        cond_method=task_config['conditioning']['main_sampler'],
                                        conditioning=c,
                                        ddim_use_original_steps=True,
                                        batch_size=args.n_samples_per_class,
                                        shape=[3, 128, 128], # Dimension of latent space
                                        verbose=False,
                                        unconditional_guidance_scale=args.ddim_scale,
                                        unconditional_conditioning=None, 
                                        eta=args.ddim_eta)
    
    # Exception) In case of inpainting
    if measure_config['operator'] ['name'] == 'inpainting':
      operator_fn = partial(operator.forward, mask=mask)
      measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
      sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn, operator_fn=operator_fn)

      # Forward measurement model
      y = operator_fn(ref_img)
      y_n = noiser(y)

    else:
      y = operator.forward(ref_img)
      y_n = noiser(y).to(device)
 
    # Sampling
    samples_ddim, _ = sample_fn(measurement=y_n)
    
    x_samples_ddim = model.decode_first_stage(samples_ddim.detach())

    # Post-processing samples
    label = skimage.color.rgb2gray(clear_color(y_n))
    reconstructed = skimage.color.rgb2gray(clear_color(x_samples_ddim))
    true = skimage.color.rgb2gray(clear_color(ref_img))

    # Saving images
    dxchange.write_tiff(true, os.path.join(out_path, 'input', fname+'_true'), dtype='float32', overwrite=True)
    dxchange.write_tiff(label, os.path.join(out_path, 'label', fname+'_label'), dtype='float32', overwrite=True)
    dxchange.write_tiff(reconstructed, os.path.join(out_path, 'recon', fname+'_recon'), dtype='float32', overwrite=True)
    
    '''
    plt.imsave(os.path.join(out_path, 'input', fname+'_true.png'), true)
    plt.imsave(os.path.join(out_path, 'label', fname+'_label.png'), label)
    plt.imsave(os.path.join(out_path, 'recon', fname+'_recon.png'), reconstructed)
    '''

    psnr_cur = psnr(true, reconstructed)

    print('PSNR:', psnr_cur)
