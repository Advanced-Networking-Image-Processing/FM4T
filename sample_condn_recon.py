from ldm_inverse.condition_methods import get_conditioning_method
from ldm.models.diffusion.p1e_4_pg20_gamma1e4 import DDIMSampler
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
    model = load_model_from_config(config, args.diffusion_config)
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str)
parser.add_argument('--ldm_config', default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml", type=str)
parser.add_argument('--diffusion_config', default="models/ldm/model.ckpt", type=str)
parser.add_argument('--task_config', default="configs/tasks/CT_recon_config.yaml", type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='./results/trial_May1_25/CT_recon/sino_samples_256/admm_match_sf1_p1e_4/pg20_gamma_var/gamma1e4')
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
# Identical upto this point

# Prepare Operator and noise
measure_config = task_config['measurement']
print('measure_config', measure_config)
operator = get_operator(device=device, **measure_config['operator'])
noiser = get_noise(**measure_config['noise'])
#print(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")
#print('operator.forward', operator.forward)

# Prepare conditioning method
cond_config = task_config['conditioning']
print('cond_config', cond_config)
#blabla
cond_method = get_conditioning_method(cond_config['method'], model, operator, noiser, **cond_config['params'])
measurement_cond_fn = cond_method.conditioning
#print('measurement_cond_fn', measurement_cond_fn)
print(f"Conditioning sampler : {task_config['conditioning']['main_sampler']}")

# Instantiating sampler
sample_fn = partial(sampler.posterior_sampler, measurement_cond_fn=measurement_cond_fn, operator_fn=operator.forward,
                                        S=args.ddim_steps,
                                        cond_method=task_config['conditioning']['main_sampler'],
                                        conditioning=None,
                                        ddim_use_original_steps=True,
                                        batch_size=args.n_samples_per_class,
                                        shape=[3, 64, 64], # Dimension of latent space
                                        verbose=False,
                                        unconditional_guidance_scale=args.ddim_scale,
                                        unconditional_conditioning=None, 
                                        eta=args.ddim_eta)


# Working directory
out_path = os.path.join(args.save_dir)
os.makedirs(out_path, exist_ok=True)

'''
for img_dir in ['input', 'recon', 'progress', 'label']:
    os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
'''

# Prepare dataloader
data_config = task_config['data']
transform = transforms.Compose([transforms.ToTensor()])
#transform = transforms.Compose([transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )
dataset = get_dataset(**data_config, transforms=transform)
print('dataset', dataset)
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
    print('ref_img shape', ref_img.shape)

    # Instantiating sampler
    sample_fn = partial(sampler.posterior_sampler, measurement_cond_fn=measurement_cond_fn, operator_fn=operator.forward,
                                        S=args.ddim_steps,
                                        cond_method=task_config['conditioning']['main_sampler'],
                                        conditioning=None,
                                        ddim_use_original_steps=True,
                                        batch_size=args.n_samples_per_class,
                                        shape=[3, 64, 64], # Dimension of latent space
                                        verbose=False,
                                        unconditional_guidance_scale=args.ddim_scale,
                                        unconditional_conditioning=None, 
                                        eta=args.ddim_eta)


    # Exception) In case of inpainting
    if measure_config['operator'] ['name'] == 'inpainting':
      mask = mask_gen(ref_img)
      print('mask shape', mask.shape)
      mask = mask[:, 0, :, :].unsqueeze(dim=0)
      print('mask shape', mask.shape)
      operator_fn = partial(operator.forward, mask=mask)
      #print('operator_fn.shape', operator_fn.shape)
      measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
      #print('measurement_cond_fn.shape', measurment_cond_fn.shape)
      sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn, operator_fn=operator_fn)
      #print('sample_fn.shape', sample_fn.shape)

      # Forward measurement model
      y = operator_fn(ref_img)
      print('y.shape', y.shape)
      y_n = noiser(y)
      print('y_n shape', y_n.shape)

    else:
      y = operator.forward(ref_img)
      y_n = noiser(y).to(device)

    # Sampling
    # This is where we have the iterations
    print('y_n shape', y_n.shape)
    # Everything cooks over here. The rest of the code is supporting functions.
    samples_ddim, _ = sample_fn(measurement=y_n)
    print('x_samples_ddim in line #162e', np.min(samples_ddim.detach().cpu().numpy()), np.max(samples_ddim.detach().cpu().numpy()))

    x_samples_ddim = model.decode_first_stage(samples_ddim.detach())
    
    print('y_n min and max', np.min(y_n.detach().cpu().numpy()), np.max(y_n.detach().cpu().numpy()))
    #print('x_samples_ddim min and max', np.min(x_samples_ddim.detach().cpu().numpy()), np.max(x_samples_ddim.detach().cpu().numpy()))
    print('ref_img min and max', np.min(ref_img.detach().cpu().numpy()), np.max(ref_img.detach().cpu().numpy()))
    
    print('y_n shape', y_n.shape)
    print('x_samples_ddim shape', x_samples_ddim.shape)
    print('ref_img shape', ref_img.shape)
    
    #y_n = np.transpose(y_n.detach().cpu(), (0, 3, 1, 2))
    # Post-processing samples
    label = clear_color(y_n)
    #print('label shape', label.shape)
    reconstructed = clear_color(x_samples_ddim)
    #print('reconstructed shape', reconstructed.shape)
    true = clear_color(ref_img)
    #print('true shape', true.shape)

    print('label min and max', np.min(label), np.max(label))
    print('reconstructed min and max', np.min(reconstructed), np.max(reconstructed))
    print('true min and max', np.min(true), np.max(true))

    # Saving images
    dxchange.write_tiff(skimage.color.rgb2gray(true), os.path.join(out_path, 'input', fname+'_true'), dtype='float32', overwrite=True)
    dxchange.write_tiff(skimage.color.rgb2gray(label), os.path.join(out_path, 'label', fname+'_label'), dtype='float32', overwrite=True)
    dxchange.write_tiff(skimage.color.rgb2gray(reconstructed), os.path.join(out_path, 'recon', fname+'_recon'), dtype='float32', overwrite=True)
   
