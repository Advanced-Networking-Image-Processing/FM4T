"""
Conditional reconstruction sample generation script using latent diffusion models.

This script performs conditional reconstruction tasks (e.g., CT reconstruction, inpainting, 
super-resolution) using latent diffusion models with DDIM sampling.
"""

import argparse
import os
from functools import partial

import numpy as np
import torch
import torchvision.transforms as transforms
import dxchange
import skimage
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio as psnr

from ldm_inverse.condition_methods import get_conditioning_method
from ldm_inverse.measurements import get_noise, get_operator
from ldm.models.diffusion.p1e_4_pg20_gamma1e4 import DDIMSampler
from data.dataloader import get_dataset, get_dataloader
from scripts.utils import clear_color, mask_generator
from model_loader import load_model_from_config, load_yaml

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Conditional reconstruction using latent diffusion models"
    )
    parser.add_argument('--model_config', type=str, help='Model configuration file path')
    parser.add_argument(
        '--ldm_config', 
        default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml", 
        type=str,
        help='LDM configuration file path'
    )
    parser.add_argument(
        '--diffusion_config', 
        default="models/ldm/model.ckpt", 
        type=str,
        help='Diffusion model checkpoint path'
    )
    parser.add_argument(
        '--task_config', 
        default="configs/tasks/CT_recon_config.yaml", 
        type=str,
        help='Task configuration file path'
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='./results/trial_May1_25/CT_recon/sino_samples_256/admm_match_sf1_p1e_4/pg20_gamma_var/gamma1e4',
        help='Output directory for saving results'
    )
    parser.add_argument('--ddim_steps', default=500, type=int, help='Number of DDIM steps')
    parser.add_argument('--ddim_eta', default=0.0, type=float, help='DDIM eta parameter')
    parser.add_argument(
        '--n_samples_per_class', 
        default=1, 
        type=int, 
        help='Number of samples per class'
    )
    parser.add_argument(
        '--ddim_scale', 
        default=1.0, 
        type=float, 
        help='DDIM guidance scale'
    )
    return parser.parse_args()


def get_model(args):
    """Load and return the diffusion model."""
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_config)
    return model


def setup_device():
    """Setup and return the appropriate device."""
    device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
    print(f"Device set to {device_str}.")
    return torch.device(device_str)


def setup_measurement_operator(measure_config, device):
    """Setup measurement operator and noise."""
    print('measure_config:', measure_config)
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    return operator, noiser


def setup_conditioning_method(cond_config, model, operator, noiser):
    """Setup conditioning method."""
    print('cond_config:', cond_config)
    cond_method = get_conditioning_method(
        cond_config['method'], model, operator, noiser, **cond_config['params']
    )
    measurement_cond_fn = cond_method.conditioning
    print(f"Conditioning sampler: {cond_config['main_sampler']}")
    return cond_method, measurement_cond_fn


def create_sample_function(sampler, measurement_cond_fn, operator, args, task_config):
    """Create the sampling function with all necessary parameters."""
    return partial(
        sampler.posterior_sampler,
        measurement_cond_fn=measurement_cond_fn,
        operator_fn=operator.forward,
        S=args.ddim_steps,
        cond_method=task_config['conditioning']['main_sampler'],
        conditioning=None,
        ddim_use_original_steps=True,
        batch_size=args.n_samples_per_class,
        shape=[3, 64, 64],  # Dimension of latent space
        verbose=False,
        unconditional_guidance_scale=args.ddim_scale,
        unconditional_conditioning=None,
        eta=args.ddim_eta
    )


def setup_output_directories(save_dir):
    """Create output directories."""
    os.makedirs(save_dir, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(save_dir, img_dir), exist_ok=True)


def prepare_dataloader(data_config):
    """Prepare and return the data loader."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = get_dataset(**data_config, transforms=transform)
    print('dataset:', dataset)
    return get_dataloader(dataset, batch_size=1, num_workers=0, train=False)


def handle_inpainting_case(measure_config, operator, cond_method, ref_img):
    """Handle special case for inpainting tasks."""
    mask_gen = mask_generator(**measure_config['mask_opt'])
    mask = mask_gen(ref_img)
    print('mask shape:', mask.shape)
    mask = mask[:, 0, :, :].unsqueeze(dim=0)
    print('mask shape after processing:', mask.shape)
    
    operator_fn = partial(operator.forward, mask=mask)
    measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
    return operator_fn, measurement_cond_fn, mask


def perform_measurement(operator, noiser, ref_img, measure_config, device, cond_method=None):
    """Perform forward measurement with noise."""
    if measure_config['operator']['name'] == 'inpainting':
        operator_fn, measurement_cond_fn, mask = handle_inpainting_case(
            measure_config, operator, cond_method, ref_img
        )
        y = operator_fn(ref_img)
        print('y.shape (inpainting):', y.shape)
        y_n = noiser(y)
        print('y_n shape (inpainting):', y_n.shape)
        return y_n, operator_fn, measurement_cond_fn
    else:
        y = operator.forward(ref_img)
        y_n = noiser(y).to(device)
        return y_n, None, None


def save_results(fname, out_path, true_img, label_img, reconstructed_img):
    """Save the reconstruction results."""
    # Convert to grayscale and save
    true_gray = skimage.color.rgb2gray(true_img)
    label_gray = skimage.color.rgb2gray(label_img)
    recon_gray = skimage.color.rgb2gray(reconstructed_img)
    
    dxchange.write_tiff(
        true_gray, 
        os.path.join(out_path, 'input', f'{fname}_true'), 
        dtype='float32', 
        overwrite=True
    )
    dxchange.write_tiff(
        label_gray, 
        os.path.join(out_path, 'label', f'{fname}_label'), 
        dtype='float32', 
        overwrite=True
    )
    dxchange.write_tiff(
        recon_gray, 
        os.path.join(out_path, 'recon', f'{fname}_recon'), 
        dtype='float32', 
        overwrite=True
    )


def run_inference(args, task_config, model, sampler, operator, noiser, measurement_cond_fn, 
                 cond_method, device):
    """Main inference loop."""
    # Setup data loader
    data_config = task_config['data']
    loader = prepare_dataloader(data_config)
    
    # Setup output directories  
    setup_output_directories(args.save_dir)
    
    # Get measurement configuration
    measure_config = task_config['measurement']
    
    for i, ref_img in enumerate(loader):
        print(f"Inference for image {i}")
        print(f"Input image shape: {ref_img.shape}")
        
        fname = str(i).zfill(3)
        ref_img = ref_img.to(device)
        
        # Create base sample function
        sample_fn = create_sample_function(
            sampler, measurement_cond_fn, operator, args, task_config
        )
        
        # Perform measurement
        y_n, operator_fn, updated_measurement_cond_fn = perform_measurement(
            operator, noiser, ref_img, measure_config, device, cond_method
        )
        
        # Update sample function for inpainting case
        if operator_fn is not None and updated_measurement_cond_fn is not None:
            sample_fn = partial(
                sample_fn, 
                measurement_cond_fn=updated_measurement_cond_fn, 
                operator_fn=operator_fn
            )
        
        print(f'Measurement shape: {y_n.shape}')
        
        # Perform sampling (main reconstruction step)
        samples_ddim, _ = sample_fn(measurement=y_n)
        print(f'Samples DDIM range: [{np.min(samples_ddim.detach().cpu().numpy()):.4f}, '
              f'{np.max(samples_ddim.detach().cpu().numpy()):.4f}]')
        
        # Decode from latent space
        x_samples_ddim = model.decode_first_stage(samples_ddim.detach())
        
        # Print statistics
        print_statistics(y_n, x_samples_ddim, ref_img)
        
        # Post-process samples
        label = clear_color(y_n)
        reconstructed = clear_color(x_samples_ddim)
        true = clear_color(ref_img)
        
        print_processed_statistics(label, reconstructed, true)
        
        # Save results
        save_results(fname, args.save_dir, true, label, reconstructed)


def print_statistics(y_n, x_samples_ddim, ref_img):
    """Print min/max statistics for debugging."""
    print(f'y_n range: [{np.min(y_n.detach().cpu().numpy()):.4f}, '
          f'{np.max(y_n.detach().cpu().numpy()):.4f}]')
    print(f'ref_img range: [{np.min(ref_img.detach().cpu().numpy()):.4f}, '
          f'{np.max(ref_img.detach().cpu().numpy()):.4f}]')
    print(f'y_n shape: {y_n.shape}')
    print(f'x_samples_ddim shape: {x_samples_ddim.shape}')
    print(f'ref_img shape: {ref_img.shape}')


def print_processed_statistics(label, reconstructed, true):
    """Print statistics for processed images."""
    print(f'label range: [{np.min(label):.4f}, {np.max(label):.4f}]')
    print(f'reconstructed range: [{np.min(reconstructed):.4f}, {np.max(reconstructed):.4f}]')
    print(f'true range: [{np.min(true):.4f}, {np.max(true):.4f}]')


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configurations
    task_config = load_yaml(args.task_config)
    
    # Setup device
    device = setup_device()
    
    # Load model
    model = get_model(args)
    sampler = DDIMSampler(model)
    
    # Setup measurement operator and noise
    measure_config = task_config['measurement']
    operator, noiser = setup_measurement_operator(measure_config, device)
    
    # Setup conditioning method
    cond_config = task_config['conditioning']
    cond_method, measurement_cond_fn = setup_conditioning_method(
        cond_config, model, operator, noiser
    )
    
    # Run inference
    run_inference(
        args, task_config, model, sampler, operator, noiser, 
        measurement_cond_fn, cond_method, device
    )


if __name__ == "__main__":
    main()