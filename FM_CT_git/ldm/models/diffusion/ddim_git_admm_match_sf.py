"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from scripts.utils import *

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor
from skimage.metrics import peak_signal_noise_ratio as psnr
from scripts.utils import clear_color
import dxchange
import skimage


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        if ddim_num_steps < 1000:
          ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                    ddim_timesteps=self.ddim_timesteps,
                                                                                    eta=ddim_eta,verbose=verbose)
          self.register_buffer('ddim_sigmas', ddim_sigmas)
          self.register_buffer('ddim_alphas', ddim_alphas)
          self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
          self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
              (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                          1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        """
        Sampling wrapper function for UNCONDITIONAL sampling.
        """

        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates


    def posterior_sampler(self, measurement, measurement_cond_fn, operator_fn,
               S,
               batch_size,
               shape,
               cond_method=None,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        """
        Sampling wrapper function for inverse problem solving.
        """
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        print('I am in line #158 of DDIM.py in the posterior_sampler class')
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        print('I am in line #164 of DDIM.py')
        print('measurement_cond_fn', measurement_cond_fn)

        if cond_method is None or cond_method == 'resample':
            samples, intermediates = self.resample_sampling(measurement, measurement_cond_fn,
                                                    conditioning, size,
                                                        operator_fn=operator_fn,
                                                        callback=callback,
                                                        img_callback=img_callback,
                                                        quantize_denoised=quantize_x0,
                                                        mask=mask, x0=x0,
                                                        ddim_use_original_steps=False,
                                                        noise_dropout=noise_dropout,
                                                        temperature=temperature,
                                                        score_corrector=score_corrector,
                                                        corrector_kwargs=corrector_kwargs,
                                                        x_T=x_T,
                                                        log_every_t=log_every_t,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=unconditional_conditioning,
                                                        )
            
        else:
            raise ValueError(f"Condition method string '{cond_method}' not recognized.")
        
        return samples, intermediates


    def resample_sampling(self, measurement, measurement_cond_fn, cond, shape, operator_fn=None,
                     inter_timesteps=10, x_T=None, ddim_use_original_steps=False,
                     callback=None, timesteps=None, quantize_denoised=False,
                     mask=None, x0=None, img_callback=None, log_every_t=100,
                     temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                     unconditional_guidance_scale=1., unconditional_conditioning=None,):
        """
        DDIM-based sampling function for ReSample.

        Arguments:
            measurement:            Measurement vector y in y=Ax+n.
            measurement_cond_fn:    Function to perform DPS. 
            operator_fn:            Operator to perform forward operation A(.)
            inter_timesteps:        Number of timesteps to perform time travelling.

        """

        device = self.model.betas.device
        b = shape[0]
        print('x_T', x_T)
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        # Need for measurement consistency
        alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas 
        alphas_prev = self.model.alphas_cumprod_prev if ddim_use_original_steps else self.ddim_alphas_prev
        betas = self.model.betas
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        
        #print('cond', cond)
        

        # Initializing the ADMM variables z, p, and, v
        img_z = img
        img_p = img_z * 1
        img_v = torch.zeros_like(img_z) # dual variable
        img_z = img_z.requires_grad_()
        
        phy_steps = int(total_steps * 0.67)
        angles = torch.tensor(np.linspace(0, np.pi, 256, endpoint=False))
    
        for i, step in enumerate(iterator):   
            print('i', i)
            # Instantiating parameters
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device, requires_grad=False) # Needed for ReSampling
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device, requires_grad=False) # Needed for ReSampling
            b_t = torch.full((b, 1, 1, 1), betas[index], device=device, requires_grad=False)            
            
            img_p = img_p.requires_grad_()
            
            if i == 0:
                img_z = img_z
                ub = 0.01
                lb = -0.01
            else:
                img_z = img_p - img_v

            #print('img shape', img.shape)

            # Unconditional sampling step
            # z-update step
            # pred_z0 is from DDIM, img_z0_hat is computing \hat{x}_0 using Tweedie's formula
            img_z1, pred_z0, img_z0_hat = self.p_sample_ddim(img_z, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            
            # p-update step
            img_p = img_z1 + img_v
           
            if index < phy_steps:   
               
                # TODO: also make this not hard-coded
                if index % 10 == 0 :
                    
                    '''
                    nub = torch.quantile(img_z1_pixel, 0.99)
                    nlb = torch.quantile(img_z1_pixel, 0.01)

                    pseudo_p0_pixel = (pseudo_p0_pixel - nlb)/(nub - nlb)
                    pseudo_p0_pixel = pseudo_p0_pixel*(ub - lb) + lb
                    '''
                    
                    #print('img_z1 99 percentile and 1 percentile', torch.quantile(img_z1, 0.99), torch.quantile(img_z1, 0.01))
                    #print('img_p 99 percentile and 1 percentile', torch.quantile(img_p, 0.99), torch.quantile(img_p, 0.01)) 
                    
                    img_p1, pred_p0, img_p0_hat = self.p_sample_ddim(img_p, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                              quantize_denoised=quantize_denoised, temperature=temperature,
                                              noise_dropout=noise_dropout, score_corrector=score_corrector,
                                              corrector_kwargs=corrector_kwargs,
                                              unconditional_guidance_scale=unconditional_guidance_scale,
                                              unconditional_conditioning=unconditional_conditioning)
            
                    #print('img_p1 99 percentile and 1 percentile', torch.quantile(img_p1, 0.99), torch.quantile(img_p1, 0.01)) 
                    #print('img_p0_hat 99 percentile and 1 percentile', torch.quantile(img_p0_hat, 0.99), torch.quantile(img_p0_hat, 0.01))
                    #print('pred_p0 99 percentile and 1 percentile', torch.quantile(pred_p0, 0.99), torch.quantile(pred_p0, 0.01))

                    img_p2 = img_p1
                    img_p_t = img_p2.detach().clone()
                   
                    # This is as per the paper.
                    # Increased the constant to 1e6 for trial6 run.
                    if index >= 0:
                        sigma = 1e6*((1 - a_prev) / a_t) * (1 - (a_t / a_prev))  
                    else:
                        sigma = 0.5

                    
                    if i >= 0:
                        print('Optimization in pixel space')
                        print('index', index)
                        
                        # Enforcing consistency via pixel-based optimization
                        pseudo_p0 = img_p0_hat.detach() 
                        pseudo_p0_pixel = self.model.decode_first_stage(pseudo_p0)
                        
                        nub = torch.quantile(pseudo_p0_pixel, 0.99)
                        nlb = torch.quantile(pseudo_p0_pixel, 0.01)

                        pseudo_p0_pixel = (pseudo_p0_pixel - nlb)/(nub - nlb)
                        pseudo_p0_pixel = pseudo_p0_pixel*(ub - lb) + lb
                        

                        opt_var = self.pixel_optimization(measurement=measurement, 
                                                          x_prime=pseudo_p0_pixel,
                                                          operator_fn=operator_fn)
                        
                        
                        ub_new = torch.quantile(opt_var, 0.99)
                        lb_new = torch.quantile(opt_var, 0.01)
                        opt_var = (opt_var - lb)/(ub - lb)
                        opt_var = opt_var*(nub - nlb) + nlb
                        ub, lb = ub_new, lb_new
                        #print('ub, lb', ub, lb)

                        opt_var = self.model.encode_first_stage(opt_var) # Going back into latent space
                        img_p2 = self.stochastic_resample(pseudo_x0=opt_var, x_t=img_p_t, a_t=a_prev, sigma=sigma)            
                        img_p = img_p2
                        
            img_v += img_z - img_p 
            
            # Callback functions if needed
            if callback: callback(i)
            if img_callback: img_callback(pred_p0, i)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img_p)
                intermediates['pred_x0'].append(pred_z0)

            sample_img_p = self.model.decode_first_stage(img_p.detach())
            print('sample_img_p min and max', np.min(sample_img_p.detach().cpu().numpy()), np.max(sample_img_p.detach().cpu().numpy()))

        #img_p = img_p.detach().clone()
        
        #img_p is in 0-1 range. We need to convet it to the original Unconditional sampling range.
        
        '''
        img_p = img_p.detach().clone()
        img_p_pixel = self.model.decode_first_stage(img_p.detach())
        nub = torch.quantile(img_p_pixel, 0.99)
        nlb = torch.quantile(img_p_pixel, 0.01)
        img_p_pixel = (img_p_pixel - nlb)/(nub - nlb)
        img_p_pixel = img_p_pixel*(ub - lb) + lb
        '''
        
        img_p = img_p.detach().clone() 

        return img_p, intermediates


    def pixel_optimization(self, measurement, x_prime, operator_fn, eps=1e-3, max_iters=100):
        """
        Function to compute argmin_x ||y - A(x)||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            x_prime:               Estimation of \hat{x}_0 using Tweedie's formula
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        """

        loss = torch.nn.MSELoss() # MSE loss

        opt_var = x_prime.detach().clone()
        opt_var = opt_var.requires_grad_()
        optimizer = torch.optim.AdamW([opt_var], lr=1e-2) # Initializing optimizer
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons
        rho = 1e-5 # rho changed to 1e-5 from 1e-3

        angles = torch.tensor(np.linspace(0, np.pi, 256, endpoint=False))

        # Training loop

        for _ in range(max_iters):
            optimizer.zero_grad()
            
            measurement_t = measurement.permute(0, 2, 3, 1)
            sino_opt_var = ct_parallel_project_2d_batch(opt_var.permute(0,2,3,1), angles)  
            
            '''
            meas_ql = torch.quantile(measurement_t, 0.05); meas_qu = torch.quantile(measurement_t, 0.95)
            sino_opt_var = meas_ql + (((sino_opt_var - sino_opt_var.min()) / (sino_opt_var.max() - sino_opt_var.min())) * (meas_qu - meas_ql))
            '''
            #print('measurement_t, sino_opt_var, x_prime, opt_var shapes', measurement_t.shape, sino_opt_var.shape, x_prime.shape, opt_var.shape)
            

            measurement_loss = loss(measurement_t, sino_opt_var) + ((rho/2)*loss(x_prime, opt_var))
            print('measurement_loss in pixel_optimization step', measurement_loss)

            measurement_loss.backward(retain_graph=True) # Take GD step
            optimizer.step()

            # Convergence criteria
            if measurement_loss < eps**2: # needs tuning according to noise level for early stopping
                break

        return opt_var


    def latent_optimization(self, measurement, z_init, z_init_1, operator_fn, eps=1e-3, max_iters=500, lr=None):

        """
        Function to compute argmin_z ||y - A( D(z) )||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            z_init:                Starting point for optimization
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        
        Optimal parameters seem to be at around 500 steps, 200 steps for inpainting.

        """

        # Base case
        if not z_init.requires_grad:
            z_init = z_init.requires_grad_()
        
        '''
        if not z_init_1.requires_grad:
            z_init_1 = z_init_1.requires_grad()
        '''

        if lr is None:
            lr_val = 5e-3
        else:
            lr_val = lr.item()

        loss = torch.nn.MSELoss() # MSE loss
        optimizer = torch.optim.AdamW([z_init], lr=lr_val) # Initializing optimizer ###change the learning rate
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop
        init_loss = 0
        losses = []
        
        for itr in range(max_iters):
            optimizer.zero_grad()
            output = loss(measurement, operator_fn( self.model.differentiable_decode_first_stage( z_init ))) + loss(z_init, z_init_1)

            if itr == 0:
                init_loss = output.detach().clone()
                
            output.backward() # Take GD step
            optimizer.step()
            cur_loss = output.detach().cpu().numpy() 
            
            # Convergence criteria

            if itr < 200: # may need tuning for early stopping
                losses.append(cur_loss)
            else:
                losses.append(cur_loss)
                if losses[0] < cur_loss:
                    break
                else:
                    losses.pop(0)
                    
            if cur_loss < eps**2:  # needs tuning according to noise level for early stopping
                break


        return z_init, init_loss       


    def stochastic_resample(self, pseudo_x0, x_t, a_t, sigma):
        """
        Function to resample x_t based on ReSample paper.
        """
        device = self.model.betas.device
        noise = torch.randn_like(pseudo_x0, device=device)
        return (sigma * a_t.sqrt() * pseudo_x0 + (1 - a_t) * x_t)/(sigma + 1 - a_t) + noise * torch.sqrt(1/(1/sigma + 1/(1-a_t)))


    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        """
        Function for unconditional sampling using DDIM.
        """

        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates


    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device
        
        #print('c in p_sample_ddim', c)
        #blabla


        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        # Computing \hat{x}_0 via Tweedie's formula
        pseudo_x0 = (x - sqrt_one_minus_at**2 * e_t) / a_t.sqrt()
        return x_prev, pred_x0, pseudo_x0


    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)


    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec



    def ddecode(self, x_latent, cond=None, t_start=50, temp = 1, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps, temperature = temp, 
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec


               
