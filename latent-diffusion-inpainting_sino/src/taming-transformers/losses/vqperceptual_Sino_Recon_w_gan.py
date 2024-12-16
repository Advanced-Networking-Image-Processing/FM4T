import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import dxchange

import numpy as np
import torch_radon

import skimage

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
torch.autograd.set_detect_anomaly(True)

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss 


def Rec_obj(inputs, reconstructions, ramp_filt, volume, radon):
    
    n_angs = 512; image_size = 512; loss_factor = 1
    inputs1 = inputs.view(-1, image_size, n_angs)
    
    # Ramp filter operation in here (SB on June 4, 2024)
    pad = 512 # for 512 x 512 image size
    padded_input = torch.nn.functional.pad(inputs1, (0, pad, 0, 0))
   
    inp_fft = torch.fft.rfft(padded_input, norm = "ortho")
    filtered_inp = inp_fft * ramp_filt

    filt_sino_inp = torch.fft.irfft(filtered_inp, norm="ortho")
    filt_sino_inp = filt_sino_inp[:, :, :-pad] * (3.141592653589793 / (2 * n_angs))
    filt_sino_inpf = filt_sino_inp.view(-1, 3, image_size, image_size).to(torch.float) 
    inp_obj = radon.backward(filt_sino_inpf)
    
    loss_rec_obj = loss_factor * torch.abs(inp_obj.contiguous() - reconstructions.contiguous())
   
    del filt_sino_inpf, filt_sino_inp, filtered_inp, inp_fft, padded_input, pad, inputs1, n_angs, image_size, loss_factor
    
    return loss_rec_obj, inp_obj


class VQLPIPSWithDiscriminatorSino(nn.Module):
    # disc_factor = 0 in the next line to disable the Discriminator loss
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"I am in the VQLPIPSWithDiscSino function. VQLPIPSWithDiscriminatorSino running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        
        
        # Load the sinogram
        fp = '/homes/sruban/nicky_LDM_Torch_Radon/latent-diffusion-inpainting/src/taming-transformers/taming/modules/losses/Ramp_filt_FFT.pt'
        ramp_filt = torch.load(fp, map_location = torch.device('cuda'))
        ramp_filt = ramp_filt.view(1, 1, -1)
     
        #eps = 1e-7
        image_size = 512; n_angs = 512 
        angles = torch.tensor(np.linspace(0, np.pi, n_angs, endpoint=False), dtype=torch.float32, device=torch.device('cuda')) 
   
        volume = torch_radon.volumes.Volume2D()
        volume.set_size(image_size, image_size)
 
        radon = torch_radon.ParallelBeam(volume=volume, angles=angles, det_spacing=1.0, det_count=image_size)
        
        # Object Reconstruction losses
        rec_loss, inp_obj = Rec_obj(inputs, reconstructions, ramp_filt, volume, radon)
        
        del ramp_filt, radon, volume

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inp_obj.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])
        
        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)
        
        # now the GAN part
        if optimizer_idx == 0:
            
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            
            codebook_loss1 = torch.where(torch.isnan(codebook_loss), torch.zeros_like(codebook_loss), codebook_loss)
            codebook_loss1 = torch.mean(codebook_loss1)
     
            if torch.isnan(codebook_loss1):
                print('codebook_loss is nan')
                codebook_loss1 = torch.tensor([1e-9], device=torch.device('cuda'))
            else:
                #print('loss_sec_deg is not nan')
                codebook_loss1 = codebook_loss1 + 1e-9
            
            loss = nll_loss + (d_weight * disc_factor * g_loss) + (self.codebook_weight * codebook_loss1.mean())
            
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss1.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log
        
        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            
            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
