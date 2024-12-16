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

def second_deg_loss(inputs, reconstructions):
    #print('inputs shape', inputs.shape)
    #print('reconstructions shape', reconstructions.shape)
    
    ip_1 = kornia.filters.spatial_gradient(inputs)
    #print('ip_1 shape', ip_1.shape)

    ip_1_sz = list(ip_1.shape)
    eps_ip1 = (5e-8) * torch.rand(ip_1_sz, device=torch.device('cuda')) + (5e-8)
    ip_1 = ip_1 + eps_ip1
       
    ip_1x = ip_1[:, :, 0, :, :]; ip_1y = ip_1[:, :, 1, :, :]
 
    ip_2_1x = kornia.filters.spatial_gradient(ip_1x)     
    ip_2_1x_sz = list(ip_2_1x.shape)
    eps_ip_2_1x = (5e-8) * torch.rand(ip_2_1x_sz,device=torch.device('cuda')) + (5e-8)
    ip_2_1x = ip_2_1x + eps_ip_2_1x

    ip_2_1y = kornia.filters.spatial_gradient(ip_1y) 
    ip_2_1y_sz = list(ip_2_1y.shape)
    eps_ip_2_1y = (5e-8) * torch.rand(ip_2_1y_sz,device=torch.device('cuda')) + (5e-8)
    ip_2_1y = ip_2_1y + eps_ip_2_1y

    rec_1 = kornia.filters.spatial_gradient(reconstructions)
    
    rec_1_sz = list(rec_1.shape)
    eps_rec1 = (5e-8) * torch.rand(rec_1_sz,device=torch.device('cuda')) + (5e-8)
    rec_1 = rec_1 + eps_rec1

    rec_1x = rec_1[:, :, 0, :, :]; rec_1y = rec_1[:, :, 1, :, :]
    
    rec_2_1x = kornia.filters.spatial_gradient(rec_1x)
    rec_2_1x_sz = list(rec_2_1x.shape)
    eps_rec_2_1x = (5e-8) * torch.rand(rec_2_1x_sz,device=torch.device('cuda')) + (5e-8)
    rec_2_1x = rec_2_1x + eps_rec_2_1x

    rec_2_1y = kornia.filters.spatial_gradient(rec_1y)
    rec_2_1y_sz = list(rec_2_1y.shape)
    eps_rec_2_1y = (5e-8) * torch.rand(rec_2_1y_sz,device=torch.device('cuda')) + (5e-8)
    rec_2_1y = rec_2_1y + eps_rec_2_1y

    #print('rec_2_1x and rec_2_1y shapes', rec_2_1x.shape, rec_2_1y.shape)
    #dxchange.write_tiff(rec_2_1x.cpu().numpy(), 'hessian_images/rec_2_1x', dtype='float32', overwrite=True)
    #dxchange.write_tiff(rec_2_1y.cpu().numpy(), 'hessian_images/rec_2_1y', dtype='float32', overwrite=True)
    
    eps_diff = (5e-8) * torch.rand(rec_2_1y_sz, device=torch.device('cuda')) + (5e-8)
    
    diff_2 = torch.square(rec_2_1x - ip_2_1x) + torch.square(rec_2_1y - ip_2_1y) + eps_diff
    #dxchange.write_tiff(diff_2.cpu().numpy(), 'hessian_images/diff_2', dtype='float32', overwrite=True)
    #print('diff_2 shape', diff_2.shape)
    #loss_sec_deg = torch.sum(torch.sqrt(diff_2))
    loss_sec_deg1 = torch.mean(torch.sqrt(diff_2))
    #print('loss_sec_deg', loss_sec_deg)
    
    # Next line added on May 14, 2024 by SB
    loss_sec_deg = torch.where(torch.isnan(loss_sec_deg1), torch.zeros_like(loss_sec_deg1), loss_sec_deg1)
    eps = 1e-7

    if torch.isnan(loss_sec_deg):
        print('loss_sec_deg is nan')
        loss_sec_deg = torch.tensor([eps], device=torch.device('cuda'))
    else:
        #print('loss_sec_deg is not nan')
        loss_sec_deg = loss_sec_deg + eps

    #print('loss_sec_deg', loss_sec_deg)
    #blabla

    return loss_sec_deg

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss 

def Rec_obj(inputs, reconstructions):
    n_angs = 512; image_size = 512
    
    #dxchange.write_tiff(inputs.cpu().numpy(), 'ip_rec_images1/inp_sino', dtype='float32', overwrite=True)
    #dxchange.write_tiff(reconstructions.cpu().numpy(), 'ip_rec_images1/rec_sino', dtype='float32', overwrite=True)

    angles = torch.tensor(np.linspace(0, np.pi, n_angs, endpoint=False), dtype=torch.float32, device=torch.device('cuda'))
    #print('angles.grad_fn', angles.grad_fn)

    #print('inputs.grad_fn', inputs.grad_fn)
    #print('reconstructions.grad_fn', reconstructions.grad_fn)
    
    '''
    angles_sk = np.linspace(0, 180, 512, endpoint=False)
    
    sino_in_rec = np.zeros((2, 3, 512, 512))
    sino_out_rec = np.zeros((2, 3, 512, 512))

    for ii in range(2):
        for jj in range(3):
            sino_in = inputs[ii, jj, :, :].cpu().numpy()
            sino_in_rec[ii, jj, :, :] = skimage.transform.iradon(np.transpose(sino_in), angles_sk)
            
            sino_out = reconstructions[ii, jj, :, :].cpu().numpy()
            sino_out_rec[ii, jj, :, :] = skimage.transform.iradon(np.transpose(sino_out), angles_sk)

    #print('inputs shape', inputs.shape)
    #print('reconstructions shape', reconstructions.shape)
     
    dxchange.write_tiff(sino_in_rec, 'ip_rec_images1/inp_rec_sino', dtype='float32', overwrite=True)
    dxchange.write_tiff(sino_out_rec, 'ip_rec_images1/out_rec_sino', dtype='float32', overwrite=True)
    
    print('I am here in Rec_obj function !!!')
    '''

    #eps = 1e-7
    volume = torch_radon.volumes.Volume2D()
    volume.set_size(image_size, image_size)

    radon = torch_radon.ParallelBeam(volume=volume, angles=angles, det_spacing=1.0, det_count=image_size)
    
    '''
    inputs = radon.filter_sinogram(inputs, "butterworth")
    reconstructions = radon.filter_sinogram(reconstructions, "boom")

    print('inp_filt.grad_fn', inputs.grad_fn)
    print('rec_filt.grad_fn', reconstructions.grad_fn)
    blabla

    inp_obj = radon.backward(inp_filt)
    rec_obj = radon.backward(rec_filt)
    '''

    inp_obj = radon.backward(inputs)
    rec_obj = radon.backward(reconstructions)
    
    #print(inp_obj.grad_fn)
    #print(rec_obj.grad_fn)
    
    '''
    print('inp_obj min and max', torch.min(inp_obj), torch.max(inp_obj))
    print('rec_obj min and max', torch.min(rec_obj), torch.max(rec_obj))
    '''

    #dxchange.write_tiff(inp_obj.cpu().numpy(), 'ip_rec_images1/inp1', dtype='float32', overwrite=True)
    #dxchange.write_tiff(rec_obj.cpu().numpy(), 'ip_rec_images1/rec1', dtype='float32', overwrite=True)
    loss_factor = 0.1

    #loss_rec_obj = torch.mean(torch.square(rec_obj - inp_obj) + eps)
    #loss_rec_obj = torch.mean(torch.abs(rec_obj - inp_obj))
    loss_rec_obj = loss_factor * torch.mean(torch.square(rec_obj - inp_obj))

    #print('loss_rec_obj min and max', torch.min(loss_rec_obj), torch.max(loss_rec_obj))
    #print('loss_rec_obj req_grad', loss_rec_obj.grad_fn)

    return loss_rec_obj

class VQLPIPSWithDiscriminator(nn.Module):
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
        print(f"I am in the VQLPIPSWithDisc function. VQLPIPSWithDiscriminator running with {disc_loss} loss.")
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
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        #print('I am here in VQLPIPSWithDiscri forward now !!!')        

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
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
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
            #print('d_loss', d_loss)
            

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

class VQLPIPSWithDiscriminatorSino(nn.Module):
    # disc_factor = 0 in the next line to disable the Discriminator loss
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=0.0, disc_weight=1.0,
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
        
        '''
        inputs = inputs.cpu()
        reconstructions = reconstructions.cpu()
        print('max and min of inputs', torch.max(inputs), torch.min(inputs))
        print('max and min of recons', torch.max(reconstructions), torch.min(reconstructions))

        inp_n = torch.isnan(inputs)
        rec_n = torch.isnan(reconstructions)

        if inp_n.any():
            print('any input is nan')
            dxchange.write_tiff(inputs, 'nan_test/input88', dtype='float32', overwrite=True)
        else:
            print('any input is not nan')
            dxchange.write_tiff(inputs, 'nan_test/input88', dtype='float32', overwrite=True)
              
        if rec_n.any():
            print('any reconstructions is nan')
            dxchange.write_tiff(reconstructions, 'nan_test/recons88', dtype='float32', overwrite=True)
        else:
            print('any reconstructions is not nan')
            dxchange.write_tiff(reconstructions, 'nan_test/recons88', dtype='float32', overwrite=True)
        '''
       
        #print('inputs.grad_fn', inputs.grad_fn)
        #print('reconstructions.grad_fn', reconstructions.grad_fn)

        inp_sz = list(inputs.shape)
        eps_ip = (5e-8) * torch.rand(inp_sz, device=torch.device('cuda')) + (5e-8)
        
        recons_sz = list(reconstructions.shape)
        eps_rec = (5e-8) * torch.rand(recons_sz, device=torch.device('cuda')) + (5e-8)
    
        inputs = inputs + eps_ip
        reconstructions = reconstructions + eps_rec
        
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        #print('self.perceptual_weight', self.perceptual_weight)

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss1 = rec_loss
        
        nll_loss = torch.where(torch.isnan(nll_loss1), torch.zeros_like(nll_loss1), nll_loss1)
        nll_loss = torch.mean(nll_loss)
     
        if torch.isnan(nll_loss):
            print('nll_loss is nan')
            nll_loss = torch.tensor([1e-7], device=torch.device('cuda'))
        else:
            #print('loss_sec_deg is not nan')
            nll_loss = nll_loss + 1e-7

        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        #nll_loss = torch.mean(nll_loss)

        # Sinogram domain losses
        #print('inputs shape', inputs.shape)
        #print('reconstructions shape', reconstructions.shape)
                
        # Second order loss
        loss_sec_deg = second_deg_loss(inputs, reconstructions)

        # Object Reconstruction losses
        loss_rec_obj = Rec_obj(inputs, reconstructions)

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
            
            '''
            print('disc_factor', disc_factor)
            print('d_weight', d_weight)
            print('g_loss', g_loss)
            '''
            
            #loss = nll_loss + (d_weight * disc_factor * g_loss) + (self.codebook_weight * codebook_loss.mean())
            #loss = nll_loss + loss_sec_deg + (self.codebook_weight * codebook_loss.mean())
             
            codebook_loss1 = torch.where(torch.isnan(codebook_loss), torch.zeros_like(codebook_loss), codebook_loss)
            codebook_loss1 = torch.mean(codebook_loss1)
     
            if torch.isnan(codebook_loss1):
                print('codebook_loss is nan')
                codebook_loss1 = torch.tensor([1e-7], device=torch.device('cuda'))
            else:
                #print('loss_sec_deg is not nan')
                codebook_loss1 = codebook_loss1 + 1e-7
            
            '''
            print('nll_loss', nll_loss)
            print('loss_sec_deg', loss_sec_deg)
            print('loss_rec_obj', loss_rec_obj)
            print('codebook_loss1', codebook_loss1.mean())
            '''

            loss = nll_loss + loss_sec_deg + loss_rec_obj + (d_weight * disc_factor * g_loss) + (self.codebook_weight * codebook_loss1.mean())
            #print('self.codebook_weight', self.codebook_weight)
            #print('loss_fl', loss_fl)
            '''
            print('nll_loss', nll_loss)
            print('loss_sec_deg', loss_sec_deg)
            print('self.codebook_weight * codebook_loss1.mean()', self.codebook_weight * codebook_loss1.mean())
            print('loss', loss)
            '''
            #print('loss', loss)

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   #"{}/total_frstlst".format(split): loss_fl.clone().detach().mean(), 
                   "{}/quant_loss".format(split): codebook_loss1.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/sec_deg_loss".format(split): loss_sec_deg.detach().mean(),
                   "{}/rec_obj_loss".format(split): loss_rec_obj.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   #"{}/d_weight".format(split): d_weight.detach(),
                   #"{}/disc_factor".format(split): torch.tensor(disc_factor),
                   #"{}/g_loss".format(split): g_loss.detach().mean(),
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
            
            #print('d_loss', d_loss)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
