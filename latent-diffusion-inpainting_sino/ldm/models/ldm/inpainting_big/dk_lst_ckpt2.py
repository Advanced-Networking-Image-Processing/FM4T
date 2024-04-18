import torch
import os

#dictionary = torch.load("last_updated.ckpt", map_location='cpu')
dictionary = torch.load("retr_auto_old_diff/updated_ldm_ep15_u.ckpt", map_location='cpu')
print('dictionary keys', dictionary.keys())

'''
del dictionary['epoch']
del dictionary['global_step']
del dictionary['pytorch-lightning_version']
del dictionary['callbacks']
del dictionary['optimizer_states']
del dictionary['lr_schedulers']
print('dictionary keys', dictionary.keys())
'''

'''
new_dict = {}
keys=dictionary['state_dict'].keys()

for i, params in enumerate(keys):
    print('i', i)
    print('params', params)
    new_dict[params] = dictionary['state_dict'][params]

    #if 'ddim_sigmas' != params and 'ddim_alphas' != params and 'ddim_alphas_prev' != params and 'ddim_sqrt_one_minus_alphas' != params:
    #    new_dict[params] = dictionary['state_dict'][params]

dictionary['state_dict'] = new_dict
'''

#torch.save(dictionary, "retr_auto_old_diff/updated_ldm_ep15_u.ckpt")
