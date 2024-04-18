import torch
import os

dictionary = torch.load("last_updated.ckpt", map_location='cpu')
print('dictionary keys', dictionary.keys())
blabla

#dictionary = torch.load("retr_auto_old_diff/updated_ldm_ep15.ckpt", map_location='cpu')
new_dict = {}
keys=dictionary['state_dict'].keys()

for i, params in enumerate(keys):
    print('i', i)
    print('params', params)
    #if 'ddim_sigmas' != params and 'ddim_alphas' != params and 'ddim_alphas_prev' != params and 'ddim_sqrt_one_minus_alphas' != params:
    #    new_dict[params] = dictionary['state_dict'][params]

'''
model = torch.nn.Linear(2048, 640)

for param in model.parameters():
    param.data = torch.randn(param.data.size())

s_d = model.state_dict()
keys = s_d.keys()

for i, params in enumerate(keys):
    new_dict[params] = s_d[params]
'''

#dictionary['state_dict'] = new_dict
#torch.save(dictionary, "last_updated.ckpt")
