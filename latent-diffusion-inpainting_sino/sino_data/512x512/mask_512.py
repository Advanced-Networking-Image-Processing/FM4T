import h5py
import numpy as np
import dxchange
import random
import time


shape_array = [100, 512, 512]
sino_mask = np.zeros(shape=shape_array, dtype=np.uint8)
print(sino_mask.shape)
st1 = time.time()

for ii in range(shape_array[0]):
    sixty_percnt = random.sample(range(512), 308)
    sino_mask[ii, sixty_percnt, :] = 255

st2 = time.time()
print('st2 - st1', st2 - st1)

print(np.min(sino_mask), np.max(sino_mask))
blabla

file_path = 'data/mask/ds-simu-tr1-sino_mask.h5'
with h5py.File(file_path, "w") as fd:
    fd.create_dataset("mask", data=sino_mask, dtype=np.uint8)

st2 = time.time()
print('st2 - st1', st2 - st1) 
