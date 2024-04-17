import h5py
import numpy as np
import dxchange


filename='data/sino/diff_size_obj/ds-simu-test_sino_20_small.h5'
with h5py.File(filename, "r") as f:
    sino_data = f["sino"]
    print(sino_data.shape)
    dxchange.write_tiff(sino_data, 'data/sino/diff_size_obj/20_small', dtype='float32', overwrite=True)

filename='data/image/diff_size_obj/ds-simu-test_image_20_small.h5'
with h5py.File(filename, "r") as f:
    sino_data = f["image"]
    print(sino_data.shape)
    dxchange.write_tiff(sino_data, 'data/image/diff_size_obj/20_small', dtype='float32', overwrite=True)
