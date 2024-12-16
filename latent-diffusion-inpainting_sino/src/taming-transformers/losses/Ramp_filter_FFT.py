import numpy as np
import torch
import scipy.fft

fftmodule = scipy.fft

# Ramp filter for filtering the sinogram
size = 1024
n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int), np.arange(size / 2 - 1, 0, -2, dtype=int)))
f = np.zeros(size)
f[0] = 0.25
f[1::2] = -1 / (np.pi * n) ** 2

fourier_filter = 2 * np.real(fftmodule.fft(f))

fourier_filt_r = fourier_filter[:size//2+1]
fourier_filt_r1 = fourier_filter[:size//2]

#print('fourier_filt_r shape', fourier_filt_r.shape)
#print('fourier_filt_r', fourier_filt_r)

torch_filt = torch.from_numpy(fourier_filt_r)
torch_filt1 = torch.from_numpy(fourier_filt_r1)

#print(torch_filt.dtype)
#torch.save(torch_filt, 'Ramp_filt_FFT.pt')
torch.save(torch_filt1, 'Ramp_filt_FFT1.pt')




