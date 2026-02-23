# Foundation Model for Computed Tomography for several downstream tasks

## Steps
### 1) Clone the repository and checkout the branch FM_v2

     git clone https://github.com/Advanced-Networking-Image-Processing/FM4T
     git checkout FM_v2

### 2 a)  Download the trained Latent Diffusion Models from https://anl.box.com/s/k73c1i62alqpr0yk82baq1dwx4i2w5zd
      Use LDM_not_finetuned/model.ckpt as the model weights for all sinogram to object reconstruction downstream tasks.
      Save this downloaded model in the path "models/ldm/model.ckpt"

### 2 b)  Download the trained Latent Diffusion Models from https://anl.box.com/s/k73c1i62alqpr0yk82baq1dwx4i2w5zd
      Use LDM_Sinogram_uncond/epoch=000205.ckpt as the model weights for all sinogram to object reconstruction downstream tasks.
      Save this downloaded model in the path "models/ldm/epoch=000205.ckpt"

### 3) Setup the environment, resolve dependencies, and run sample reconstruction script:
      uv run python sample_condn_recon.py

### 4) For other downstream tasks in the Sinogram domain:     
        (a) Missing Wedge: uv run sample_condition_sino_MW.py
        (b) Sparse Reconstruction: uv run sample_condition_sino_Spr_Rec.py
        (c) Denoising: uv run sample_condition_sino_denoise.py
        (d) Super Resolution: uv run sample_condition_sino_super_res_2x.py

