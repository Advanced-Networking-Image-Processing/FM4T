# Foundation Model for Computed Tomography for several downstream tasks

## Steps
### 1) Clone the repository

     git clone https://github.com/Advanced-Networking-Image-Processing/FM4T/tree/FM_v2/FM_CT_git

     cd FM_CT_git

### 2) Download the trained Latent Diffusion Models from https://anl.box.com/s/k73c1i62alqpr0yk82baq1dwx4i2w5zd
      Use LDM_not_finetuned/model.ckpt as the model weights for all the downstream tasks.
      Save this downloaded model in the path "models/ldm/model.ckpt"

### 3) Setup the environment. Install the dependencies :

       conda env create -f environment.yaml
       conda install -c conda-forge dxchange

### 4) For the Object reconstruction from Sinogram:

        CUDA_VISIBLE_DEVICES=0 python sample_condn_recon.py

### 5) For other downstream tasks in the Sinogram domain:     
        (a) Missing Wedge: CUDA_VISIBLE_DEVICES=0 python sample_condition_sino_MW.py
        (b) Sparse Reconstruction:  CUDA_VISIBLE_DEVICES=0 python sample_condition_sino_Spr_Rec.py
        (c) Denoising:  CUDA_VISIBLE_DEVICES=0 python sample_condition_sino_denoise.py
        (d) Super Resolution: CUDA_VISIBLE_DEVICES=0 python sample_condition_sino_super_res.py

