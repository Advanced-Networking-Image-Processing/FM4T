# FM4T: Foundation Models for Tomography
This is the initial repository for our work. 

The trained model files are in this Box folder : https://anl.box.com/s/w9hdvmug71qydp3kwo14caytnyc57er2

The ReadMe files are also provided.

They are too big to be uploaded in Github, and has to be uploaded in Box.

Run the code latent-diffusion-inpainting_sino/sino_data/512x512/simu_512.py to generate the Shapes data which is used for training the SDM model.

For training just the Autoencoder part of the SDM model:
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --base ldm/models/first_stage_models/vq-f4-noattn/config.yaml --resume ldm/models/first_stage_models/vq-f4-noattn/model.ckpt --stage 0 -t --gpus 0,1,2,3,

For training the entire SDM model (Autoencoder + Diffusion process) :
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --base ldm/models/ldm/inpainting_big/config.yaml --resume logs/checkpoints/epoch_425.ckpt --stage 1 -t --gpus 0,1,2,3,

Note: The above training is done on one Node with GPUs 4, 5, 6, 7 only. Not tried with multiple nodes. Please do research on that for running it.
