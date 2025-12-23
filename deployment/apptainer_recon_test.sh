apptainer exec --nv --bind /grand/hp-ptycho:/mnt/hp-ptycho apptainer_fm4t.sif \
    uv run --directory='/FM4T' sample_condn_recon.py --diffusion_config='/mnt/hp-ptycho/bicer/FM4T/models/ldm/model.ckpt' \
    --save_dir='/mnt/hp-ptycho/bicer/FM4T/results/trial_May1_25/CT_recon/sino_samples_256/admm_match_sf1_p1e_4/pg20_gamma_var/gamma1e4'
