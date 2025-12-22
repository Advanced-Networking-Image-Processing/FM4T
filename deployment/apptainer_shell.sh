# --fakeroot sets you as root user inside the container otherwise it is your regular user on host/polaris
# --bind (or -B) mounts host directories into the container: you can access hp-ptycho folder via /mnt/hp-ptycho inside container
# --nv exposes host NVIDIA GPUs and drivers to the container
apptainer shell --nv --fakeroot --bind /grand/hp-ptycho:/mnt/hp-ptycho apptainer_fm4t.sif
