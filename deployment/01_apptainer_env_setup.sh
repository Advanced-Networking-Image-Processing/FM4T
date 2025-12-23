ml use /soft/modulefiles
ml spack-pe-base
ml apptainer

export BASE_SCRATCH_DIR=/local/scratch/ # For Polaris
export APPTAINER_TMPDIR=$BASE_SCRATCH_DIR/apptainer-tmpdir
mkdir -p $APPTAINER_TMPDIR

export APPTAINER_CACHEDIR=$BASE_SCRATCH_DIR/apptainer-cachedir
mkdir -p $APPTAINER_CACHEDIR

# For internet access
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

apptainer version # should return 1.4.1