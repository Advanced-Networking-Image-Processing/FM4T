#reset all variables
GPU_IDS=(0)
SCAN_NUMS=()
j=0
i=0


# iterate over scan numbers and GPU IDs simultaneously
for j in {1..9}; do
    SCAN_NUMS=($(yes 156 | head -n $j))
    GPU_ID=${GPU_IDS[0]}

    for i in "${!SCAN_NUMS[@]}"; do
        SCAN_NUM=${SCAN_NUMS[$i]}
        NUM_SCANS=${#SCAN_NUMS[@]}
        echo "Starting reconstruction parallel=$NUM_SCANS GPU=$GPU_ID for scan=$SCAN_NUM"

        # print terminal command before executing
        echo "Executing: time CUDA_VISIBLE_DEVICES=${GPU_ID} \
          apptainer exec --nv --bind /grand/hp-ptycho:/mnt/hp-ptycho apptainer_tomocupy.sif \
            tomocupy recon  --file-name /mnt/hp-ptycho/bicer/data/tomobank/chip/tomo_00085.h5 \
            --nsino-per-chunk 8 --reconstruction-type full --out-path-name /mnt/hp-ptycho/bicer/data/tomobank/chip_rec/tomo_00085_rec_${j}_${i} &> log.r${NUM_SCANS}.${j}_${i}"

        time CUDA_VISIBLE_DEVICES=${GPU_ID} \
          apptainer exec --nv --bind /grand/hp-ptycho:/mnt/hp-ptycho apptainer_tomocupy.sif \
            tomocupy recon  --file-name /mnt/hp-ptycho/bicer/data/tomobank/chip/tomo_00085.h5 \
            --nsino-per-chunk 8 --reconstruction-type full \
            --out-path-name /mnt/hp-ptycho/bicer/data/tomobank/chip_rec/tomo_00085_rec_${j}_${i} &> log.r${NUM_SCANS}.${j}_${i} &
    done
    echo "Waiting for reconstructions to complete for iteration with ${j} scans..."
    wait 
    echo "Completed reconstructions for iteration with ${j} scans."   
done

#for i in "${!SCAN_NUMS[@]}"; do
#    SCAN_NUM=${SCAN_NUMS[$i]}
#    echo "Starting reconstruction for scan $SCAN_NUM on GPU $GPU_ID"
#    #check if recon directory exists if not create
#
#    # print terminal command before executing
#    echo "Executing: CUDA_VISIBLE_DEVICES=$GPU_ID \
#      apptainer exec --nv --bind /grand/hp-ptycho:/mnt/hp-ptycho apptainer_tomocupy.sif \
#        tomocupy recon  --file-name /mnt/hp-ptycho/bicer/data/tomobank/chip/tomo_00085.h5 \
#        --nsino-per-chunk 8 --reconstruction-type full &> log.r${NUM_SCANS}.$i"
#    
#    # Run reconstruction in the background
#    #time CUDA_VISIBLE_DEVICES=$GPU_ID  \ 
#    #  apptainer exec --nv --bind /grand/hp-ptycho:/mnt/hp-ptycho apptainer_tomocupy.sif \
#    #    tomocupy recon  --file-name /mnt/hp-ptycho/bicer/data/tomobank/chip/tomo_00085.h5 \
#    #    --nsino-per-chunk 8 --reconstruction-type full &> log.r${NUM_SCANS}.$i &
#done    
#
## Wait for all background processes to finish
#wait
#echo "All reconstructions completed."
