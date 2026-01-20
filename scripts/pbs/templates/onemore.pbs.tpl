#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -l select=1:ncpus=${NCPUS}:ngpus=${NGPUS}:mem=${MEM}:gpu_mem=${GPU_MEM}:scratch_ssd=${SCRATCH_SSD}
#PBS -l walltime=${WALLTIME}
#PBS -j oe

function print_time() {
    duration=$SECONDS
    h=$((duration/3600))
    m=$(((duration%3600)/60))
    s=$((duration%60))
    printf '[time] %02d:%02d:%02d elapsed\n' "$h" "$m" "$s"
}

print_time
SECONDS=0

# Setup
source "/storage/brno2/home/$USER/logllm/repo/scripts/setup.sh"
print_time

export DATASET="${DATASET}"
export FT_PREFIX="${FT_PREFIX}"
export RUN_VALIDATION="1"
export MEMORY_DEBUG="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export N_EPOCHS_3="10"

# Run training
python "$SCRATCHDIR/repo/train.py" "$VARIANT"
print_time

# Copy results back to home directory
rsync -avu "$SCRATCHDIR/repo/" "$HOMEDIR/repo/"
print_time

# Run evaluation
python "$SCRATCHDIR/repo/eval.py" "$VARIANT"
print_time

# Teardown
mamba deactivate
clean_scratch
