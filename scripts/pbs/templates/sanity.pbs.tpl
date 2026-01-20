#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -l select=1:ncpus=${NCPUS}:ngpus=${NGPUS}:mem=${MEM}:gpu_mem=${GPU_MEM}:scratch_ssd=${SCRATCH_SSD}
#PBS -l walltime=${WALLTIME}
#PBS -j oe

SECONDS=0

# Setup
source "/storage/brno2/home/$USER/logllm/repo/scripts/setup.sh"

export DATASET="${DATASET}"
export FT_PREFIX="${FT_PREFIX}"
export RUN_VALIDATION="1"
export MEMORY_DEBUG="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
python "$SCRATCHDIR/repo/train.py"

# Copy results back to home directory
rsync -avu "$SCRATCHDIR/repo/" "$HOMEDIR/repo/"

# Run evaluation
python "$SCRATCHDIR/repo/eval.py"

# Teardown
mamba deactivate
duration=$SECONDS
h=$((duration/3600))
m=$(((duration%3600)/60))
s=$((duration%60))
printf '%d hours, %d minutes and %d seconds elapsed.\n' "$h" "$m" "$s"
clean_scratch
