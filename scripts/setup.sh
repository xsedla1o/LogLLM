#!/bin/bash
# Source this script to set up the environment for the job
HOMEDIR="/storage/brno2/home/$USER/logllm"
export TMPDIR=$SCRATCH
export BASE=$SCRATCH

# if scratch directory is not set, issue error message and exit
test -n "$SCRATCH" || { echo >&2 "Variable SCRATCH is not set!"; exit 1; }

cp -a $HOMEDIR/* "$SCRATCH"
cd "$SCRATCH" || { echo >&2 "Cannot change to working directory!"; exit 1; }

module add python/3.9.12-gcc-10.2.1-rg2lpmk
module add cuda/12.6.1-gcc-10.2.1-hplxoqp
module add mambaforge

mkdir "$SCRATCH/llm-env"
tar -xzf "$SCRATCH/logllm-env.tar.gz" -C "$SCRATCH/llm-env"
"$SCRATCH/llm-env/bin/conda-unpack"
conda config --set env_prompt '({name})'
mamba activate "$SCRATCH/llm-env"
pip install flash-attn --no-build-isolation

rm "$SCRATCH/logllm-env.tar.gz"

nvidia-smi
