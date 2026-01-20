#!/usr/bin/env bash
set -euo pipefail

# Datasets and variants (modes)
DATASETS=(HDFS BGL Thunderbird Liberty)
VARIANTS=(sanity seqcls onemore nobert)

# Number of repetitions per (dataset, variant)
NUM_RUNS=5

# Per-dataset + variant resource config
# Fills NCPUS, NGPUS, MEM, GPU_MEM, SCRATCH_SSD, WALLTIME
get_pbs_resources() {
  local dataset="$1"
  local variant="$2"

  # Defaults
  NCPUS=2
  NGPUS=1
  MEM="96gb"
  GPU_MEM="24gb"
  SCRATCH_SSD="200gb"
  WALLTIME="24:00:00"

  case "${dataset}:${variant}" in
    HDFS:sanity)
      ;& # Fallthrough
    HDFS:seqcls)
      WALLTIME="82:00:00"
      ;;

    BGL:sanity)
      ;& # Fallthrough
    BGL:seqcls)
      WALLTIME="24:00:00"
      ;;

    Thunderbird:sanity)
      ;& # Fallthrough
    Thunderbird:seqcls)
      WALLTIME="48:00:00"
      ;;

    Liberty:sanity)
      ;& # Fallthrough
    Liberty:seqcls)
      WALLTIME="24:00:00"
      ;;

    HDFS:nobert)
      GPU_MEM="44gb"
      WALLTIME="126:00:00"
      MICRO_BATCH_SIZE="4"
      BATCH_SIZE="32"
      ;;
    BGL:nobert)
      GPU_MEM="60gb"
      WALLTIME="14:00:00"
      MICRO_BATCH_SIZE="1"
      BATCH_SIZE="8"
      ;;
    Thunderbird:nobert)
      GPU_MEM="60gb"
      WALLTIME="40:00:00"
      MICRO_BATCH_SIZE="1"
      BATCH_SIZE="32"
      ;;
    Liberty:nobert)
      GPU_MEM="60gb"
      WALLTIME="16:00:00"
      MICRO_BATCH_SIZE="2"
      BATCH_SIZE="32"
      ;;

    HDFS:onemore)
      WALLTIME="96:00:00"
      ;;
    BGL:onemore)
      WALLTIME="28:00:00"
      ;;
    Thunderbird:onemore)
      WALLTIME="72:00:00"
      ;;
    Liberty:onemore)
      WALLTIME="36:00:00"
      ;;

    # Catch-all (optional) to avoid silent mistakes
    *)
      echo "No resource config for combination: ${dataset}:${variant}" >&2
      return 1
      ;;
  esac
}

