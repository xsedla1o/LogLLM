#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

TEMPLATE_DIR="$ROOT/templates"
OUT_SCRIPTS_DIR="$ROOT/generated"
OUT_SUBMIT_DIR="$HOME/logllm_jobs/auto"

# Selection vars (can be provided via CLI)
MODES=""
DATASETS_SEL=""
RUNS_SPEC=""
DRY_RUN=0

die() { echo "Error: $*" >&2; exit 1; }

parse_runs() {
  local spec="$1"
  local -a out=()

  # Accept: "1..5" or "1,3,5"
  if [[ "$spec" =~ ^[0-9]+\.\.[0-9]+$ ]]; then
    local a="${spec%%..*}"
    local b="${spec##*..}"
    (( a <= b )) || die "Bad run range: $spec"
    for ((i=a; i<=b; i++)); do out+=("$i"); done
  else
    IFS=',' read -r -a out <<< "$spec"
    [[ "${#out[@]}" -gt 0 ]] || die "Bad runs spec: $spec"
    for x in "${out[@]}"; do
      [[ "$x" =~ ^[0-9]+$ ]] || die "Bad run id '$x' in runs spec '$spec'"
    done
  fi

  printf '%s\n' "${out[@]}"
}

usage() {
  cat <<EOF
Usage:
  $0 [-m <modes>] [-d <datasets>] [-r <runs>] [-n|--dry-run] [-h]

Options:
  -m, --modes    Comma-separated modes/variants to generate (default: all variants from config)
  -d, --datasets Comma-separated datasets to generate (default: all datasets from config)
  -r, --runs     Runs: "1..N" (default: 1..NUM_RUNS) or "1,3,5"
  -n, --dry-run  Print what would be generated, don't create files
  -h, --help     Show this help

Examples:
  $0 -m sanity,seqcls -d BGL,HDFS -r 1..3
EOF
}

# Parse CLI args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--modes) MODES="${2:-}"; shift 2 ;;
    -d|--datasets) DATASETS_SEL="${2:-}"; shift 2 ;;
    -r|--runs) RUNS_SPEC="${2:-}"; shift 2 ;;
    -n|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1 (use --help)" ;;
  esac
done

# Decide which variants/datasets/runs to iterate
# If not provided, fall back to config values
if [[ -z "$MODES" ]]; then
  # Copy the VARIANTS array directly
  MODES_ARR=("${VARIANTS[@]}")
else
  IFS=',' read -r -a MODES_ARR <<< "$MODES"
fi

if [[ -z "$DATASETS_SEL" ]]; then
  # Copy the DATASETS array directly
  DATASETS_ARR=("${DATASETS[@]}")
else
  IFS=',' read -r -a DATASETS_ARR <<< "$DATASETS_SEL"
fi

if [[ -z "$RUNS_SPEC" ]]; then
  RUNS_SPEC="1..$NUM_RUNS"
fi

mapfile -t RUNS_ARR < <(parse_runs "$RUNS_SPEC")

# Helper to check membership in arrays
_in_array() {
  local item="$1"; shift
  local arr=("$@")
  for e in "${arr[@]}"; do
    [[ "$e" == "$item" ]] && return 0
  done
  return 1
}

# Validate requested modes and datasets against config (help catch typos)
for m in "${MODES_ARR[@]}"; do
  if ! _in_array "$m" "${VARIANTS[@]}"; then
    die "Requested mode/variant '$m' is not defined in VARIANTS (config.sh)"
  fi
done
for d in "${DATASETS_ARR[@]}"; do
  if ! _in_array "$d" "${DATASETS[@]}"; then
    die "Requested dataset '$d' is not defined in DATASETS (config.sh)"
  fi
done

# Generate scripts only for selected combinations
for variant in "${MODES_ARR[@]}"; do
  template="$TEMPLATE_DIR/${variant}.pbs.tpl"
  if [[ ! -f "$template" ]]; then
    echo "Template not found: $template" >&2
    continue
  fi

  for dataset in "${DATASETS_ARR[@]}"; do
    # Fill NCPUS, NGPUS, MEM, GPU_MEM, SCRATCH_SSD, WALLTIME
    if ! get_pbs_resources "$dataset" "$variant"; then
      # Skip combos you haven't defined yet
      continue
    fi

    for run in "${RUNS_ARR[@]}"; do
      job_name="${variant}_${dataset}_r${run}"

      # Environment for envsubst
      export JOB_NAME="$job_name"
      export VARIANT="$variant"
      export DATASET="$dataset"
      export RUN="$run"
      export FT_PREFIX="${variant}/${run}"

      if [[ -n "${MICRO_BATCH_SIZE:-}" ]]; then
        export MICRO_BATCH_SIZE
      fi
      if [[ -n "${BATCH_SIZE:-}" ]]; then
        export BATCH_SIZE
      fi

      export NCPUS NGPUS MEM GPU_MEM SCRATCH_SSD WALLTIME

      # Whitelist so vars like $SECONDS and $SCRATCHDIR don't getsubstituted
      evntsubst_vars='
        ${JOB_NAME}
        ${VARIANT}
        ${DATASET}
        ${RUN}
        ${FT_PREFIX}

        ${NCPUS}
        ${NGPUS}
        ${MEM}
        ${GPU_MEM}
        ${SCRATCH_SSD}
        ${WALLTIME}

        ${MICRO_BATCH_SIZE}
        ${BATCH_SIZE}
      '

      # Where the PBS script lives (scripts-only tree)
      script_dir="$OUT_SCRIPTS_DIR/${variant}/${dataset}"
      out_script="$script_dir/${job_name}.pbs"

      # Where you'll cd and run qsub (logs end up here)
      submit_dir="$OUT_SUBMIT_DIR/${variant}/${dataset}"

      if (( DRY_RUN )); then
        echo "[dry-run] Would generate: $out_script (submit from: $submit_dir)"
        continue
      fi

      mkdir -p "$script_dir"
      mkdir -p "$submit_dir"

      # Generate PBS script from template
      envsubst "$evntsubst_vars" <"$template" >"$out_script"
      chmod +x "$out_script"

      echo "Generated: $out_script (submit from: $submit_dir)"
    done
  done
done
