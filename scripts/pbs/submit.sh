#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   ./submit_all.sh -m sanity,seqcls -d BGL,HDFS
#   ./submit_all.sh -m sanity -d BGL -r 1..5
#   ./submit_all.sh -m sanity,seqcls -d BGL,HDFS -r 1,3,5 --dry-run
#
# Assumes generated scripts are in:   pbs/generated/<mode>/<dataset>/<mode>_<dataset>_r<run>.pbs
# Assumes submission dirs are in:     $HOME/logllm_jobs/auto/<mode>/<dataset>

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$ROOT/generated"
SUBMIT_ROOT="${HOME}/logllm_jobs/auto"

MODES=""
DATASETS=""
RUNS_SPEC="1..5"
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
  $0 -m <modes> -d <datasets> [-r <runs>] [--submit-root DIR] [--scripts-root DIR] [--dry-run]

Options:
  -m, --modes        Comma-separated modes, e.g. "sanity,seqcls"
  -d, --datasets     Comma-separated datasets, e.g. "BGL,HDFS"
  -r, --runs         Runs: "1..5" (default) or "1,3,5"
      --submit-root  Submission tree root (default: $SUBMIT_ROOT)
      --scripts-root Generated scripts root (default: $SCRIPTS_ROOT)
      --dry-run      Print what would be submitted, don't call qsub
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--modes) MODES="${2:-}"; shift 2 ;;
    -d|--datasets) DATASETS="${2:-}"; shift 2 ;;
    -r|--runs) RUNS_SPEC="${2:-}"; shift 2 ;;
    --submit-root) SUBMIT_ROOT="${2:-}"; shift 2 ;;
    --scripts-root) SCRIPTS_ROOT="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1 (use --help)" ;;
  esac
done

[[ -n "$MODES" ]] || die "Missing -m/--modes"
[[ -n "$DATASETS" ]] || die "Missing -d/--datasets"

IFS=',' read -r -a MODES_ARR <<< "$MODES"
IFS=',' read -r -a DATASETS_ARR <<< "$DATASETS"

mapfile -t RUNS_ARR < <(parse_runs "$RUNS_SPEC")

# Submit all combinations
for mode in "${MODES_ARR[@]}"; do
  [[ -n "$mode" ]] || continue
  for dataset in "${DATASETS_ARR[@]}"; do
    [[ -n "$dataset" ]] || continue

    submit_dir="$SUBMIT_ROOT/$mode/$dataset"
    scripts_dir="$SCRIPTS_ROOT/$mode/$dataset"

    [[ -d "$submit_dir" ]] || die "Submit dir not found: $submit_dir (did you run generate_jobs.sh?)"
    [[ -d "$scripts_dir" ]] || die "Scripts dir not found: $scripts_dir (did you run generate_jobs.sh?)"

    for run in "${RUNS_ARR[@]}"; do
      job="${mode}_${dataset}_r${run}.pbs"
      job_path="$scripts_dir/$job"

      [[ -f "$job_path" ]] || die "PBS script not found: $job_path"

      if (( DRY_RUN )); then
        echo "[dry-run] (cd '$submit_dir' && qsub '$job_path')"
      else
        ( cd "$submit_dir" && qsub "$job_path" )
      fi
    done
  done
done
