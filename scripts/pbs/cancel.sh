#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   ./cancel_all.sh -m sanity,seqcls -d BGL,HDFS
#   ./cancel_all.sh -m sanity -d BGL -r 1..5
#   ./cancel_all.sh -m sanity,seqcls -d BGL,HDFS -r 1,3,5 --dry-run
#
# Robust cancellation:
# - reads job IDs from qstat -wu -u "$USER"
# - for each job ID, fetches full Job_Name via qstat -f <jobid>
# - matches full name against ^<mode>_<dataset>_r<run>$
# - cancels with qdel

MODES=""
DATASETS=""
RUNS_SPEC="1..5"
DRY_RUN=0

die() { echo "Error: $*" >&2; exit 1; }

parse_runs() {
  local spec="$1"
  local -a out=()

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
  $0 -m <modes> -d <datasets> [-r <runs>] [--dry-run]

Options:
  -m, --modes     Comma-separated modes, e.g. "sanity,seqcls"
  -d, --datasets  Comma-separated datasets, e.g. "BGL,HDFS"
  -r, --runs      Runs: "1..5" (default) or "1,3,5"
      --dry-run   Print what would be cancelled, don't call qdel
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--modes) MODES="${2:-}"; shift 2 ;;
    -d|--datasets) DATASETS="${2:-}"; shift 2 ;;
    -r|--runs) RUNS_SPEC="${2:-}"; shift 2 ;;
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

# Build an ERE for the FULL job name
# Example: ^(sanity|seqcls)_(BGL|HDFS)_r(1|2|3|4|5)$
modes_re="$(printf '%s|' "${MODES_ARR[@]}" | sed 's/|$//')"
datasets_re="$(printf '%s|' "${DATASETS_ARR[@]}" | sed 's/|$//')"
runs_re="$(printf '%s|' "${RUNS_ARR[@]}" | sed 's/|$//')"
name_re="^(${modes_re})_(${datasets_re})_r(${runs_re})$"

# Get job IDs (first column) from qstat -wu -u "$USER"
qstat_out="$(qstat -wu "$USER" 2>/dev/null || true)"
[[ -n "$qstat_out" ]] || die "qstat returned no output (or failed). Are qstat/qdel available here?"

mapfile -t JOBIDS < <(
  awk '
    BEGIN { }
    /^Job ID/ { next }
    /^-+/ { next }
    /^[[:space:]]*$/ { next }
    { print $1 }
  ' <<< "$qstat_out"
)

if [[ "${#JOBIDS[@]}" -eq 0 ]]; then
  echo "No jobs found for user $USER."
  exit 0
fi

# For each job id, read full Job_Name from qstat -f
to_cancel=()
for jobid in "${JOBIDS[@]}"; do
  printf "\rChecking job ID: %s" "$jobid" >&2

  # Some PBS setups need the full server suffix; jobid from qstat already includes it.
  full="$(qstat -f "$jobid" 2>/dev/null || true)"
  [[ -n "$full" ]] || continue

  jobname="$(
    awk -F' = ' '
      $1 ~ /^[[:space:]]*Job_Name$/ { print $2; exit }
    ' <<< "$full"
  )"

  [[ -n "${jobname:-}" ]] || continue

  if [[ "$jobname" =~ $name_re ]]; then
    to_cancel+=("$jobid:$jobname")
  fi
done
printf "\n" >&2

if [[ "${#to_cancel[@]}" -eq 0 ]]; then
  echo "No matching jobs found for:"
  echo "  modes:    $MODES"
  echo "  datasets: $DATASETS"
  echo "  runs:     $RUNS_SPEC"
  exit 0
fi

# Cancel
for item in "${to_cancel[@]}"; do
  jobid="${item%%:*}"
  jobname="${item#*:}"

  if (( DRY_RUN )); then
    echo "[dry-run] qdel $jobid    # $jobname"
  else
    echo "qdel $jobid    # $jobname"
    qdel "$jobid"
  fi
done
