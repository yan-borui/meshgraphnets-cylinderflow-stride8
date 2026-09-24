#!/usr/bin/env bash
# Retrain dynamics on stored frames 0..64 using the existing four-GPU recipe.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-BEST_EFFORT}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
python_bin=${PYTHON:-python}
action=${1:-train}
case "$action" in
    train|resume) ;;
    *) printf 'Usage: bash scripts/train_prefix65_4gpu.sh train|resume\n' >&2; exit 2 ;;
esac
: "${DATA:?Set DATA to the existing 75-frame HDF5}"
: "${MANIFEST:?Set MANIFEST to its matching manifest}"
: "${PREPARED:?Set PREPARED to this dataset and method preparation directory}"
: "${RESULT_ROOT:?Set RESULT_ROOT to a new prefix65 run directory}"
: "${CUDA_VISIBLE_DEVICES:?Set exactly four allocated GPU IDs}"
for required in "$DATA" "$MANIFEST" "$PREPARED/ready.json" "$PREPARED/normalization.json" "$PREPARED/data_identity.json"; do
    if [[ ! -f "$required" ]]; then
        printf 'Required input is missing: %s\n' "$required" >&2
        exit 2
    fi
done
extra=()
if [[ "$action" == resume ]]; then
    if [[ ! -f "$RESULT_ROOT/config.json" ]]; then
        printf 'Resume requires an existing prefix65 run.\n' >&2; exit 2
    fi
    extra+=(--resume)
fi
exec "$python_bin" -m cylinderflow.run_four_gpu \
    --config "$code_root/cylinderflow_config_prefix65_4gpu.json" \
    --dataset "$DATA" --manifest "$MANIFEST" --prepared "$PREPARED" \
    --output-dir "$RESULT_ROOT" "${extra[@]}"
