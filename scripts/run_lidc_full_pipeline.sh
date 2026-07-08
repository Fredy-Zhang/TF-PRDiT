#!/usr/bin/env bash
# Run the full LIDC TF-PRDiT workflow:
#   1) train stage 1 local denoiser
#   2) write the stage 1 checkpoint into a generated stage 2 config
#   3) train stage 2 global residual path
#   4) run unconditional sampling
#   5) run X-ray conditional sampling
#
# Common overrides:
#   NPROC=8 scripts/run_lidc_full_pipeline.sh
#   STAGE1_CKPT=/path/to/stage1.pt SKIP_STAGE1=1 scripts/run_lidc_full_pipeline.sh
#   STAGE2_CKPT=/path/to/stage2.pt SKIP_STAGE1=1 SKIP_STAGE2=1 scripts/run_lidc_full_pipeline.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_DIR="${CONFIG_DIR:-configs}"
RESULTS_DIR="${RESULTS_DIR:-results}"

STAGE1_CONFIG="${STAGE1_CONFIG:-lidc_stage1_local.yaml}"
STAGE2_TEMPLATE_CONFIG="${STAGE2_TEMPLATE_CONFIG:-lidc_stage2_global.yaml}"
STAGE2_GENERATED_CONFIG="${STAGE2_GENERATED_CONFIG:-lidc_stage2_global.auto.yaml}"

NPROC="${NPROC:-4}"
NNODES="${NNODES:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
TORCHRUN="${TORCHRUN:-torchrun}"

NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-1000}"
UNCOND_BATCH_SIZE="${UNCOND_BATCH_SIZE:-4}"
UNCOND_TOTAL_SAMPLES="${UNCOND_TOTAL_SAMPLES:-20}"
COND_NUM_SAMPLES="${COND_NUM_SAMPLES:-100}"
COND_NUM_SAVE_SAMPLES="${COND_NUM_SAVE_SAMPLES:-10}"
ROTATIONS="${ROTATIONS:-2}"

UNCOND_OUTPUT_DIR="${UNCOND_OUTPUT_DIR:-outputs/lidc_unconditional}"
COND_OUTPUT_DIR="${COND_OUTPUT_DIR:-outputs/lidc_conditional}"

SKIP_STAGE1="${SKIP_STAGE1:-0}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
SKIP_UNCOND="${SKIP_UNCOND:-0}"
SKIP_COND="${SKIP_COND:-0}"
COND_NO_SAVE_INTERMEDIATE="${COND_NO_SAVE_INTERMEDIATE:-0}"

STAGE1_CKPT="${STAGE1_CKPT:-}"
STAGE2_CKPT="${STAGE2_CKPT:-}"

log() {
  printf "\n[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

latest_checkpoint_for_model() {
  local model_name="$1"
  python - "$RESULTS_DIR" "$model_name" <<'PY'
import os
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
model_name = sys.argv[2].replace("/", "-")

checkpoints = []
if results_dir.exists():
    for exp_dir in results_dir.glob(f"*-{model_name}"):
        ckpt_dir = exp_dir / "checkpoints"
        checkpoints.extend(ckpt_dir.glob("*.pt"))

if not checkpoints:
    raise SystemExit(f"No checkpoints found for model {model_name} under {results_dir}")

latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
print(latest)
PY
}

config_value() {
  local config_name="$1"
  local dotted_key="$2"
  python - "$CONFIG_DIR/$config_name" "$dotted_key" <<'PY'
import sys
import yaml

path, dotted_key = sys.argv[1], sys.argv[2]
with open(path, "r") as f:
    data = yaml.safe_load(f)

value = data
for key in dotted_key.split("."):
    value = value[key]
print(value)
PY
}

write_stage2_config() {
  local template_name="$1"
  local output_name="$2"
  local pretrained_path="$3"

  python - "$CONFIG_DIR/$template_name" "$CONFIG_DIR/$output_name" "$pretrained_path" <<'PY'
import sys
import yaml

template_path, output_path, pretrained_path = sys.argv[1:4]
with open(template_path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["pretrained_path"] = pretrained_path

with open(output_path, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

print(output_path)
PY
}

run_train() {
  local config_name="$1"
  shift
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" "${TORCHRUN}" \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC}" \
    train.py \
    --config "${config_name}" \
    "$@"
}

require_file "${CONFIG_DIR}/${STAGE1_CONFIG}"
require_file "${CONFIG_DIR}/${STAGE2_TEMPLATE_CONFIG}"

STAGE1_MODEL="$(config_value "${STAGE1_CONFIG}" "model.name")"
STAGE2_MODEL="$(config_value "${STAGE2_TEMPLATE_CONFIG}" "model.name")"

log "Stage 1 config: ${STAGE1_CONFIG} (${STAGE1_MODEL})"
log "Stage 2 template config: ${STAGE2_TEMPLATE_CONFIG} (${STAGE2_MODEL})"

if [[ "${SKIP_STAGE1}" == "1" ]]; then
  if [[ -z "${STAGE1_CKPT}" ]]; then
    STAGE1_CKPT="$(latest_checkpoint_for_model "${STAGE1_MODEL}")"
  fi
  log "Skipping stage 1; using stage 1 checkpoint: ${STAGE1_CKPT}"
else
  log "Training LIDC stage 1 local denoiser"
  run_train "${STAGE1_CONFIG}" --from_scratch
  STAGE1_CKPT="$(latest_checkpoint_for_model "${STAGE1_MODEL}")"
  log "Detected latest stage 1 checkpoint: ${STAGE1_CKPT}"
fi

require_file "${STAGE1_CKPT}"

log "Writing generated stage 2 config with model.pretrained_path=${STAGE1_CKPT}"
write_stage2_config "${STAGE2_TEMPLATE_CONFIG}" "${STAGE2_GENERATED_CONFIG}" "${STAGE1_CKPT}"

if [[ "${SKIP_STAGE2}" == "1" ]]; then
  if [[ -z "${STAGE2_CKPT}" ]]; then
    STAGE2_CKPT="$(latest_checkpoint_for_model "${STAGE2_MODEL}")"
  fi
  log "Skipping stage 2; using stage 2 checkpoint: ${STAGE2_CKPT}"
else
  log "Training LIDC stage 2 global residual path"
  run_train "${STAGE2_GENERATED_CONFIG}"
  STAGE2_CKPT="$(latest_checkpoint_for_model "${STAGE2_MODEL}")"
  log "Detected latest stage 2 checkpoint: ${STAGE2_CKPT}"
fi

require_file "${STAGE2_CKPT}"

if [[ "${SKIP_UNCOND}" == "1" ]]; then
  log "Skipping unconditional sampling"
else
  log "Running unconditional CT sampling"
  python sample.py \
    --config "${STAGE2_GENERATED_CONFIG}" \
    --ckpt "${STAGE2_CKPT}" \
    --num-sampling-steps "${NUM_SAMPLING_STEPS}" \
    --num-samples "${UNCOND_BATCH_SIZE}" \
    --total-samples "${UNCOND_TOTAL_SAMPLES}" \
    --output-dir "${UNCOND_OUTPUT_DIR}" \
    --new
fi

if [[ "${SKIP_COND}" == "1" ]]; then
  log "Skipping conditional sampling"
else
  log "Running X-ray conditional sampling"
  COND_ARGS=()
  if [[ "${COND_NO_SAVE_INTERMEDIATE}" == "1" ]]; then
    COND_ARGS+=(--no-save-intermediate)
  fi

  python sample_xrays.py \
    --config "${STAGE2_GENERATED_CONFIG}" \
    --ckpt "${STAGE2_CKPT}" \
    --num-sampling-steps "${NUM_SAMPLING_STEPS}" \
    --num-samples "${COND_NUM_SAMPLES}" \
    --num-save-samples "${COND_NUM_SAVE_SAMPLES}" \
    --rotations "${ROTATIONS}" \
    --output-dir "${COND_OUTPUT_DIR}" \
    --new \
    "${COND_ARGS[@]}"
fi

log "Pipeline complete"
log "Stage 1 checkpoint: ${STAGE1_CKPT}"
log "Stage 2 checkpoint: ${STAGE2_CKPT}"
log "Generated stage 2 config: ${CONFIG_DIR}/${STAGE2_GENERATED_CONFIG}"
log "Unconditional output: ${UNCOND_OUTPUT_DIR}"
log "Conditional output: ${COND_OUTPUT_DIR}"
