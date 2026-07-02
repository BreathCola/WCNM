#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACTION="${1:-help}"
MODE="${2:-print}"
BOOTSTRAP="output/tier2_c03_r8_oneshot_shared_d_bootstrap_g00000_07000_v1"
BRANCH_A="output/tier2_c03_r8_oneshot_v4_rstart_g03000_to_g15000"
BRANCH_B="output/tier2_c03_r8_oneshot_v4_rstart_g07000_to_g15000"
MASK="$ROOT/data/TiHuBird/specular_masks_reviewed_v1/manifest.json"
RESOLUTION=8
EXPERIMENT="C03-r8 Tier 2 onset study"
RETRY_IDENTITY="oneshot_v4_memory_bounded_retry"
ALLOCATOR_POLICY="adaptive_pressure_cache_and_ray_retry_v1"
REFERENCE_PEAK_ALLOCATED_BYTES=0
MINIMUM_PROJECTED_HEADROOM_BYTES=268435456
PRESSURE_RELEASE_FREE_BYTES=2147483648
MEMORY_RETRY_MIN_CHUNK_SIZE=512
PYTORCH_ALLOCATOR_CONFIG="max_split_size_mb:128,garbage_collection_threshold:0.8"
PHASE_PREFIX="experiment=${EXPERIMENT};retry=${RETRY_IDENTITY};resolution=${RESOLUTION};phase="

COMMON_MODEL=(
  --source_path data/TiHuBird
  --model_type surfel
  --experiment "$EXPERIMENT"
  --roughness_min 0.03
  --normal_priors diffrender_priors_candidates/axis_smoke/C03/normal
  --normal_prior_space camera
  --resolution "$RESOLUTION"
  --optimizer_type default
  --lambda_dssim 0.2
  --lambda_norm 0.04
  --lambda_mono 0.01
  --lambda_perc 0.01
  --position_lr_init 0.00016
  --position_lr_final 0.0000016
  --position_lr_delay_mult 0.01
  --position_lr_max_steps 30000
  --feature_lr 0.0025
  --material_lr 0.0025
  --opacity_lr 0.025
  --scaling_lr 0.005
  --rotation_lr 0.001
  --percent_dense 0.01
  --densify_from_iter 500
  --densify_until_iter 15000
  --densification_interval 100
  --densify_grad_threshold 0.0002
  --opacity_reset_interval 3000
  --disable_viewer
)

COMMON_STAGE_B=(
  "${COMMON_MODEL[@]}"
  --stage stage_b
  --reflection_init_mode random_bbox
  --reflection_init_count 4096
  --reflection_init_seed 0
  --ray_background scene
  --ray_chunk_size 2048
  --ray_cutoff_sigma 3.0
  --ray_hit_threshold 0.0001
  --ray_epsilon_scale 0.0001
  --material_alpha_threshold 0.0001
  --specular_k0 0.9
  --reflection_position_lr_init 0.00016
  --reflection_position_lr_final 0.0000016
  --reflection_position_lr_delay_mult 0.01
  --reflection_position_lr_max_steps 20000
  --reflection_color_lr 0.0025
  --reflection_opacity_lr 0.025
  --reflection_scaling_lr 0.005
  --reflection_rotation_lr 0.001
  --reflection_percent_dense 0.01
  --reflection_densify_from_iter 100
  --reflection_densify_until_iter 5000
  --reflection_densification_interval 100
  --reflection_densify_grad_threshold 0.0002
  --reflection_min_opacity 0.005
  --reflection_prune_unhit_after 500
  --debug_interval 100
  --operator_gate_continuation
  --tier2_retry_identity "$RETRY_IDENTITY"
  --stage_b_allocator_policy "$ALLOCATOR_POLICY"
  --stage_b_reference_peak_allocated_bytes "$REFERENCE_PEAK_ALLOCATED_BYTES"
  --stage_b_minimum_projected_headroom_bytes "$MINIMUM_PROJECTED_HEADROOM_BYTES"
  --stage_b_pressure_release_free_bytes "$PRESSURE_RELEASE_FREE_BYTES"
  --stage_b_memory_retry_min_chunk_size "$MEMORY_RETRY_MIN_CHUNK_SIZE"
)

quote_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_or_print() {
  local log_path="$1"
  shift
  if [[ "$MODE" != "--execute" ]]; then
    echo "# experiment=$EXPERIMENT"
    echo "# retry_identity=$RETRY_IDENTITY"
    echo "# resolution=$RESOLUTION"
    echo "# allocator_policy=$ALLOCATOR_POLICY"
    echo "# PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_ALLOCATOR_CONFIG"
    quote_command "$@"
    echo "# log: $log_path"
    return
  fi
  {
    echo "# experiment=$EXPERIMENT"
    echo "# retry_identity=$RETRY_IDENTITY"
    echo "# resolution=$RESOLUTION"
    echo "# allocator_policy=$ALLOCATOR_POLICY"
    echo "# PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_ALLOCATOR_CONFIG"
    quote_command "$@"
    "$@"
  } 2>&1 | tee "$log_path"
}

require_file() {
  if [[ "$MODE" == "--execute" ]]; then
    [[ -f "$1" ]] || { echo "missing required source: $1" >&2; exit 3; }
  fi
}

require_new_dir() {
  if [[ "$MODE" == "--execute" ]]; then
    [[ ! -e "$1" ]] || { echo "refusing existing output directory: $1" >&2; exit 3; }
  fi
}

stage_b_long() {
  local branch_name="$1" branch gpu jit start_global start_local telemetry_name log_path
  local -a checkpoints
  if [[ "$branch_name" == "a" ]]; then
    branch="$BRANCH_A"; gpu=0; jit=/tmp/rtgs-bvh-tier2-oneshot-v4-gpu0
    start_global=3100; start_local=100
    telemetry_name=formal_g03101_15000.jsonl
    log_path=output/tier2_c03_r8_oneshot_v4_a_formal_g03101_15000.log
    checkpoints=(3200 3500 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000)
  elif [[ "$branch_name" == "b" ]]; then
    branch="$BRANCH_B"; gpu=1; jit=/tmp/rtgs-bvh-tier2-oneshot-v4-gpu1
    start_global=7100; start_local=100
    telemetry_name=formal_g07101_15000.jsonl
    log_path=output/tier2_c03_r8_oneshot_v4_b_formal_g07101_15000.log
    checkpoints=(7200 7500 8000 9000 10000 11000 12000 13000 14000 15000)
  else
    echo "unknown branch: $branch_name" >&2
    exit 2
  fi
  require_file "$branch/chkpnt${start_global}.pth"
  run_or_print "$log_path" \
    env CUDA_VISIBLE_DEVICES="$gpu" RTGS_BVH_JIT_ROOT="$jit" \
      PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_ALLOCATOR_CONFIG" \
      PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n RT-GS python train.py \
      "${COMMON_STAGE_B[@]}" --model_path "$branch" \
      --start_checkpoint "$branch/chkpnt${start_global}.pth" \
      --lambda_spec 0.2 --specular_masks "$MASK" \
      --iterations 15000 --test_iterations 15000 \
      --save_iterations "${checkpoints[@]}" --checkpoint_iterations "${checkpoints[@]}" \
      --operator_gate_expected_global_start "$start_global" \
      --operator_gate_expected_r_local_start "$start_local" \
      --stage_b_telemetry_jsonl "$branch/telemetry/$telemetry_name" \
      --stage_b_telemetry_max_steps "$((15000 - start_global))" \
      --stage_b_telemetry_phase_tag "${PHASE_PREFIX}${branch_name}_formal"
}

run_all() {
  if [[ "$MODE" != "--execute" ]]; then
    echo "export RTGS_TIER2_ACK_RESOLUTION8=YES"
    echo "./tools/tier2_operator.sh run-all --execute"
    return
  fi
  if [[ "${RTGS_TIER2_ACK_RESOLUTION8:-}" != "YES" ]]; then
    echo "run-all requires RTGS_TIER2_ACK_RESOLUTION8=YES" >&2
    exit 4
  fi
  exec env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n RT-GS python tools/tier2_oneshot.py run-all
}

status() {
  env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n RT-GS python tools/tier2_oneshot.py status
}

final_audit() {
  local -a command=(env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
    conda run --no-capture-output -n RT-GS python tools/tier2_oneshot.py final-audit)
  if [[ "$MODE" == "--execute" ]]; then "${command[@]}"; else quote_command "${command[@]}"; fi
}

stage_b_gate() {
  local branch="$1" gpu="$2" jit="$3" source_kind="$4" source="$5"
  local start_global="$6" start_local="$7" end_global="$8" max_steps="$9"
  local phase_tag="${10}" telemetry_name="${11}" log_path="${12}" lambda_spec="${13}"
  if [[ "$source_kind" == "diffuse" ]]; then
    require_file "$source"
    require_new_dir "$branch"
  else
    require_file "$source"
  fi
  local source_args mask_args
  if [[ "$source_kind" == "diffuse" ]]; then
    source_args=(--diffuse_init_checkpoint "$source")
  else
    source_args=(--start_checkpoint "$source")
  fi
  if [[ "$lambda_spec" == "0" ]]; then
    mask_args=(--lambda_spec 0)
  else
    mask_args=(--lambda_spec 0.2 --specular_masks "$MASK")
  fi
  run_or_print "$log_path" \
    env CUDA_VISIBLE_DEVICES="$gpu" RTGS_BVH_JIT_ROOT="$jit" \
      PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_ALLOCATOR_CONFIG" \
      PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n RT-GS python train.py \
      "${COMMON_STAGE_B[@]}" --model_path "$branch" "${source_args[@]}" "${mask_args[@]}" \
      --iterations "$end_global" --test_iterations "$end_global" \
      --save_iterations "$end_global" --checkpoint_iterations "$end_global" \
      --operator_gate_expected_global_start "$start_global" \
      --operator_gate_expected_r_local_start "$start_local" \
      --stage_b_telemetry_jsonl "$branch/telemetry/$telemetry_name" \
      --stage_b_telemetry_max_steps "$max_steps" \
      --stage_b_telemetry_phase_tag "$phase_tag"
}

case "$ACTION" in
  run-all) run_all ;;
  status) status ;;
  final-audit) final_audit ;;
  a-phase-a) stage_b_gate "$BRANCH_A" 0 /tmp/rtgs-bvh-tier2-oneshot-v4-gpu0 diffuse "$BOOTSTRAP/chkpnt3000.pth" 3000 0 3100 100 "${PHASE_PREFIX}a_warmup" warmup_g03001_03100.jsonl output/tier2_c03_r8_oneshot_v4_a_warmup_g03001_03100.log 0 ;;
  a-long) stage_b_long a ;;
  b-phase-a) stage_b_gate "$BRANCH_B" 1 /tmp/rtgs-bvh-tier2-oneshot-v4-gpu1 diffuse "$BOOTSTRAP/chkpnt7000.pth" 7000 0 7100 100 "${PHASE_PREFIX}b_warmup" warmup_g07001_07100.jsonl output/tier2_c03_r8_oneshot_v4_b_warmup_g07001_07100.log 0 ;;
  b-long) stage_b_long b ;;
  help|*)
    cat <<'EOF'
Run or inspect the C03-r8 Tier-2 one-shot study:
  run-all --execute | status | final-audit --execute
Internal coordinator actions: a-phase-a, a-long, b-phase-a, b-long.
EOF
    ;;
esac
