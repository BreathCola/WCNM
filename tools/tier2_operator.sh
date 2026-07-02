#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACTION="${1:-help}"
MODE="${2:-print}"
BOOTSTRAP="output/tier2_c03_r8_shared_d_bootstrap_g00000_07000_v1"
BRANCH_A="output/tier2_c03_r8_rstart_g03000_to_g15000_v1"
BRANCH_B="output/tier2_c03_r8_rstart_g07000_to_g15000_v1"
MASK="specular_masks_reviewed_v1/manifest.json"
RESOLUTION=8
EXPERIMENT="C03-r8 Tier 2 onset study"
PHASE_PREFIX="experiment=${EXPERIMENT};resolution=${RESOLUTION};phase="

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
  --ray_chunk_size 4096
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
)

quote_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_or_print() {
  local log_path="$1"
  shift
  echo "# experiment=$EXPERIMENT"
  echo "# resolution=$RESOLUTION"
  quote_command "$@"
  if [[ "$MODE" != "--execute" ]]; then
    echo "# log: $log_path"
    return
  fi
  "$@" 2>&1 | tee "$log_path"
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

bootstrap() {
  require_new_dir "$BOOTSTRAP"
  run_or_print output/tier2_c03_r8_shared_d_bootstrap_g00000_07000_v1.log \
    env CUDA_VISIBLE_DEVICES=1 RTGS_BVH_JIT_ROOT=/tmp/rtgs-bvh-tier2-gpu1 \
      PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n RT-GS python train.py \
      "${COMMON_MODEL[@]}" --stage stage_a --model_path "$BOOTSTRAP" \
      --iterations 7000 --test_iterations 7000 --save_iterations 7000 \
      --checkpoint_iterations 1000 2000 3000 4000 5000 6000 7000 \
      --debug_interval 1000 --require_nonzero_mono \
      --operator_gate_continuation --operator_gate_expected_global_start 0 \
      --d_bootstrap_telemetry_jsonl "$BOOTSTRAP/telemetry/bootstrap_g00001_07000.jsonl" \
      --d_bootstrap_telemetry_max_steps 7000 \
      --d_bootstrap_telemetry_phase_tag "${PHASE_PREFIX}shared_d_bootstrap"
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

audit_gate() {
  local checkpoint="$1" telemetry="$2" run_dir="$3" log="$4" g0="$5" g1="$6"
  shift 6
  local args=(
    env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
    conda run --no-capture-output -n RT-GS python tools/audit_tier2_gate.py
    --checkpoint "$checkpoint" --telemetry "$telemetry" --run-dir "$run_dir" --log "$log"
    --expected-global-start "$g0" --expected-global-end "$g1"
  )
  if [[ $# -eq 2 ]]; then
    args+=(--expected-r-local-start "$1" --expected-r-local-end "$2")
  elif [[ $# -eq 1 ]]; then
    if [[ "$1" == "running" ]]; then args+=(--allow-running-log); fi
    if [[ "$1" == "ply" ]]; then args+=(--require-ply); fi
  fi
  if [[ "$MODE" == "--execute" ]]; then
    "${args[@]}"
  else
    quote_command "${args[@]}"
  fi
}

next_gate() {
  local branch_name="$1" end="$2"
  [[ "$end" =~ ^[0-9]+$ ]] || { echo "END must be an integer" >&2; exit 2; }
  local branch gpu jit onset minimum
  if [[ "$branch_name" == "a" ]]; then
    branch="$BRANCH_A"; gpu=0; jit=/tmp/rtgs-bvh-tier2-gpu0; onset=3000; minimum=5000
  else
    branch="$BRANCH_B"; gpu=1; jit=/tmp/rtgs-bvh-tier2-gpu1; onset=7000; minimum=9000
  fi
  (( end >= minimum && end <= 15000 && end % 1000 == 0 )) || {
    echo "invalid $branch_name next endpoint: $end" >&2; exit 2;
  }
  local start=$((end - 1000))
  local start_local=$((start - onset))
  stage_b_gate "$branch" "$gpu" "$jit" start "$branch/chkpnt${start}.pth" \
    "$start" "$start_local" "$end" 1000 \
    "${PHASE_PREFIX}${branch_name}_g$((start + 1))_${end}" \
    "gate_g$((start + 1))_${end}.jsonl" \
    "output/tier2_c03_r8_${branch_name}_g$((start + 1))_${end}.log" 0.2
}

audit_next() {
  local branch_name="$1" end="$2" branch onset
  if [[ "$branch_name" == "a" ]]; then branch="$BRANCH_A"; onset=3000; else branch="$BRANCH_B"; onset=7000; fi
  local start=$((end - 1000))
  audit_gate "$branch/chkpnt${end}.pth" "$branch/telemetry/gate_g$((start + 1))_${end}.jsonl" \
    "$branch" "output/tier2_c03_r8_${branch_name}_g$((start + 1))_${end}.log" \
    "$((start + 1))" "$end" "$((start - onset + 1))" "$((end - onset))"
}

case "$ACTION" in
  bootstrap) bootstrap ;;
  protect-3000)
    require_file "$BOOTSTRAP/chkpnt3000.pth"
    if [[ "$MODE" == "--execute" ]]; then chmod a-w "$BOOTSTRAP/chkpnt3000.pth"; else quote_command chmod a-w "$BOOTSTRAP/chkpnt3000.pth"; fi ;;
  protect-bootstrap)
    if [[ "$MODE" == "--execute" ]]; then chmod a-w "$BOOTSTRAP"/chkpnt{1000,2000,3000,4000,5000,6000,7000}.pth; else echo "chmod a-w $BOOTSTRAP/chkpnt{1000,2000,3000,4000,5000,6000,7000}.pth"; fi ;;
  a-phase-a) stage_b_gate "$BRANCH_A" 0 /tmp/rtgs-bvh-tier2-gpu0 diffuse "$BOOTSTRAP/chkpnt3000.pth" 3000 0 3100 100 "${PHASE_PREFIX}a_phase_a" gate1_g03001_03100.jsonl output/tier2_c03_r8_a_gate1_g03001_03100.log 0 ;;
  a-gate2) stage_b_gate "$BRANCH_A" 0 /tmp/rtgs-bvh-tier2-gpu0 start "$BRANCH_A/chkpnt3100.pth" 3100 100 3205 105 "${PHASE_PREFIX}a_gate2" gate2_g03101_03205.jsonl output/tier2_c03_r8_a_gate2_g03101_03205.log 0.2 ;;
  a-gate3) stage_b_gate "$BRANCH_A" 0 /tmp/rtgs-bvh-tier2-gpu0 start "$BRANCH_A/chkpnt3205.pth" 3205 205 3500 295 "${PHASE_PREFIX}a_gate3" gate3_g03206_03500.jsonl output/tier2_c03_r8_a_gate3_g03206_03500.log 0.2 ;;
  a-gate4) stage_b_gate "$BRANCH_A" 0 /tmp/rtgs-bvh-tier2-gpu0 start "$BRANCH_A/chkpnt3500.pth" 3500 500 4000 500 "${PHASE_PREFIX}a_gate4" gate4_g03501_04000.jsonl output/tier2_c03_r8_a_gate4_g03501_04000.log 0.2 ;;
  b-phase-a) stage_b_gate "$BRANCH_B" 1 /tmp/rtgs-bvh-tier2-gpu1 diffuse "$BOOTSTRAP/chkpnt7000.pth" 7000 0 7100 100 "${PHASE_PREFIX}b_phase_a" gate1_g07001_07100.jsonl output/tier2_c03_r8_b_gate1_g07001_07100.log 0 ;;
  b-gate2) stage_b_gate "$BRANCH_B" 1 /tmp/rtgs-bvh-tier2-gpu1 start "$BRANCH_B/chkpnt7100.pth" 7100 100 7205 105 "${PHASE_PREFIX}b_gate2" gate2_g07101_07205.jsonl output/tier2_c03_r8_b_gate2_g07101_07205.log 0.2 ;;
  b-gate3) stage_b_gate "$BRANCH_B" 1 /tmp/rtgs-bvh-tier2-gpu1 start "$BRANCH_B/chkpnt7205.pth" 7205 205 7500 295 "${PHASE_PREFIX}b_gate3" gate3_g07206_07500.jsonl output/tier2_c03_r8_b_gate3_g07206_07500.log 0.2 ;;
  b-gate4) stage_b_gate "$BRANCH_B" 1 /tmp/rtgs-bvh-tier2-gpu1 start "$BRANCH_B/chkpnt7500.pth" 7500 500 8000 500 "${PHASE_PREFIX}b_gate4" gate4_g07501_08000.jsonl output/tier2_c03_r8_b_gate4_g07501_08000.log 0.2 ;;
  a-next) next_gate a "${3:?END required}" ;;
  b-next) next_gate b "${3:?END required}" ;;
  audit-bootstrap-3000) audit_gate "$BOOTSTRAP/chkpnt3000.pth" "$BOOTSTRAP/telemetry/bootstrap_g00001_07000.jsonl" "$BOOTSTRAP" output/tier2_c03_r8_shared_d_bootstrap_g00000_07000_v1.log 1 3000 running ;;
  audit-bootstrap-7000) audit_gate "$BOOTSTRAP/chkpnt7000.pth" "$BOOTSTRAP/telemetry/bootstrap_g00001_07000.jsonl" "$BOOTSTRAP" output/tier2_c03_r8_shared_d_bootstrap_g00000_07000_v1.log 1 7000 ply ;;
  audit-a-phase-a) audit_gate "$BRANCH_A/chkpnt3100.pth" "$BRANCH_A/telemetry/gate1_g03001_03100.jsonl" "$BRANCH_A" output/tier2_c03_r8_a_gate1_g03001_03100.log 3001 3100 1 100 ;;
  audit-a-gate2) audit_gate "$BRANCH_A/chkpnt3205.pth" "$BRANCH_A/telemetry/gate2_g03101_03205.jsonl" "$BRANCH_A" output/tier2_c03_r8_a_gate2_g03101_03205.log 3101 3205 101 205 ;;
  audit-a-gate3) audit_gate "$BRANCH_A/chkpnt3500.pth" "$BRANCH_A/telemetry/gate3_g03206_03500.jsonl" "$BRANCH_A" output/tier2_c03_r8_a_gate3_g03206_03500.log 3206 3500 206 500 ;;
  audit-a-gate4) audit_gate "$BRANCH_A/chkpnt4000.pth" "$BRANCH_A/telemetry/gate4_g03501_04000.jsonl" "$BRANCH_A" output/tier2_c03_r8_a_gate4_g03501_04000.log 3501 4000 501 1000 ;;
  audit-b-phase-a) audit_gate "$BRANCH_B/chkpnt7100.pth" "$BRANCH_B/telemetry/gate1_g07001_07100.jsonl" "$BRANCH_B" output/tier2_c03_r8_b_gate1_g07001_07100.log 7001 7100 1 100 ;;
  audit-b-gate2) audit_gate "$BRANCH_B/chkpnt7205.pth" "$BRANCH_B/telemetry/gate2_g07101_07205.jsonl" "$BRANCH_B" output/tier2_c03_r8_b_gate2_g07101_07205.log 7101 7205 101 205 ;;
  audit-b-gate3) audit_gate "$BRANCH_B/chkpnt7500.pth" "$BRANCH_B/telemetry/gate3_g07206_07500.jsonl" "$BRANCH_B" output/tier2_c03_r8_b_gate3_g07206_07500.log 7206 7500 206 500 ;;
  audit-b-gate4) audit_gate "$BRANCH_B/chkpnt8000.pth" "$BRANCH_B/telemetry/gate4_g07501_08000.jsonl" "$BRANCH_B" output/tier2_c03_r8_b_gate4_g07501_08000.log 7501 8000 501 1000 ;;
  audit-a-next) audit_next a "${3:?END required}" ;;
  audit-b-next) audit_next b "${3:?END required}" ;;
  help|*)
    cat <<'EOF'
Print a command (default) or execute one bounded gate with --execute:
  bootstrap | protect-3000 | protect-bootstrap
  a-phase-a | a-gate2 | a-gate3 | a-gate4 | a-next --execute END
  b-phase-a | b-gate2 | b-gate3 | b-gate4 | b-next --execute END
  audit-bootstrap-3000 --execute | audit-bootstrap-7000 --execute
  audit-a-phase-a --execute | audit-a-gate2 --execute | audit-a-gate3 --execute
  audit-a-gate4 --execute | audit-a-next --execute END
  audit-b-phase-a --execute | audit-b-gate2 --execute | audit-b-gate3 --execute
  audit-b-gate4 --execute | audit-b-next --execute END
EOF
    ;;
esac
