#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/xhchen/code/GR00T-WholeBodyControl"
SONIC_OUTPUT_DIR="${SONIC_OUTPUT_DIR:-${REPO_DIR}/outputs/sonic_root_tracking_new_critic}"


repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
isaac_sim_root="${SONIC_ISAAC_SIM_ROOT:-/home/xhchen/code/isaacsim-5.1.0}"
python="${SONIC_PYTHON:-/home/xhchen/miniconda3/envs/sonic/bin/python}"
checkpoint="${SONIC_CHECKPOINT:-$repo_dir/sonic_v1_1/last.pt}"
motion_file="${SONIC_MOTION_FILE:-$repo_dir/data/motion_lib_bones_seed/robot_filtered}"
num_envs="${SONIC_NUM_ENVS:-4096}"
checkpoint_frequency="${SONIC_CKPT_SAVE_EVERY:-2000}"
max_checkpoints="${SONIC_MAX_CKPTS:-5}"
last_frequency="${SONIC_LAST_SAVE_EVERY:-50}"
use_wandb="${SONIC_USE_WANDB:-true}"
wandb_run_name="${SONIC_WANDB_RUN_NAME:-sonic_root_tracking_fused_new_critic}"
output_dir="${SONIC_OUTPUT_DIR:-}"
export WANDB_API_KEY="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"
for required in "$isaac_sim_root/setup_conda_env.sh" "$python" "$checkpoint" "$motion_file"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing training input: $required" >&2
        exit 1
    fi
done

set +u
source "$isaac_sim_root/setup_conda_env.sh"
set -u
cd "$repo_dir"

train_args=(
    +exp=manager/universal_token/all_modes/sonic_v1_1_root_tracking
    "checkpoint=$checkpoint"
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$motion_file"
    "num_envs=$num_envs"
    headless=true
    "use_wandb=$use_wandb"
    "wandb_run_name=$wandb_run_name"
    "callbacks.model_save.save_frequency=$checkpoint_frequency"
    "callbacks.model_save.max_saved_checkpoints=$max_checkpoints"
    "callbacks.model_save.save_last_frequency=$last_frequency"
)
if [[ -n "$output_dir" ]]; then
    train_args+=("experiment_dir=$output_dir")
fi

exec "$python" gear_sonic/train_agent_trl.py "${train_args[@]}" "$@"
