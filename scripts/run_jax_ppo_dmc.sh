#!/bin/bash --login
#SBATCH -p gpuL              # A100 GPUs
#SBATCH -G 1                 # 1 GPU
#SBATCH -t 1-0               # Wallclock limit
#SBATCH -n 1                 # One Slurm task
#SBATCH -c 12                # CPU cores available to the host code.

set -euo pipefail

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(pwd)"
echo "Script directory: ${SCRIPT_DIR}"

CONDA_ENV="${CONDA_ENV:-mujoco_playground}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}"

ENVS="${ENVS:-HumanoidRun HopperHop CheetahRun}"
SEEDS="${SEEDS:-1}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-10000000}"
NUM_ENVS="${NUM_ENVS:-2048}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
NUM_EVALS="${NUM_EVALS:-10}"
NUM_VIDEOS="${NUM_VIDEOS:-0}"
IMPL="${IMPL:-jax}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-prednet_rl}"
WANDB_ENTITY="${WANDB_ENTITY:-maytusp}"
WANDB_GROUP="${WANDB_GROUP:-jax_ppo_dmc}"
WANDB_MODE="${WANDB_MODE:-online}"
LOGDIR="${LOGDIR:-logs/jax_ppo_dmc}"
RUN_EVALS="${RUN_EVALS:-true}"
LOG_TRAINING_METRICS="${LOG_TRAINING_METRICS:-false}"
TRAINING_METRICS_STEPS="${TRAINING_METRICS_STEPS:-1000000}"

for env_name in ${ENVS}; do
  for seed in ${SEEDS}; do
    python learning/train_jax_ppo.py \
      --env_name="${env_name}" \
      --impl="${IMPL}" \
      --seed="${seed}" \
      --num_timesteps="${NUM_TIMESTEPS}" \
      --num_envs="${NUM_ENVS}" \
      --num_eval_envs="${NUM_EVAL_ENVS}" \
      --num_evals="${NUM_EVALS}" \
      --num_videos="${NUM_VIDEOS}" \
      --use_wandb="${USE_WANDB}" \
      --wandb_project="${WANDB_PROJECT}" \
      --wandb_entity="${WANDB_ENTITY}" \
      --wandb_group="${WANDB_GROUP}" \
      --wandb_mode="${WANDB_MODE}" \
      --run_evals="${RUN_EVALS}" \
      --log_training_metrics="${LOG_TRAINING_METRICS}" \
      --training_metrics_steps="${TRAINING_METRICS_STEPS}" \
      --logdir="${LOGDIR}" \
      --suffix="jax_ppo_dmc_seed${seed}"
  done
done
