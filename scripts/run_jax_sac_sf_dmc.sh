#!/bin/bash --login
#SBATCH -p gpuL              # A100 GPUs
#SBATCH -G 1                 # 1 GPU
#SBATCH -t 1-0               # Wallclock limit
#SBATCH -n 1                 # One Slurm task
#SBATCH -c 12                # CPU cores available to the host code.

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate mjcp

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}"

ENVS="${ENVS:-CheetahRun CheetahRunBackward CheetahRunFast CheetahFlip}"
SEEDS="${SEEDS:-1}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-10000000}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
NUM_EVALS="${NUM_EVALS:-10}"
NUM_VIDEOS="${NUM_VIDEOS:-0}"
BATCH_SIZE="${BATCH_SIZE:-512}"
GRAD_UPDATES_PER_STEP="${GRAD_UPDATES_PER_STEP:-8}"
MIN_REPLAY_SIZE="${MIN_REPLAY_SIZE:-8192}"
MAX_REPLAY_SIZE="${MAX_REPLAY_SIZE:-4194304}"
IMPL="${IMPL:-jax}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-prednet_rl}"
WANDB_ENTITY="${WANDB_ENTITY:-maytusp}"
WANDB_GROUP="${WANDB_GROUP:-jax_sac_sf_dmc}"
WANDB_MODE="${WANDB_MODE:-online}"
LOGDIR="${LOGDIR:-logs/jax_sac_sf_dmc}"
SF_DIM="${SF_DIM:-16}"
SF_LOSS_WEIGHT="${SF_LOSS_WEIGHT:-1.0}"
NORMALIZE_SF_FEATURES="${NORMALIZE_SF_FEATURES:-true}"

for env_name in ${ENVS}; do
  for seed in ${SEEDS}; do
    python learning/train_jax_sac.py \
      --algo="sac_sf" \
      --env_name="${env_name}" \
      --impl="${IMPL}" \
      --seed="${seed}" \
      --num_timesteps="${NUM_TIMESTEPS}" \
      --num_envs="${NUM_ENVS}" \
      --num_eval_envs="${NUM_EVAL_ENVS}" \
      --num_evals="${NUM_EVALS}" \
      --num_videos="${NUM_VIDEOS}" \
      --batch_size="${BATCH_SIZE}" \
      --grad_updates_per_step="${GRAD_UPDATES_PER_STEP}" \
      --min_replay_size="${MIN_REPLAY_SIZE}" \
      --max_replay_size="${MAX_REPLAY_SIZE}" \
      --sf_dim="${SF_DIM}" \
      --sf_loss_weight="${SF_LOSS_WEIGHT}" \
      --normalize_sf_features="${NORMALIZE_SF_FEATURES}" \
      --use_wandb="${USE_WANDB}" \
      --wandb_project="${WANDB_PROJECT}" \
      --wandb_entity="${WANDB_ENTITY}" \
      --wandb_group="${WANDB_GROUP}" \
      --wandb_mode="${WANDB_MODE}" \
      --logdir="${LOGDIR}" \
      --suffix="jax_sac_sf_dmc_seed${seed}"
  done
done
