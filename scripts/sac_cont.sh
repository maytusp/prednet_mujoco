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

ALGO="${ALGO:-sac}"
TASK_SEQUENCE="${TASK_SEQUENCE:-CheetahRun,CheetahRunBackward,CheetahRunFast,CheetahFlip}"
SEEDS="${SEEDS:-1}"
NUM_EXPOSURES="${NUM_EXPOSURES:-1}"
PHASE_TIMESTEPS="${PHASE_TIMESTEPS:-10000000}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-10000000}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
NUM_EVALS="${NUM_EVALS:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
GRAD_UPDATES_PER_STEP="${GRAD_UPDATES_PER_STEP:-8}"
MIN_REPLAY_SIZE="${MIN_REPLAY_SIZE:-8192}"
MAX_REPLAY_SIZE="${MAX_REPLAY_SIZE:-4194304}"
IMPL="${IMPL:-jax}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-prednet_rl}"
WANDB_ENTITY="${WANDB_ENTITY:-maytusp}"
WANDB_GROUP="${WANDB_GROUP:-jax_${ALGO}_cont_dmc}"
WANDB_MODE="${WANDB_MODE:-online}"
LOGDIR="${LOGDIR:-logs/jax_${ALGO}_cont_dmc}"
SF_DIM="${SF_DIM:-16}"
NORMALIZE_SF_FEATURES="${NORMALIZE_SF_FEATURES:-true}"
SF_TASK_LR="${SF_TASK_LR:-1e-5}"
PREDNET_GAMMAS="${PREDNET_GAMMAS:-0.1,0.5,0.95}"
PREDNET_LOSS_WEIGHT="${PREDNET_LOSS_WEIGHT:-0.1}"
PREDNET_SELF_WEIGHT="${PREDNET_SELF_WEIGHT:-1.0}"
PREDNET_TOPDOWN_WEIGHT="${PREDNET_TOPDOWN_WEIGHT:-1.0}"
PREDNET_USE_TASK_VECTOR="${PREDNET_USE_TASK_VECTOR:-false}"

for seed in ${SEEDS}; do
  cmd=(
    python learning/train_jax_sac_cont.py
    --algo="${ALGO}"
    --task_sequence="${TASK_SEQUENCE}"
    --num_exposures="${NUM_EXPOSURES}"
    --phase_timesteps="${PHASE_TIMESTEPS}"
    --num_timesteps="${NUM_TIMESTEPS}"
    --impl="${IMPL}"
    --seed="${seed}"
    --num_envs="${NUM_ENVS}"
    --num_eval_envs="${NUM_EVAL_ENVS}"
    --num_evals="${NUM_EVALS}"
    --batch_size="${BATCH_SIZE}"
    --grad_updates_per_step="${GRAD_UPDATES_PER_STEP}"
    --min_replay_size="${MIN_REPLAY_SIZE}"
    --max_replay_size="${MAX_REPLAY_SIZE}"
    --use_wandb="${USE_WANDB}"
    --wandb_project="${WANDB_PROJECT}"
    --wandb_entity="${WANDB_ENTITY}"
    --wandb_group="${WANDB_GROUP}"
    --wandb_mode="${WANDB_MODE}"
    --logdir="${LOGDIR}"
    --suffix="jax_${ALGO}_cont_dmc_seed${seed}"
  )

  if [[ "${ALGO}" == "sac_sf" ]]; then
    cmd+=(
      --sf_dim="${SF_DIM}"
      --normalize_sf_features="${NORMALIZE_SF_FEATURES}"
      --sf_task_lr="${SF_TASK_LR}"
    )
  elif [[ "${ALGO}" == "sac_prednet" ]]; then
    cmd+=(
      --prednet_gammas="${PREDNET_GAMMAS}"
      --prednet_loss_weight="${PREDNET_LOSS_WEIGHT}"
      --prednet_self_weight="${PREDNET_SELF_WEIGHT}"
      --prednet_topdown_weight="${PREDNET_TOPDOWN_WEIGHT}"
      --prednet_use_task_vector="${PREDNET_USE_TASK_VECTOR}"
      --sf_dim="${SF_DIM}"
      --normalize_sf_features="${NORMALIZE_SF_FEATURES}"
      --sf_task_lr="${SF_TASK_LR}"
    )
  fi

  "${cmd[@]}"
done
