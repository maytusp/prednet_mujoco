#!/bin/bash --login
#SBATCH -p gpuL              # A100 GPUs
#SBATCH -G 1                 # 1 GPU
#SBATCH -t 1-0               # Wallclock limit
#SBATCH -n 1                 # One Slurm task
#SBATCH -c 12                # CPU cores available to the host code.

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(pwd)"
echo "Script directory: ${SCRIPT_DIR}"

ENVS="${ENVS:-HumanoidRun HopperHop CheetahRun}" bash scripts/run_jax_ppo_dmc.sh
