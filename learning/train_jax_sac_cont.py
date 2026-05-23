# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Continual SAC or SAC-SF training over a sequence of tasks."""

import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from custombrax.training.agents.sac import networks as sac_networks
from custombrax.training.agents.sac import train as sac
from custombrax.training.agents.sac_sf_simple import networks as sac_sf_networks
from custombrax.training.agents.sac_sf_simple import train as sac_sf
from etils import epath
from ml_collections import config_dict
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params

try:
  import tensorboardX
except ImportError:
  tensorboardX = None

try:
  import wandb
except ImportError:
  wandb = None


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("MUJOCO_GL", "egl")

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "CheetahRun",
    f"Fallback environment. One of {', '.join(registry.ALL_ENVS)}",
)
_TASK_SEQUENCE = flags.DEFINE_list(
    "task_sequence",
    None,
    "Comma-separated task/env sequence, e.g. CheetahRun,CheetahRunBackward.",
)
_NUM_EXPOSURES = flags.DEFINE_integer(
    "num_exposures", 1, "Number of passes over task_sequence."
)
_PHASE_TIMESTEPS = flags.DEFINE_list(
    "phase_timesteps",
    None,
    "Optional comma-separated timesteps per phase. Reuses the last value if"
    " there are more phases than entries.",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_ALGO = flags.DEFINE_enum("algo", "sac", ["sac", "sac_sf"], "Learner.")
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(
    "playground_config_overrides", None, "JSON env config overrides."
)
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Optional checkpoint for the first phase."
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_USE_WANDB = flags.DEFINE_boolean("use_wandb", False, "Use Weights & Biases.")
_WANDB_PROJECT = flags.DEFINE_string(
    "wandb_project", "prednet_rl", "Weights & Biases project name."
)
_WANDB_ENTITY = flags.DEFINE_string(
    "wandb_entity", "maytusp", "Weights & Biases entity name."
)
_WANDB_GROUP = flags.DEFINE_string(
    "wandb_group", None, "Weights & Biases group name."
)
_WANDB_MODE = flags.DEFINE_enum(
    "wandb_mode", "online", ["online", "offline", "disabled"], "W&B mode."
)
_USE_TB = flags.DEFINE_boolean("use_tb", False, "Use TensorBoard.")
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Timesteps per phase when explicitly set."
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Evaluations per phase")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 1.0, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", None, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_DISCOUNTING = flags.DEFINE_float("discounting", 0.99, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 128, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 512, "Batch size")
_GRAD_UPDATES_PER_STEP = flags.DEFINE_integer(
    "grad_updates_per_step", 8, "Gradient updates per env step"
)
_MIN_REPLAY_SIZE = flags.DEFINE_integer(
    "min_replay_size", 8192, "Minimum replay size before training"
)
_MAX_REPLAY_SIZE = flags.DEFINE_integer(
    "max_replay_size", 1048576 * 4, "Maximum replay size"
)
_TAU = flags.DEFINE_float("tau", 0.005, "Target critic update rate")
_DETERMINISTIC_EVAL = flags.DEFINE_boolean(
    "deterministic_eval", False, "Use deterministic eval policy"
)
_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "hidden_layer_sizes", [256, 256], "SAC policy/Q hidden layer sizes"
)
_Q_NETWORK_LAYER_NORM = flags.DEFINE_boolean(
    "q_network_layer_norm", True, "Use layer norm in Q network"
)
_SF_DIM = flags.DEFINE_integer(
    "sf_dim", 0, "Successor-feature dimension for --algo=sac_sf"
)
_SF_LOSS_WEIGHT = flags.DEFINE_float(
    "sf_loss_weight", 1.0, "Weight for SAC-SF auxiliary SF loss"
)
_NORMALIZE_SF_FEATURES = flags.DEFINE_boolean(
    "normalize_sf_features", True, "L2-normalize SF basis features"
)
_LOGDIR = flags.DEFINE_string("logdir", None, "Directory for logging.")
_WARP_KERNEL_CACHE_DIR = flags.DEFINE_string(
    "warp_kernel_cache_dir", None, "Directory for caching compiled Warp kernels."
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in mujoco_playground.dm_control_suite._envs:
    return dm_control_suite_params.brax_sac_config(env_name, _IMPL.value)

  env_config = registry.get_default_config(env_name)
  return config_dict.create(
      num_timesteps=5_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.99,
      learning_rate=1e-3,
      num_envs=128,
      num_eval_envs=128,
      batch_size=512,
      grad_updates_per_step=8,
      max_replay_size=1048576 * 4,
      min_replay_size=8192,
      tau=0.005,
      network_factory=config_dict.create(q_network_layer_norm=True),
  )


def _optional_string(value: str | None) -> str | None:
  return None if value is None or value == "" else value


def _task_sequence() -> list[str]:
  return [task for task in (_TASK_SEQUENCE.value or [_ENV_NAME.value]) if task]


def _phase_timesteps(phase_index: int) -> int | None:
  if not _PHASE_TIMESTEPS.value:
    return None
  values = [int(v) for v in _PHASE_TIMESTEPS.value]
  return values[min(phase_index, len(values) - 1)]


def _resolve_checkpoint(path: str | None):
  if path is None:
    return None
  ckpt_path = epath.Path(path).resolve()
  if ckpt_path.is_dir():
    latest_ckpts = [ckpt for ckpt in ckpt_path.glob("*") if ckpt.is_dir()]
    if latest_ckpts:
      latest_ckpts.sort(key=lambda x: int(x.name))
      return latest_ckpts[-1]
  return ckpt_path


def _apply_overrides(params: config_dict.ConfigDict, phase_index: int) -> None:
  phase_timesteps = _phase_timesteps(phase_index)
  if phase_timesteps is not None:
    params.num_timesteps = phase_timesteps
  elif _NUM_TIMESTEPS.present:
    params.num_timesteps = _NUM_TIMESTEPS.value
  if _NUM_EVALS.present:
    params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present and _EPISODE_LENGTH.value is not None:
    params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    params.action_repeat = _ACTION_REPEAT.value
  if _DISCOUNTING.present:
    params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    params.learning_rate = _LEARNING_RATE.value
  if _NUM_ENVS.present:
    params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    params.batch_size = _BATCH_SIZE.value
  if _GRAD_UPDATES_PER_STEP.present:
    params.grad_updates_per_step = _GRAD_UPDATES_PER_STEP.value
  if _MIN_REPLAY_SIZE.present:
    params.min_replay_size = _MIN_REPLAY_SIZE.value
  if _MAX_REPLAY_SIZE.present:
    params.max_replay_size = _MAX_REPLAY_SIZE.value
  if _TAU.present:
    params.tau = _TAU.value

  if not hasattr(params, "network_factory"):
    params.network_factory = config_dict.create()
  if _HIDDEN_LAYER_SIZES.present:
    params.network_factory.hidden_layer_sizes = tuple(
        map(int, _HIDDEN_LAYER_SIZES.value)
    )
  if _Q_NETWORK_LAYER_NORM.present:
    params.network_factory.q_network_layer_norm = (
        _Q_NETWORK_LAYER_NORM.value
    )


def main(argv):
  del argv
  if _WARP_KERNEL_CACHE_DIR.value is not None:
    import warp as wp  # pylint: disable=g-import-not-at-top
    wp.config.kernel_cache_dir = _WARP_KERNEL_CACHE_DIR.value

  tasks = _task_sequence()
  if not tasks:
    raise ValueError("No tasks provided.")
  if _ALGO.value == "sac_sf" and _SF_DIM.value <= 0:
    raise ValueError("--sf_dim must be positive when --algo=sac_sf")

  env_cfg_overrides = {"impl": _IMPL.value}
  if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
    env_cfg_overrides.update(json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value))

  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  exp_name = f"{tasks[0]}-{_ALGO.value}-cont-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  logdir = epath.Path(_LOGDIR.value or "logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Continual task sequence: {tasks} x {_NUM_EXPOSURES.value}")
  print(f"Logs are being stored in: {logdir}")

  if _USE_WANDB.value:
    if wandb is None:
      raise ImportError("wandb is required for --use_wandb.")
    wandb.init(
        project=_WANDB_PROJECT.value,
        entity=_optional_string(_WANDB_ENTITY.value),
        group=_optional_string(_WANDB_GROUP.value),
        name=exp_name,
        mode=_WANDB_MODE.value,
        config={
            "algo": f"{_ALGO.value}_cont",
            "task_sequence": tasks,
            "num_exposures": _NUM_EXPOSURES.value,
            "sf_dim": _SF_DIM.value,
        },
    )

  writer = None
  if _USE_TB.value and tensorboardX is not None:
    writer = tensorboardX.SummaryWriter(logdir)

  restore_checkpoint_path = _resolve_checkpoint(_LOAD_CHECKPOINT_PATH.value)
  restore_params = None
  cumulative_steps = 0
  phase_index = 0
  times = [time.monotonic()]

  network_module = sac_sf_networks if _ALGO.value == "sac_sf" else sac_networks
  train_module = sac_sf if _ALGO.value == "sac_sf" else sac

  for exposure_id in range(_NUM_EXPOSURES.value):
    for task_id, env_name in enumerate(tasks):
      phase_dir = logdir / f"phase_{phase_index:03d}_{env_name}"
      ckpt_path = phase_dir / "checkpoints"
      ckpt_path.mkdir(parents=True, exist_ok=True)

      env_cfg = registry.get_default_config(env_name)
      sac_params = get_rl_config(env_name)
      _apply_overrides(sac_params, phase_index)

      with open(phase_dir / "config.json", "w", encoding="utf-8") as fp:
        json.dump(env_cfg.to_dict(), fp, indent=4)

      env = registry.load(
          env_name, config=env_cfg, config_overrides=env_cfg_overrides
      )
      eval_env = registry.load(
          env_name,
          config=registry.get_default_config(env_name),
          config_overrides=env_cfg_overrides,
      )

      training_params = dict(sac_params)
      training_params.pop("network_factory", None)
      num_eval_envs = training_params.pop("num_eval_envs", 128)
      training_params.pop("num_resets_per_eval", None)

      if _DOMAIN_RANDOMIZATION.value:
        training_params["randomization_fn"] = registry.get_domain_randomizer(
            env_name
        )

      network_factory = functools.partial(
          network_module.make_sac_networks, **sac_params.network_factory
      )

      extra_train_kwargs = {}
      if _ALGO.value == "sac_sf":
        extra_train_kwargs.update(
            sf_dim=_SF_DIM.value,
            sf_loss_weight=_SF_LOSS_WEIGHT.value,
            normalize_sf_features=_NORMALIZE_SF_FEATURES.value,
        )

      def progress(num_steps, metrics, env_name=env_name):
        global_step = cumulative_steps + num_steps
        metrics = {
            **metrics,
            "phase/task_id": task_id,
            "phase/exposure_id": exposure_id,
            "phase/index": phase_index,
        }
        if _USE_WANDB.value:
          wandb.log(metrics, step=global_step)
        if writer is not None:
          for key, value in metrics.items():
            writer.add_scalar(key, value, global_step)
          writer.flush()
        if "eval/episode_reward" in metrics:
          print(
              f"{global_step}: {env_name} reward="
              f"{metrics['eval/episode_reward']:.3f}"
          )

      print(
          f"\n=== Phase {phase_index}: exposure={exposure_id} "
          f"task={task_id} env={env_name} ==="
      )
      make_policy, restore_params, _ = train_module.train(
          environment=env,
          eval_env=eval_env,
          progress_fn=progress,
          network_factory=network_factory,
          seed=_SEED.value + phase_index,
          restore_checkpoint_path=(
              restore_checkpoint_path if restore_params is None else None
          ),
          restore_params=restore_params,
          return_q_params=True,
          checkpoint_logdir=ckpt_path,
          wrap_env_fn=wrapper.wrap_for_brax_training,
          num_eval_envs=num_eval_envs,
          deterministic_eval=_DETERMINISTIC_EVAL.value,
          **training_params,
          **extra_train_kwargs,
      )
      del make_policy
      restore_checkpoint_path = None
      cumulative_steps += int(sac_params.num_timesteps)
      phase_index += 1

  print("Done continual SAC training.")
  print(f"Total walltime: {time.monotonic() - times[0]}")
  if writer is not None:
    writer.close()
  if _USE_WANDB.value:
    wandb.finish()


def run():
  app.run(main)


if __name__ == "__main__":
  run()
