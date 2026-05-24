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
"""Train a SAC or SAC-SF agent using JAX on the specified environment."""

import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from custombrax.training.agents.sac import checkpoint as sac_checkpoint
from custombrax.training.agents.sac import networks as sac_networks
from custombrax.training.agents.sac import train as sac
from custombrax.training.agents.sac_sf_simple import checkpoint as sac_sf_checkpoint
from custombrax.training.agents.sac_sf_simple import networks as sac_sf_networks
from custombrax.training.agents.sac_sf_simple import train as sac_sf
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
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
    "CartpoleBalance",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_ALGO = flags.DEFINE_enum("algo", "sac", ["sac", "sac_sf"], "Learner.")
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(
    "playground_config_overrides",
    None,
    "Overrides for the playground env config.",
)
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
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
    "wandb_mode",
    "online",
    ["online", "offline", "disabled"],
    "Weights & Biases logging mode.",
)
_USE_TB = flags.DEFINE_boolean("use_tb", False, "Use TensorBoard.")
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
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
    "hidden_layer_sizes",
    [256, 256],
    "SAC policy/Q hidden layer sizes",
)
_Q_NETWORK_LAYER_NORM = flags.DEFINE_boolean(
    "q_network_layer_norm", True, "Use layer norm in Q network"
)
_SF_DIM = flags.DEFINE_integer(
    "sf_dim", 0, "Successor-feature dimension for --algo=sac_sf"
)
_SF_LOSS_WEIGHT = flags.DEFINE_float(
    "sf_loss_weight", 0.0, "Weight for SAC-SF auxiliary SF loss"
)
_NORMALIZE_SF_FEATURES = flags.DEFINE_boolean(
    "normalize_sf_features", True, "L2-normalize SF basis features"
)
_SF_TASK_LR = flags.DEFINE_float(
    "sf_task_lr", 1e-5, "Learning rate for SAC-SF reward task vector"
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


def _resolve_checkpoint(path: str | None):
  if path is None:
    print("No checkpoint path provided, not restoring from checkpoint")
    return None
  ckpt_path = epath.Path(path).resolve()
  if ckpt_path.is_dir():
    latest_ckpts = [ckpt for ckpt in ckpt_path.glob("*") if ckpt.is_dir()]
    if latest_ckpts:
      latest_ckpts.sort(key=lambda x: int(x.name))
      print(f"Restoring from: {latest_ckpts[-1]}")
      return latest_ckpts[-1]
  print(f"Restoring from checkpoint: {ckpt_path}")
  return ckpt_path


def main(argv):
  del argv

  if _WARP_KERNEL_CACHE_DIR.value is not None:
    import warp as wp  # pylint: disable=g-import-not-at-top
    wp.config.kernel_cache_dir = _WARP_KERNEL_CACHE_DIR.value

  env_cfg = registry.get_default_config(_ENV_NAME.value)
  sac_params = get_rl_config(_ENV_NAME.value)

  if _NUM_TIMESTEPS.present:
    sac_params.num_timesteps = _NUM_TIMESTEPS.value
  if _NUM_EVALS.present:
    sac_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    sac_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present and _EPISODE_LENGTH.value is not None:
    sac_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    sac_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    sac_params.action_repeat = _ACTION_REPEAT.value
  if _DISCOUNTING.present:
    sac_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    sac_params.learning_rate = _LEARNING_RATE.value
  if _NUM_ENVS.present:
    sac_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    sac_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    sac_params.batch_size = _BATCH_SIZE.value
  if _GRAD_UPDATES_PER_STEP.present:
    sac_params.grad_updates_per_step = _GRAD_UPDATES_PER_STEP.value
  if _MIN_REPLAY_SIZE.present:
    sac_params.min_replay_size = _MIN_REPLAY_SIZE.value
  if _MAX_REPLAY_SIZE.present:
    sac_params.max_replay_size = _MAX_REPLAY_SIZE.value
  if _TAU.present:
    sac_params.tau = _TAU.value

  if not hasattr(sac_params, "network_factory"):
    sac_params.network_factory = config_dict.create()
  if _HIDDEN_LAYER_SIZES.present:
    sac_params.network_factory.hidden_layer_sizes = tuple(
        map(int, _HIDDEN_LAYER_SIZES.value)
    )
  if _Q_NETWORK_LAYER_NORM.present:
    sac_params.network_factory.q_network_layer_norm = (
        _Q_NETWORK_LAYER_NORM.value
    )

  env_cfg_overrides = {"impl": _IMPL.value}
  if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
    env_cfg_overrides.update(json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value))

  env = registry.load(
      _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
  )

  print(f"Environment Config:\n{env_cfg}")
  if env_cfg_overrides:
    print(f"Environment Config Overrides:\n{env_cfg_overrides}\n")
  print(f"{_ALGO.value.upper()} Training Parameters:\n{sac_params}")

  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{_ALGO.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  logdir = epath.Path(_LOGDIR.value or "logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  if _USE_WANDB.value and not _PLAY_ONLY.value:
    if wandb is None:
      raise ImportError("wandb is required for --use_wandb.")
    wandb.init(
        project=_WANDB_PROJECT.value,
        entity=_optional_string(_WANDB_ENTITY.value),
        group=_optional_string(_WANDB_GROUP.value),
        name=exp_name,
        mode=_WANDB_MODE.value,
    )
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({
        "algo": _ALGO.value,
        "env_name": _ENV_NAME.value,
        "seed": _SEED.value,
        "num_timesteps": sac_params.num_timesteps,
        "num_envs": sac_params.num_envs,
        "num_eval_envs": sac_params.get("num_eval_envs", 128),
        "sf_dim": _SF_DIM.value,
    })

  writer = None
  if _USE_TB.value and not _PLAY_ONLY.value and tensorboardX is not None:
    writer = tensorboardX.SummaryWriter(logdir)

  restore_checkpoint_path = _resolve_checkpoint(_LOAD_CHECKPOINT_PATH.value)

  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  training_params = dict(sac_params)
  training_params.pop("network_factory", None)
  num_eval_envs = training_params.pop("num_eval_envs", 128)
  training_params.pop("num_resets_per_eval", None)

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  network_module = sac_sf_networks if _ALGO.value == "sac_sf" else sac_networks
  train_module = sac_sf if _ALGO.value == "sac_sf" else sac
  checkpoint_module = (
      sac_sf_checkpoint if _ALGO.value == "sac_sf" else sac_checkpoint
  )
  network_factory = functools.partial(
      network_module.make_sac_networks, **sac_params.network_factory
  )

  extra_train_kwargs = {}
  if _ALGO.value == "sac_sf":
    if _SF_DIM.value <= 0:
      raise ValueError("--sf_dim must be positive when --algo=sac_sf")
    extra_train_kwargs.update(
        sf_dim=_SF_DIM.value,
        sf_loss_weight=_SF_LOSS_WEIGHT.value,
        normalize_sf_features=_NORMALIZE_SF_FEATURES.value,
        sf_task_lr=_SF_TASK_LR.value,
    )

  times = [time.monotonic()]

  def progress(num_steps, metrics):
    times.append(time.monotonic())
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)
    if _USE_TB.value and not _PLAY_ONLY.value and writer is not None:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if "eval/episode_reward" in metrics:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")

  if _PLAY_ONLY.value:
    if restore_checkpoint_path is None:
      raise ValueError("--play_only requires --load_checkpoint_path")
    inference_fn = checkpoint_module.load_policy(
        restore_checkpoint_path,
        deterministic=True,
    )
    print("Loaded policy checkpoint.")
  else:
    train_fn = functools.partial(
        train_module.train,
        **training_params,
        network_factory=network_factory,
        seed=_SEED.value,
        restore_checkpoint_path=restore_checkpoint_path,
        checkpoint_logdir=ckpt_path,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
        deterministic_eval=_DETERMINISTIC_EVAL.value,
        **extra_train_kwargs,
    )

    eval_env = registry.load(
        _ENV_NAME.value,
        config=registry.get_default_config(_ENV_NAME.value),
        config_overrides=env_cfg_overrides,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )

    print("Done training.")
    if len(times) > 1:
      print(f"Time to JIT compile: {times[1] - times[0]}")
      print(f"Time to train: {times[-1] - times[1]}")

    inference_fn = make_inference_fn(params, deterministic=True)

  if writer is not None:
    writer.close()

  if _NUM_VIDEOS.value <= 0:
    print("Skipping inference and video rendering because --num_videos=0.")
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.finish()
    return

  print("Starting inference...")
  jit_inference_fn = jax.jit(inference_fn)

  infer_env = registry.load(
      _ENV_NAME.value,
      config=registry.get_default_config(_ENV_NAME.value),
      config_overrides=env_cfg_overrides,
  )
  wrapped_infer_env = wrapper.wrap_for_brax_training(
      infer_env,
      episode_length=sac_params.episode_length,
      action_repeat=sac_params.get("action_repeat", 1),
  )

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(wrapped_infer_env.reset)(rng)

  empty_data = reset_states.data.__class__(
      **{k: None for k in reset_states.data.__annotations__}
  )
  empty_traj = reset_states.__class__(
      **{k: None for k in reset_states.__annotations__}
  )
  empty_traj = empty_traj.replace(data=empty_data)

  def step(carry, _):
    state, rng_key = carry
    rng_key, act_key = jax.random.split(rng_key)
    act_keys = jax.random.split(act_key, _NUM_VIDEOS.value)
    act = jax.vmap(jit_inference_fn)(state.obs, act_keys)[0]
    state = wrapped_infer_env.step(state, act)
    traj_data = empty_traj.tree_replace({
        "data.qpos": state.data.qpos,
        "data.qvel": state.data.qvel,
        "data.time": state.data.time,
        "data.ctrl": state.data.ctrl,
        "data.mocap_pos": state.data.mocap_pos,
        "data.mocap_quat": state.data.mocap_quat,
        "data.xfrc_applied": state.data.xfrc_applied,
    })
    return (state, rng_key), traj_data

  @jax.jit
  def do_rollout(state, rng_key):
    _, traj = jax.lax.scan(
        step, (state, rng_key), None, length=sac_params.episode_length
    )
    return traj

  traj_stacked = do_rollout(reset_states, jax.random.PRNGKey(_SEED.value + 1))
  traj_stacked = jax.tree.map(lambda x: jp.moveaxis(x, 0, 1), traj_stacked)
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(sac_params.episode_length)
    ]

  render_every = 2
  fps = 1.0 / infer_env.dt / render_every
  print(f"FPS for rendering: {fps}")
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  for i, rollout in enumerate(trajectories):
    traj = rollout[::render_every]
    frames = infer_env.render(
        traj, height=480, width=640, scene_option=scene_option
    )
    media.write_video(logdir / f"rollout{i}.mp4", frames, fps=fps)
    print(f"Rollout video saved as '{logdir}/rollout{i}.mp4'.")

  if _USE_WANDB.value and not _PLAY_ONLY.value:
    wandb.finish()


def run():
  app.run(main)


if __name__ == "__main__":
  run()
