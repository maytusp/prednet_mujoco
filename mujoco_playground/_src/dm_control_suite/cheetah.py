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
"""Cheetah environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "cheetah.xml"
# Running speed above which reward is 1.
RUN_SPEED = 10
FAST_RUN_SPEED = 20
SPIN_SPEED = 5


def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      nworld=1,
      cam_res=(64, 64),
      use_textures=False,
      use_shadows=False,
      render_rgb=(True,),
      render_depth=(False,),
      enabled_geom_groups=[0, 1, 2],
      cam_active=(True, False),  # [side, back]
  )


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.01,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      vision_config=default_vision_config(),
      impl="warp",
      naconmax=100_000,
      njmax=100,
  )


class Run(mjx_env.MjxEnv):
  """Cheetah running or flipping environment."""

  def __init__(
      self,
      run_speed: float = RUN_SPEED,
      forward: bool = True,
      flip: bool = False,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    self._vision = self._config.vision

    self._run_speed = run_speed
    self._forward = 1 if forward else -1
    self._flip = flip

    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

    if self._vision:
      vision_kwargs = self._config.vision_config.to_dict()
      self._rc = mjx.create_render_context(
          mjm=self._mj_model,
          **vision_kwargs,
      )
      self._rc_pytree = self._rc.pytree()

  def _post_init(self) -> None:
    self._torso_id = self.mj_model.body("torso").id
    self._lowers = self._mj_model.jnt_range[3:, 0]
    self._uppers = self._mj_model.jnt_range[3:, 1]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng1 = jax.random.split(rng, 2)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[3:].set(
        jax.random.uniform(
            rng1,
            (self.mjx_model.nq - 3,),
            minval=self._lowers,
            maxval=self._uppers,
        )
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Stabilize.
    data = mjx_env.step(self.mjx_model, data, jp.zeros(self.mjx_model.nu), 200)
    data = data.replace(time=0.0)

    metrics = {
        "reward/run": jp.zeros(()),
        "reward/flip": jp.zeros(()),
    }
    info = {"rng": rng}

    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    if self._vision:
      frame = self._render_pixels(data)
      frame_stack = jp.repeat(frame, 3, axis=-1)
      info["frame_stack"] = frame_stack
      obs = frame_stack.reshape(-1)
    else:
      obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    reward = self._get_reward(data, action, state.info, state.metrics)  # pylint: disable=redefined-outer-name
    info = state.info
    if self._vision:
      frame = self._render_pixels(data)
      frame_stack = jp.concatenate(
          [state.info["frame_stack"][..., 1:], frame],
          axis=-1,
      )
      info = dict(state.info)
      info["frame_stack"] = frame_stack
      obs = frame_stack.reshape(-1)
    else:
      obs = self._get_obs(data, state.info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    return mjx_env.State(data, obs, reward, done, state.metrics, info)

  def _render_pixels(self, data: mjx.Data) -> jax.Array:
    render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
    out = mjx.render(self.mjx_model, render_data, self._rc_pytree)
    rgb = mjx.get_rgb(self._rc_pytree, 0, out[0])
    return jp.mean(rgb, axis=-1, keepdims=True) - 0.5

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    return jp.concatenate([
        data.qpos[1:],
        data.qvel,
    ])

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del action, info  # Unused.

    if self._flip:
      spin = self._forward * data.subtree_angmom[self._torso_id, 1]
      flip_reward = reward.tolerance(
          spin,
          bounds=(SPIN_SPEED, float("inf")),
          margin=SPIN_SPEED,
          value_at_margin=0,
          sigmoid="linear",
      )
      metrics["reward/flip"] = flip_reward
      return flip_reward

    speed = self._forward * mjx_env.get_sensor_data(
        self.mj_model, data, "torso_subtreelinvel"
    )[0]
    run_reward = reward.tolerance(
        speed,
        bounds=(self._run_speed, float("inf")),
        margin=self._run_speed,
        value_at_margin=0,
        sigmoid="linear",
    )
    metrics["reward/run"] = run_reward
    return run_reward

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
