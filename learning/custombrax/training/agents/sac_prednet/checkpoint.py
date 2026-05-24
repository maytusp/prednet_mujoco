# Copyright 2026 The Brax Authors.
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

"""Checkpointing for SAC-PredNet."""

from typing import Any, Union

from custombrax.training import checkpoint
from custombrax.training import types
from custombrax.training.acme import running_statistics
from custombrax.training.agents.sac_prednet import networks as sac_networks
from etils import epath
import jax.numpy as jnp
from ml_collections import config_dict

_CONFIG_FNAME = 'sac_network_config.json'


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Any,
    config: config_dict.ConfigDict,
):
  """Saves a checkpoint."""
  return checkpoint.save(path, step, params, config, _CONFIG_FNAME)


def load(
    path: Union[str, epath.Path],
):
  """Loads SAC-PredNet checkpoint."""
  return checkpoint.load(path)


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[sac_networks.SACNetworks],
) -> config_dict.ConfigDict:
  """Returns a config dict for re-creating a network from a checkpoint."""
  return checkpoint.network_config(
      observation_size, action_size, normalize_observations, network_factory
  )


def _get_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[sac_networks.SACNetworks],
) -> sac_networks.SACNetworks:
  """Generates a SAC-PredNet network given config."""
  normalize = lambda x, y: x
  kwargs = config.network_factory_kwargs
  sf_dim = kwargs.get('sf_dim', 0)
  if config.normalize_observations:
    if sf_dim:

      def normalize_state_only(observation, normalizer_params):
        state_obs = observation[..., :-sf_dim]
        task = observation[..., -sf_dim:]
        state_obs = running_statistics.normalize(state_obs, normalizer_params)
        return jnp.concatenate([state_obs, task], axis=-1)

      normalize = normalize_state_only
    else:
      normalize = running_statistics.normalize
  return network_factory(
      config.to_dict()['observation_size'],
      config.action_size,
      preprocess_observations_fn=normalize,
      **kwargs,
  )


def load_config(
    path: Union[str, epath.Path],
) -> config_dict.ConfigDict:
  """Loads SAC-PredNet config from checkpoint."""
  path = epath.Path(path)
  config_path = path / _CONFIG_FNAME
  return checkpoint.load_config(config_path)


def load_policy(
    path: Union[str, epath.Path],
    network_factory: types.NetworkFactory[
        sac_networks.SACNetworks
    ] = sac_networks.make_sac_networks,
    deterministic: bool = True,
):
  """Loads policy inference function from SAC-PredNet checkpoint."""
  path = epath.Path(path)
  config = load_config(path)
  params = load(path)
  sac_network = _get_network(config, network_factory)
  make_inference_fn = sac_networks.make_inference_fn(sac_network)

  return make_inference_fn(params, deterministic=deterministic)
