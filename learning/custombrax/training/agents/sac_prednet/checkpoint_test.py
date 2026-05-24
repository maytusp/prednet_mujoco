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

"""Test SAC-PredNet checkpointing."""

import functools

from absl import flags
from absl.testing import absltest
from custombrax.training.acme import running_statistics
from custombrax.training.agents.sac_prednet import checkpoint
from custombrax.training.agents.sac_prednet import networks as sac_networks
from etils import epath
import jax
from jax import numpy as jp


class CheckpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    flags.FLAGS.mark_as_parsed()

  def test_sac_params_config(self):
    network_factory = functools.partial(
        sac_networks.make_sac_networks,
        hidden_layer_sizes=(16, 21, 13),
        q_network_kernel_init_fn=jax.nn.initializers.orthogonal,
        q_network_kernel_init_kwargs={"scale": jp.sqrt(2.0)},
        prednet_gammas=(0.1, 0.5),
    )
    config = checkpoint.network_config(
        action_size=3,
        observation_size=1,
        normalize_observations=True,
        network_factory=network_factory,
    )
    self.assertEqual(
        config.network_factory_kwargs.to_dict()["hidden_layer_sizes"],
        (16, 21, 13),
    )
    self.assertEqual(
        config.network_factory_kwargs.to_dict()["prednet_gammas"], (0.1, 0.5)
    )
    self.assertEqual(config.action_size, 3)
    self.assertEqual(config.observation_size, 1)

  def test_save_and_load_checkpoint(self):
    path = self.create_tempdir("test")
    network_factory = functools.partial(
        sac_networks.make_sac_networks,
        hidden_layer_sizes=(16, 21, 13),
    )
    config = checkpoint.network_config(
        observation_size=1,
        action_size=3,
        normalize_observations=True,
        network_factory=network_factory,
    )

    normalize = running_statistics.normalize
    sac_network = network_factory(
        config.observation_size,
        config.action_size,
        preprocess_observations_fn=normalize,
        **config.network_factory_kwargs,
    )
    dummy_key = jax.random.PRNGKey(0)
    normalizer_params = running_statistics.init_state(jp.zeros((1,)))
    params = (normalizer_params, sac_network.policy_network.init(dummy_key))

    checkpoint.save(
        path.full_path,
        step=1,
        params=params,
        config=config,
    )

    policy_fn = checkpoint.load_policy(
        epath.Path(path.full_path) / "000000000001",
    )
    out = policy_fn(jp.zeros(1), jax.random.PRNGKey(0))
    self.assertEqual(out[0].shape, (3,))

  def test_save_and_load_task_vector_checkpoint(self):
    path = self.create_tempdir("test_task")
    sf_dim = 4
    raw_observation_size = 2
    network_factory = functools.partial(
        sac_networks.make_sac_networks,
        hidden_layer_sizes=(16, 16),
        prednet_gammas=(0.1, 0.5),
        sf_dim=sf_dim,
    )
    config = checkpoint.network_config(
        observation_size=raw_observation_size + sf_dim,
        action_size=3,
        normalize_observations=True,
        network_factory=network_factory,
    )

    def normalize_state_only(observation, normalizer_params):
      state_obs = observation[..., :-sf_dim]
      task = observation[..., -sf_dim:]
      state_obs = running_statistics.normalize(state_obs, normalizer_params)
      return jp.concatenate([state_obs, task], axis=-1)

    sac_network = network_factory(
        config.observation_size,
        config.action_size,
        preprocess_observations_fn=normalize_state_only,
        **config.network_factory_kwargs,
    )
    dummy_key = jax.random.PRNGKey(0)
    normalizer_params = running_statistics.init_state(
        jp.zeros((raw_observation_size,))
    )
    task_params = jp.ones((sf_dim,))
    params = (
        normalizer_params,
        sac_network.policy_network.init(dummy_key),
        task_params,
    )

    checkpoint.save(
        path.full_path,
        step=1,
        params=params,
        config=config,
    )

    policy_fn = checkpoint.load_policy(
        epath.Path(path.full_path) / "000000000001",
    )
    out = policy_fn(jp.zeros(raw_observation_size), jax.random.PRNGKey(0))
    self.assertEqual(out[0].shape, (3,))


if __name__ == "__main__":
  absltest.main()
