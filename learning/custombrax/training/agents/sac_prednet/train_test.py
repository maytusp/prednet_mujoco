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

"""SAC-PredNet tests."""

import pickle

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from custombrax.training import types
from custombrax.training.acme import running_statistics
from custombrax.training.agents.sac_prednet import losses as sac_losses
from custombrax.training.agents.sac_prednet import networks as sac_networks
from custombrax.training.agents.sac_prednet import train as sac
import jax
import jax.numpy as jnp


class SACTest(parameterized.TestCase):
  """Tests for SAC-PredNet module."""

  def test_prednet_network_aux_shapes(self):
    sac_network = sac_networks.make_sac_networks(
        observation_size=7,
        action_size=3,
        hidden_layer_sizes=(8, 6),
        prednet_gammas=(0.1, 0.5),
        sf_dim=4,
    )
    q_params = sac_network.q_network.init(jax.random.PRNGKey(0))
    obs = jnp.zeros((5, 7))
    action = jnp.zeros((5, 3))
    q, aux = sac_network.q_network.apply(
        None, q_params, obs, action, return_aux=True
    )

    self.assertEqual(q.shape, (5, 2))
    self.assertLen(aux['features'], 2)
    self.assertLen(aux['features'][0], 2)
    self.assertEqual(aux['features'][0][0].shape, (5, 8))
    self.assertEqual(aux['features'][0][1].shape, (5, 6))
    self.assertEqual(aux['self_predictions'][0][0][1].shape, (5, 8))
    self.assertEqual(aux['topdown_predictions'][0][0][1].shape, (5, 8))
    self.assertEqual(aux['basis_predictions'][0][1][1].shape, (5, 4))

  def test_prednet_losses_are_finite_with_task_vector(self):
    obs_size = 5
    sf_dim = 3
    action_size = 2
    policy_obs_size = obs_size + sf_dim
    sac_network = sac_networks.make_sac_networks(
        observation_size=policy_obs_size,
        action_size=action_size,
        hidden_layer_sizes=(8, 8),
        prednet_gammas=(0.1, 0.5),
        sf_dim=sf_dim,
    )
    key_policy, key_q = jax.random.split(jax.random.PRNGKey(0))
    policy_params = sac_network.policy_network.init(key_policy)
    q_params = sac_network.q_network.init(key_q)
    transitions = types.Transition(
        observation=jnp.zeros((4, policy_obs_size)),
        action=jnp.zeros((4, action_size)),
        reward=jnp.ones((4,)),
        discount=jnp.ones((4,)),
        next_observation=jnp.zeros((4, policy_obs_size)),
        extras={'state_extras': {'truncation': jnp.zeros((4,))}},
    )
    _, critic_loss, task_loss, _ = sac_losses.make_losses(
        sac_network=sac_network,
        reward_scaling=1.0,
        discounting=0.99,
        action_size=action_size,
        prednet_gammas=(0.1, 0.5),
        sf_dim=sf_dim,
    )

    critic_value, critic_metrics = critic_loss(
        q_params,
        policy_params,
        None,
        q_params,
        jnp.asarray(1.0),
        transitions,
        jax.random.PRNGKey(1),
    )
    task_value, task_metrics = task_loss(
        jnp.ones((sf_dim,)),
        q_params,
        policy_params,
        None,
        transitions,
        jax.random.PRNGKey(2),
    )

    self.assertTrue(jnp.isfinite(critic_value))
    self.assertTrue(jnp.isfinite(critic_metrics['self_prediction_loss']))
    self.assertTrue(jnp.isfinite(critic_metrics['topdown_prediction_loss']))
    self.assertTrue(jnp.isfinite(task_value))
    self.assertTrue(jnp.isfinite(task_metrics['reward_prediction_loss']))

  def test_train_smoke(self):
    fast = envs.get_environment('fast')
    _, _, metrics = sac.train(
        fast,
        num_timesteps=128,
        episode_length=64,
        num_envs=16,
        learning_rate=3e-4,
        discounting=0.99,
        batch_size=16,
        normalize_observations=True,
        reward_scaling=10,
        grad_updates_per_step=1,
        num_evals=1,
        seed=0,
        eval_env=envs.get_environment('fast'),
        max_devices_per_host=1,
    )
    self.assertIn('eval/episode_reward', metrics)

  def test_train_smoke_with_task_vector(self):
    fast = envs.get_environment('fast')
    _, params, metrics = sac.train(
        fast,
        num_timesteps=128,
        episode_length=64,
        num_envs=16,
        learning_rate=3e-4,
        discounting=0.99,
        batch_size=16,
        normalize_observations=True,
        reward_scaling=10,
        grad_updates_per_step=1,
        num_evals=1,
        seed=0,
        eval_env=envs.get_environment('fast'),
        max_devices_per_host=1,
        prednet_use_task_vector=True,
        sf_dim=4,
        return_q_params=True,
    )
    self.assertIn('eval/episode_reward', metrics)
    self.assertLen(params, 5)

  @parameterized.parameters(True, False)
  def test_network_encoding(self, normalize_observations):
    env = envs.get_environment('fast')
    original_inference, params, _ = sac.train(
        env,
        num_timesteps=128,
        episode_length=64,
        num_envs=16,
        batch_size=16,
        normalize_observations=normalize_observations,
        max_devices_per_host=1,
    )
    normalize_fn = lambda x, y: x
    if normalize_observations:
      normalize_fn = running_statistics.normalize
    sac_network = sac_networks.make_sac_networks(
        env.observation_size, env.action_size, normalize_fn
    )
    inference = sac_networks.make_inference_fn(sac_network)
    byte_encoding = pickle.dumps(params)
    decoded_params = pickle.loads(byte_encoding)

    state = env.reset(jax.random.PRNGKey(0))
    original_action = original_inference(decoded_params)(
        state.obs, jax.random.PRNGKey(0)
    )[0]
    action = inference(decoded_params)(state.obs, jax.random.PRNGKey(0))[0]
    self.assertSequenceEqual(original_action, action)

  def test_train_domain_randomize(self):
    """Test with domain randomization."""

    def rand_fn(sys, rng):
      @jax.vmap
      def get_offset(rng):
        offset = jax.random.uniform(rng, shape=(3,), minval=-0.1, maxval=0.1)
        pos = sys.link.transform.pos.at[0].set(offset)
        return pos

      sys_v = sys.tree_replace({'link.inertia.transform.pos': get_offset(rng)})
      in_axes = jax.tree.map(lambda x: None, sys)
      in_axes = in_axes.tree_replace({'link.inertia.transform.pos': 0})
      return sys_v, in_axes

    _, _, _ = sac.train(
        envs.get_environment('inverted_pendulum', backend='spring'),
        num_timesteps=128,
        num_envs=16,
        episode_length=64,
        batch_size=16,
        randomization_fn=rand_fn,
        max_devices_per_host=1,
    )


if __name__ == '__main__':
  absltest.main()
