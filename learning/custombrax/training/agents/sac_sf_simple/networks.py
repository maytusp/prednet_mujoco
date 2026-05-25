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

"""SAC networks."""

from typing import Any, Literal, Mapping, Sequence, Tuple

from custombrax.training import distribution
from custombrax.training import networks
from custombrax.training import types
from custombrax.training.types import PRNGKey
import flax
from flax import linen
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class SACNetworks:
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(sac_networks: SACNetworks):
  """Creates params and inference function for the SAC agent."""

  def _append_task(observations: types.Observation, task_params: jnp.ndarray):
    if not task_params.shape[-1]:
      return observations
    task = task_params / (jnp.linalg.norm(task_params) + 1e-8)
    task = jnp.broadcast_to(task, observations.shape[:-1] + task.shape)
    return jnp.concatenate([observations, task], axis=-1)

  def make_policy(
      params: types.PolicyParams, deterministic: bool = False
  ) -> types.Policy:

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      if len(params) == 3:
        normalizer_params, policy_params, task_params = params
        if (
            task_params.shape[-1]
            and observations.shape[-1] == normalizer_params.mean.shape[-1]
        ):
          observations = _append_task(observations, task_params)
        logits = sac_networks.policy_network.apply(
            normalizer_params, policy_params, observations
        )
      else:
        logits = sac_networks.policy_network.apply(*params, observations)
      if deterministic:
        return sac_networks.parametric_action_distribution.mode(logits), {}
      return (
          sac_networks.parametric_action_distribution.sample(
              logits, key_sample
          ),
          {},
      )

    return policy

  return make_policy


def make_sac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    policy_network_kernel_init_fn: networks.Initializer = jax.nn.initializers.lecun_uniform,
    policy_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    q_network_kernel_init_fn: networks.Initializer = jax.nn.initializers.lecun_uniform,
    q_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    sf_dim: int = 0,
    policy_network_type: Literal['mlp', 'cnn'] = 'mlp',
    policy_frame_shape: Sequence[int] = (64, 64, 3),
    policy_normalise_channels: bool = False,
    policy_cnn_output_channels: Sequence[int] = (32, 64, 64),
    policy_cnn_kernel_size: Sequence[int] = (8, 4, 3),
    policy_cnn_stride: Sequence[int] = (4, 2, 1),
    policy_cnn_padding: str = 'SAME',
    policy_cnn_max_pool: bool = False,
    policy_cnn_global_pool: str = 'avg',
) -> SACNetworks:
  """Make SAC networks."""
  policy_kernel_init_kwargs = policy_network_kernel_init_kwargs or {}
  q_kernel_init_kwargs = q_network_kernel_init_kwargs or {}

  parametric_action_distribution: distribution.ParametricDistribution
  if distribution_type == 'normal':
    parametric_action_distribution = distribution.NormalDistribution(
        event_size=action_size
    )
  elif distribution_type == 'tanh_normal':
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
  else:
    raise ValueError(
        f'Unsupported distribution type: {distribution_type}. Must be one'
        ' of "normal" or "tanh_normal".'
    )
  if policy_network_type == 'mlp':
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=policy_network_layer_norm,
        distribution_type=distribution_type,
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
        state_dependent_std=state_dependent_std,
        kernel_init=policy_network_kernel_init_fn(**policy_kernel_init_kwargs),
    )
  elif policy_network_type == 'cnn':
    policy_network = networks.make_policy_network_stacked_frames(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=policy_network_layer_norm,
        frame_shape=policy_frame_shape,
        state_size=sf_dim,
        normalise_channels=policy_normalise_channels,
        distribution_type=distribution_type,
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
        state_dependent_std=state_dependent_std,
        kernel_init=policy_network_kernel_init_fn(**policy_kernel_init_kwargs),
        cnn_output_channels=policy_cnn_output_channels,
        cnn_kernel_size=policy_cnn_kernel_size,
        cnn_stride=policy_cnn_stride,
        cnn_padding=policy_cnn_padding,
        cnn_max_pool=policy_cnn_max_pool,
        cnn_global_pool=policy_cnn_global_pool,
    )
  else:
    raise ValueError(
        f'Unsupported policy_network_type: {policy_network_type}. Must be'
        ' "mlp" or "cnn".'
    )
  if sf_dim:
    q_network = make_sf_q_network(
        observation_size,
        action_size,
        sf_dim=sf_dim,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
        kernel_init=q_network_kernel_init_fn(**q_kernel_init_kwargs),
    )
  else:
    q_network = networks.make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
        kernel_init=q_network_kernel_init_fn(**q_kernel_init_kwargs),
    )
  return SACNetworks(
      policy_network=policy_network,
      q_network=q_network,
      parametric_action_distribution=parametric_action_distribution,
  )


def make_sf_q_network(
    observation_size: int,
    action_size: int,
    sf_dim: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    n_critics: int = 2,
    layer_norm: bool = False,
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform(),
) -> networks.FeedForwardNetwork:
  """Creates a critic that values successor features with task dot product."""

  class SFQModule(linen.Module):
    """Successor-feature Q module."""

    n_critics: int

    @linen.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
      hidden = jnp.concatenate([obs, actions], axis=-1)
      task = obs[..., -sf_dim:]
      qs = []
      sfs = []
      for _ in range(self.n_critics):
        sf = networks.MLP(
            layer_sizes=list(hidden_layer_sizes) + [sf_dim],
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
        )(hidden)
        q = jnp.sum(task * sf, axis=-1, keepdims=True)
        qs.append(q)
        sfs.append(sf)
      return jnp.concatenate(qs, axis=-1), jnp.stack(sfs, axis=1)

  q_module = SFQModule(n_critics=n_critics)

  def apply(processor_params, q_params, obs, actions, return_sf=False):
    obs = preprocess_observations_fn(obs, processor_params)
    q, sf = q_module.apply(q_params, obs, actions)
    if return_sf:
      return q, sf
    return q

  dummy_obs = jnp.zeros((1, observation_size))
  dummy_action = jnp.zeros((1, action_size))
  return networks.FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs, dummy_action),
      apply=apply,
  )
