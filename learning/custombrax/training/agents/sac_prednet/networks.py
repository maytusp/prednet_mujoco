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

"""SAC-PredNet networks."""

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
  """Creates params and inference function for the SAC-PredNet agent."""

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
    prednet_gammas: Sequence[float] = (0.1, 0.5, 0.95),
    sf_dim: int = 0,
) -> SACNetworks:
  """Make SAC-PredNet networks."""
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
  q_network = make_prednet_q_network(
      observation_size,
      action_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=q_network_layer_norm,
      kernel_init=q_network_kernel_init_fn(**q_kernel_init_kwargs),
      prednet_gammas=prednet_gammas,
      sf_dim=sf_dim,
  )
  return SACNetworks(
      policy_network=policy_network,
      q_network=q_network,
      parametric_action_distribution=parametric_action_distribution,
  )


def make_prednet_q_network(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    n_critics: int = 2,
    layer_norm: bool = False,
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform(),
    prednet_gammas: Sequence[float] = (0.1, 0.5, 0.95),
    sf_dim: int = 0,
) -> networks.FeedForwardNetwork:
  """Creates a scalar Q critic with predictive-coding auxiliary heads."""
  gammas = tuple(float(gamma) for gamma in prednet_gammas)
  hidden_sizes = tuple(int(size) for size in hidden_layer_sizes)

  class PredNetQModule(linen.Module):
    """Q module exposing hidden hierarchy and predictive heads."""

    n_critics: int

    @linen.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
      critic_input = jnp.concatenate([obs, actions], axis=-1)
      qs = []
      all_features = []
      all_self_predictions = []
      all_topdown_predictions = []
      all_basis_features = []
      all_basis_predictions = []

      for critic_idx in range(self.n_critics):
        hidden = critic_input
        features = []
        for layer_idx, hidden_size in enumerate(hidden_sizes):
          hidden = linen.Dense(
              hidden_size,
              name=f'critic_{critic_idx}_hidden_{layer_idx}',
              kernel_init=kernel_init,
          )(hidden)
          hidden = activation(hidden)
          if layer_norm:
            hidden = linen.LayerNorm(
                name=f'critic_{critic_idx}_ln_{layer_idx}'
            )(hidden)
          features.append(hidden)

        q = linen.Dense(
            1,
            name=f'critic_{critic_idx}_q',
            kernel_init=kernel_init,
        )(hidden)
        qs.append(q)
        all_features.append(tuple(features))

        critic_self_predictions = []
        critic_topdown_predictions = []
        critic_basis_features = []
        critic_basis_predictions = []
        for layer_idx, feature in enumerate(features):
          layer_self_predictions = []
          layer_basis_predictions = []
          for gamma_idx, _ in enumerate(gammas):
            layer_self_predictions.append(
                linen.Dense(
                    feature.shape[-1],
                    name=(
                        f'critic_{critic_idx}_self_l{layer_idx}_g{gamma_idx}'
                    ),
                    kernel_init=kernel_init,
                )(feature)
            )
            if sf_dim:
              layer_basis_predictions.append(
                  linen.Dense(
                      sf_dim,
                      name=(
                          f'critic_{critic_idx}_basis_pred_l{layer_idx}'
                          f'_g{gamma_idx}'
                      ),
                      kernel_init=kernel_init,
                  )(feature)
              )
          critic_self_predictions.append(tuple(layer_self_predictions))
          if sf_dim:
            critic_basis_features.append(
                linen.Dense(
                    sf_dim,
                    name=f'critic_{critic_idx}_basis_l{layer_idx}',
                    kernel_init=kernel_init,
                )(feature)
            )
            critic_basis_predictions.append(tuple(layer_basis_predictions))

        for layer_idx in range(1, len(features)):
          higher_feature = features[layer_idx]
          lower_size = features[layer_idx - 1].shape[-1]
          layer_topdown_predictions = []
          for gamma_idx, _ in enumerate(gammas):
            layer_topdown_predictions.append(
                linen.Dense(
                    lower_size,
                    name=(
                        f'critic_{critic_idx}_topdown_l{layer_idx}_g'
                        f'{gamma_idx}'
                    ),
                    kernel_init=kernel_init,
                )(higher_feature)
            )
          critic_topdown_predictions.append(tuple(layer_topdown_predictions))

        all_self_predictions.append(tuple(critic_self_predictions))
        all_topdown_predictions.append(tuple(critic_topdown_predictions))
        all_basis_features.append(tuple(critic_basis_features))
        all_basis_predictions.append(tuple(critic_basis_predictions))

      aux = {
          'features': tuple(all_features),
          'self_predictions': tuple(all_self_predictions),
          'topdown_predictions': tuple(all_topdown_predictions),
          'basis_features': tuple(all_basis_features),
          'basis_predictions': tuple(all_basis_predictions),
      }
      return jnp.concatenate(qs, axis=-1), aux

  q_module = PredNetQModule(n_critics=n_critics)

  def apply(processor_params, q_params, obs, actions, return_aux=False):
    obs = preprocess_observations_fn(obs, processor_params)
    q, aux = q_module.apply(q_params, obs, actions)
    if return_aux:
      return q, aux
    return q

  dummy_obs = jnp.zeros((1, observation_size))
  dummy_action = jnp.zeros((1, action_size))
  return networks.FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs, dummy_action),
      apply=apply,
  )
