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

"""Soft Actor-Critic losses with predictive-coding auxiliary losses."""

from typing import Any, Sequence

from custombrax.training import types
from custombrax.training.agents.sac_prednet import networks as sac_networks
from custombrax.training.types import Params
from custombrax.training.types import PRNGKey
import jax
import jax.numpy as jnp

Transition = types.Transition


def _masked_mse(error: jnp.ndarray, truncation: jnp.ndarray) -> jnp.ndarray:
  error *= jnp.expand_dims(1 - truncation, -1)
  return 0.5 * jnp.mean(jnp.square(error))


def _mean_or_zero(losses: list[jnp.ndarray]) -> jnp.ndarray:
  if not losses:
    return jnp.asarray(0.0, dtype=jnp.float32)
  return sum(losses) / len(losses)


def _average_basis_predictions(aux: dict[str, Any]) -> jnp.ndarray:
  basis_predictions = []
  for critic_basis_predictions in aux['basis_predictions']:
    for layer_basis_predictions in critic_basis_predictions:
      basis_predictions.extend(layer_basis_predictions)
  return sum(basis_predictions) / len(basis_predictions)


def make_losses(
    sac_network: sac_networks.SACNetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
    prednet_gammas: Sequence[float] = (0.1, 0.5, 0.95),
    prednet_loss_weight: float = 0.1,
    prednet_self_weight: float = 1.0,
    prednet_topdown_weight: float = 1.0,
    sf_dim: int = 0,
    normalize_sf_features: bool = True,
):
  """Creates the SAC-PredNet losses."""

  target_entropy = -0.5 * action_size
  policy_network = sac_network.policy_network
  q_network = sac_network.q_network
  parametric_action_distribution = sac_network.parametric_action_distribution
  gammas = tuple(float(gamma) for gamma in prednet_gammas)

  def alpha_loss(
      log_alpha: jnp.ndarray,
      policy_params: Params,
      normalizer_params: Any,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
    """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
    dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key
    )
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    alpha = jnp.exp(log_alpha)
    alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
    return jnp.mean(alpha_loss)

  def critic_loss(
      q_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      key: PRNGKey,
  ) -> tuple[jnp.ndarray, types.Metrics]:
    q_old_action, current_aux = q_network.apply(
        normalizer_params,
        q_params,
        transitions.observation,
        transitions.action,
        return_aux=True,
    )
    next_dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, key
    )
    next_log_prob = parametric_action_distribution.log_prob(
        next_dist_params, next_action
    )
    next_action = parametric_action_distribution.postprocess(next_action)
    next_q, target_next_aux = q_network.apply(
        normalizer_params,
        target_q_params,
        transitions.next_observation,
        next_action,
        return_aux=True,
    )
    next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
    target_q = jax.lax.stop_gradient(
        transitions.reward * reward_scaling
        + transitions.discount * discounting * next_v
    )
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_loss = _masked_mse(q_error, truncation)

    self_losses = []
    topdown_losses = []
    basis_losses = []
    transition_discount = jnp.expand_dims(transitions.discount, -1)
    for critic_idx, critic_features in enumerate(current_aux['features']):
      target_critic_features = target_next_aux['features'][critic_idx]
      critic_self_predictions = current_aux['self_predictions'][critic_idx]
      target_critic_self_predictions = target_next_aux['self_predictions'][
          critic_idx
      ]
      for layer_idx, _ in enumerate(critic_features):
        for gamma_idx, gamma in enumerate(gammas):
          target = jax.lax.stop_gradient(
              target_critic_features[layer_idx]
              + transition_discount
              * gamma
              * target_critic_self_predictions[layer_idx][gamma_idx]
          )
          self_losses.append(
              _masked_mse(
                  critic_self_predictions[layer_idx][gamma_idx] - target,
                  truncation,
              )
          )

      critic_topdown_predictions = current_aux['topdown_predictions'][
          critic_idx
      ]
      target_critic_topdown_predictions = target_next_aux[
          'topdown_predictions'
      ][critic_idx]
      for topdown_idx, layer_topdown_predictions in enumerate(
          critic_topdown_predictions
      ):
        lower_layer_idx = topdown_idx
        for gamma_idx, gamma in enumerate(gammas):
          target = jax.lax.stop_gradient(
              target_critic_features[lower_layer_idx]
              + transition_discount
              * gamma
              * target_critic_topdown_predictions[topdown_idx][gamma_idx]
          )
          topdown_losses.append(
              _masked_mse(
                  layer_topdown_predictions[gamma_idx] - target,
                  truncation,
              )
          )

      if sf_dim:
        critic_basis_features = target_next_aux['basis_features'][critic_idx]
        critic_basis_predictions = current_aux['basis_predictions'][
            critic_idx
        ]
        target_critic_basis_predictions = target_next_aux[
            'basis_predictions'
        ][critic_idx]
        for layer_idx, layer_basis_predictions in enumerate(
            critic_basis_predictions
        ):
          for gamma_idx, gamma in enumerate(gammas):
            target = jax.lax.stop_gradient(
                critic_basis_features[layer_idx]
                + transition_discount
                * gamma
                * target_critic_basis_predictions[layer_idx][gamma_idx]
            )
            basis_losses.append(
                _masked_mse(
                    layer_basis_predictions[gamma_idx] - target,
                    truncation,
                )
            )

    self_prediction_loss = _mean_or_zero(self_losses)
    topdown_prediction_loss = _mean_or_zero(topdown_losses)
    basis_prediction_loss = _mean_or_zero(basis_losses)
    self_prediction_loss = self_prediction_loss + basis_prediction_loss
    aux_loss = (
        prednet_self_weight * self_prediction_loss
        + prednet_topdown_weight * topdown_prediction_loss
    )
    total_loss = q_loss + prednet_loss_weight * aux_loss
    metrics = {
        'q_loss': q_loss,
        'self_prediction_loss': self_prediction_loss,
        'topdown_prediction_loss': topdown_prediction_loss,
        'basis_prediction_loss': basis_prediction_loss,
        'aux_loss': aux_loss,
        'total_critic_loss': total_loss,
    }
    return total_loss, metrics

  def task_loss(
      task_params: Params,
      q_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      transitions: Transition,
      key: PRNGKey,
  ) -> tuple[jnp.ndarray, types.Metrics]:
    next_dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, key
    )
    next_action = parametric_action_distribution.postprocess(next_action)
    _, next_aux = q_network.apply(
        normalizer_params,
        q_params,
        transitions.next_observation,
        next_action,
        return_aux=True,
    )
    basis_features = _average_basis_predictions(next_aux)
    if normalize_sf_features:
      basis_features = basis_features / (
          jnp.linalg.norm(basis_features, axis=-1, keepdims=True) + 1e-8
      )
    predicted_reward = jnp.sum(
        jax.lax.stop_gradient(basis_features) * task_params, axis=-1
    )
    reward_target = transitions.reward * reward_scaling
    reward_prediction_loss = jnp.mean(
        jnp.square(predicted_reward - reward_target)
    )
    return reward_prediction_loss, {
        'reward_prediction_loss': reward_prediction_loss,
        'predicted_reward': jnp.mean(predicted_reward),
        'task_norm': jnp.linalg.norm(task_params),
    }

  def actor_loss(
      policy_params: Params,
      normalizer_params: Any,
      q_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
    dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key
    )
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    action = parametric_action_distribution.postprocess(action)
    q_action = q_network.apply(
        normalizer_params, q_params, transitions.observation, action
    )
    min_q = jnp.min(q_action, axis=-1)
    actor_loss = alpha * log_prob - min_q
    return jnp.mean(actor_loss)

  return alpha_loss, critic_loss, task_loss, actor_loss
