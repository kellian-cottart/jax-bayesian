import optax
import jax.numpy as jnp
import jax
from models.gaussianParameter import *


def discriminant(param):
    """ Discriminate between Bayesian parameters"""
    return hasattr(param, 'mu') and hasattr(param, 'sigma') and param.mu is not None and param.sigma is not None


def mesu(
        lr_mu: float = 1,
        lr_sigma: float = 1,
        sigma_prior: float = 0.2,
        mu_prior: float = 0,
        N_mu: int = 1e5,
        N_sigma: int = 1e5,
        clamp_grad: float = 0.) -> optax.GradientTransformation:
    """
    Optax gradient transformation for MESU.

    Args:
        lr_mu: Learning rate for mu parameters.
        lr_sigma: Learning rate for sigma parameters.
        mu_prior: Prior mean value.
        N_mu: Number of batches for mu-based synaptic memory.
        N_sigma: Number of batches for sigma-based synaptic memory.
        clamp_grad: Gradient clamping threshold.

    Returns:
        optax.GradientTransformation: The MESU update rule.
    """

    def init(params):
        # Check if all parameters are bayesian i.e that the path name contains mu or sigma
        def prior_compute(param):
            return GaussianParameter(jnp.ones_like(param.mu) * mu_prior, jnp.ones_like(param.sigma) * sigma_prior)
        prior = jax.tree_util.tree_map(
            prior_compute, params, is_leaf=discriminant)
        return {
            'step': 0,
            'prior': prior,
        }

    def update(params, gradients, state):
        state['step'] += 1
        # Update the parameters using sgd

        def update_mesu(param, grad, prior):
            """ Update the parameters based on the gradients and the prior"""
            # If clamp_grad > 0, then clamp gradients between -clamp_grad/sigma and clamp_grad/sigma
            if clamp_grad > 0:
                grad = jax.tree_util.tree_map(
                    lambda x: jnp.clip(x, -clamp_grad / param.sigma, clamp_grad / param.sigma), grad)
            variance = param.sigma ** 2
            prior_attraction_mu = variance * \
                (mu_prior - param.mu) / (N_mu * (prior.sigma ** 2))
            prior_attraction_sigma = param.sigma * \
                (prior.sigma ** 2 - variance) / (N_sigma * (prior.sigma ** 2))
            mu_update = param.mu + lr_mu * \
                (-variance * grad.mu + prior_attraction_mu)
            sigma_update = param.sigma - 0.5*lr_sigma * variance * \
                grad.sigma + 0.5 * lr_sigma * prior_attraction_sigma
            return GaussianParameter(mu_update, sigma_update)
        updates = jax.tree.map(
            update_mesu,
            params,
            gradients,
            state['prior'],
            is_leaf=discriminant
        )
        return updates, state

    return optax.GradientTransformation(init, update)
