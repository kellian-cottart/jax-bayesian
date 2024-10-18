import optax
import jax.numpy as jnp
import jax


def mesu(
        lr_mu: float = 1,
        lr_sigma: float = 1,
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
        def check_bayesian(path, param):
            if len(path) != 3:
                raise ValueError("The network doesn't have 2 parameters.")
            if not ("mu" in path[2].name or "sigma" in path[2].name):
                raise ValueError(
                    "All parameters should be Bayesian, i.e contain mu or sigma in their parameters.")
            if "sigma" in path[2].name:
                return param.copy()
            elif "mu" in path[2].name:
                return jnp.ones_like(param) * mu_prior
            else:
                return param.copy()

        prior = jax.tree_util.tree_map_with_path(check_bayesian, params)
        return {
            'step': 0,
            'prior': prior,
        }

    def update(params, gradients, state):
        state['step'] += 1
        # Update the parameters using sgd

        def update_mesu(param, grad, prior):
            """ Update the parameters based on the gradients and the prior"""
            if clamp_grad > 0:
                grad = jax.tree_util.tree_map(
                    lambda x: jnp.clip(x, -clamp_grad/param.sigma, clamp_grad/param.sigma), grad)
            # Update mu
            variance = param.sigma ** 2
            sigma_p_sq = prior.sigma ** 2
            mu_update = param.mu + lr_mu * \
                variance * (-grad.mu + (prior.mu - param.mu) /
                            (N_mu * sigma_p_sq))
            # Update sigma
            sigma_update = param.sigma + 0.5 * lr_sigma * (- variance * grad.sigma + param.sigma / (N_sigma * sigma_p_sq) * (
                sigma_p_sq - variance))

            def update_param(path, param):
                """ Update the parameters based on the path by iterating the tree
                """
                if path[-1].name == 'mu':
                    return mu_update
                elif path[-1].name == 'sigma':
                    return sigma_update
            return jax.tree_util.tree_map_with_path(update_param, param)

        def discriminant(param):
            """ Discriminate between Bayesian parameters"""
            return hasattr(param, 'mu') and hasattr(param, 'sigma') and param.mu is not None and param.sigma is not None

        updates = jax.tree.map(update_mesu, params,
                               gradients, state['prior'], is_leaf=discriminant)
        return updates, state

    return optax.GradientTransformation(init, update)
