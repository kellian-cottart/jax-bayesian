import jax.numpy as jnp
import jax
import optax


def sgd(lr: float = 0.001,):
    """
    SGD

    Args:
        lr: Learning rate for the optimizer.
    """

    def init(params):
        return {
            'step': 0,
        }

    def update(params, gradient, state):
        state['step'] += 1
        # Update the parameters using sgd

        def udpate_sgd(param, grad):
            return param - lr * grad if grad is not None else param
        updates = jax.tree_util.tree_map(udpate_sgd, params, gradient)
        return updates, state

    return optax.GradientTransformation(init, update)
