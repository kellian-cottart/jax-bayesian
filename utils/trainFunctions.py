import jax
from jax.random import split
import jax.numpy as jnp
import equinox as eqx
from optimizers import *
from utils.testFunctions import *


@eqx.filter_jit
def loss_fn(model, images, labels, samples=None, rng=None):
    """ Loss function for the model. Receives the model, images, labels, permutation and samples and returns the loss.

    Args:
        model: the model
        images: the images
        labels: the labels
        perm: the permutation of MNIST if the task is PermutedMNIST
        samples: the number of samples for the model if it is a Bayesian model
        rng: the random key
    """
    if samples is not None:
        return eqx.filter_value_and_grad(
            bayesian_loss_fn)(model, images, labels, samples, rng)
    else:
        return eqx.filter_value_and_grad(
            deterministic_loss_fn)(model, images, labels)


@eqx.filter_jit
def bayesian_loss_fn(model, images, labels, samples, rng):
    """ Loss function for Bayesian models. """
    rng = split(rng, images.shape[0])
    predictions = jax.vmap(model, in_axes=(0, None, 0))(
        images, samples, rng)
    output = jax.nn.log_softmax(predictions, axis=-1).mean(axis=1) * labels
    return -jnp.sum(output, axis=-1).sum()


@ eqx.filter_jit
def deterministic_loss_fn(model, images, labels):
    """ Loss function for deterministic models. """
    predictions = jax.vmap(model)(images)
    output = jax.nn.log_softmax(predictions, axis=-1) * labels
    return -jnp.sum(output, axis=-1).sum()
