import jax
from jax.random import split
import jax.numpy as jnp
import equinox as eqx
import functools as ft
from optimizers import *
from tqdm import tqdm
from utils.testFunctions import *
import os


def configure_optimizer(configuration, model):
    print("Configuring optimizer with configuration: ",
          configuration["optimizer"])
    select_optimizer = {
        "sgd": sgd,
        "mesu": mesu,
    }
    if not "optimizer_params" in configuration:
        raise ValueError("Optimizer parameters not found")
    try:
        optimizer = select_optimizer[configuration["optimizer"]](
            **configuration["optimizer_params"]
        )
    except KeyError as e:
        raise KeyError("Error with provided keys: ", e)

    opt_state = optimizer.init(model)
    return optimizer, opt_state


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
            bayesian_loss_fn, has_aux=True)(model, images, labels, samples, rng)
    else:
        return eqx.filter_value_and_grad(
            deterministic_loss_fn, has_aux=True)(model, images, labels)


@eqx.filter_jit
def bayesian_loss_fn(model, images, labels, samples, rng):
    """ Loss function for Bayesian models. """
    predictions = jax.vmap(model, in_axes=(0, None, None))(
        images, samples, split(rng)[0])
    output = jax.nn.log_softmax(predictions, axis=-1).mean(axis=1) * labels
    return -jnp.sum(output, axis=-1).sum(), predictions


@eqx.filter_jit
def deterministic_loss_fn(model, images, labels):
    """ Loss function for deterministic models. """
    predictions = jax.vmap(model)(images)
    output = jax.nn.log_softmax(predictions, axis=-1) * labels
    return -jnp.sum(output, axis=-1).sum(), predictions
