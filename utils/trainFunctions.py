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
def ewc_loss_fn(model, factor, fisher, old_param, images, labels):
    """ Fisher is a pytree, so is old_param. """
    predictions = jax.vmap(model)(images)
    output = jax.nn.log_softmax(predictions, axis=-1) * labels

    difference_squared = jax.tree.map(
        lambda fisher, new, old: fisher *
        (new - old) ** 2, eqx.filter(fisher, eqx.is_array), eqx.filter(model, eqx.is_array), eqx.filter(old_param, eqx.is_array))
    # sum over all the parameters
    ewc_loss = jnp.sum(jax.tree_leaves(difference_squared)[1])
    return -jnp.sum(output, axis=-1).sum() + factor/2*ewc_loss


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


@ eqx.filter_jit
def train_fn(model, task_train_images, task_train_labels, opt_state, optimizer, train_ck, special_perm=None, train_samples=None):
    """ Train the model on the task.
    Splits the model into dynamic and static parts to allow for eqx.filter_jit compilation.
    """
    dynamic_init_state, static_state = eqx.partition(
        model, eqx.is_array)

    def batch_fn(dynamic_model, opt_state, perm, keys, optimizer, images, labels, samples=None):
        # Apply permutation if provided
        if perm is not None:
            images = images.reshape(
                images.shape[0], -1)[:, perm].reshape(images.shape)
        # Combine dynamic and static parts of the model
        model = eqx.combine(dynamic_model, static_state)
        # Compute the loss and gradients
        loss, grads = loss_fn(
            model, images, labels, samples, keys)
        # Update the model using the optimizer
        dynamic_state, _ = eqx.partition(
            model, eqx.is_array)
        dynamic_state, opt_state = optimizer.update(
            dynamic_state, grads, opt_state)
        return dynamic_state, opt_state, loss

    def scan_fn(carry, data):
        dynamic_state, opt_state = carry
        images, labels, key = data
        # Train the model
        dynamic_state, opt_state, loss = batch_fn(
            dynamic_state, opt_state, special_perm, key, optimizer, images, labels, train_samples)
        return (dynamic_state, opt_state), loss

    train_ck = jax.random.split(
        train_ck, task_train_images.shape[0])
    # Use jax.lax.scan to iterate over the batches
    (dynamic_init_state, opt_state), losses = jax.lax.scan(
        f=scan_fn, init=(dynamic_init_state, opt_state), xs=(
            task_train_images, task_train_labels, train_ck)
    )
    # Combine the dynamic and static parts of the model to recover the activation functions
    model = eqx.combine(dynamic_init_state, static_state)
    return model, opt_state, losses
