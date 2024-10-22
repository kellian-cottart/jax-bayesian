import jax
import jax.numpy as jnp
import equinox as eqx
import functools as ft
from jax.random import split


@eqx.filter_jit
def test_fn_bayesian(model, images, samples=None, rng=None):
    return jax.vmap(model, in_axes=(0, None, None))(
        images, samples, split(rng)[0]).mean(axis=1)


@eqx.filter_jit
def test_fn_deterministic(model, images):
    return jax.vmap(model)(images)


@eqx.filter_jit
def compute_accuracy(model, images, labels, samples=None, rng=None):
    if samples is not None:
        predictions = test_fn_bayesian(model, images, samples, rng)
    else:
        predictions = test_fn_deterministic(model, images)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == jnp.argmax(labels, axis=-1))


@eqx.filter_jit
def permute_and_test(model, permutations, image_batch, label_batch, max_perm_parallel=25, samples=None, rng=None):
    """ We can't fit everything in one GPU when using too many permutations, so we must split permutations into batches """
    def test_batch_permutation_fn(model, image_batch, label_batch, batched_permutations, samples=None, rng=None):
        accuracies = jnp.zeros(len(batched_permutations))
        for i, perm in enumerate(batched_permutations):
            task_images = image_batch.reshape(
                image_batch.shape[0], -1)[:, perm].reshape(image_batch.shape)
            accuracies = accuracies.at[i].set(
                compute_accuracy(model, task_images, label_batch, samples, rng))
        return accuracies
    batched_permutations = jnp.array(
        jnp.split(permutations, len(permutations) // max_perm_parallel)) if len(permutations) > max_perm_parallel else jnp.array([permutations])
    accuracies = jax.vmap(test_batch_permutation_fn, in_axes=(None, None, None, 0, None, None))(
        model, image_batch, label_batch, batched_permutations, samples, split(rng)[0])
    # Flatten the results in the first two dimensions
    accuracies = accuracies.reshape(
        accuracies.shape[0] * accuracies.shape[1], -1)
    return accuracies
