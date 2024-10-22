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
    return (predictions.argmax(axis=-1) == labels.argmax(axis=-1)).mean()


@eqx.filter_jit
def permute_and_test(model, permutations, image_batch, label_batch, max_perm_parallel=25, samples=None, rng=None):
    """ When having too many permutations, we can't vmap everything at once. This function splits the permutations
    into smaller chunks and computes the accuracy for each chunk of max_perm_parallel permutations at a time.
    """
    def test_batch_permutation_fn(model, image_batch, label_batch, batched_permutations, samples=None, rng=None):
        def compute_perm_accuracy(carry, data):
            perm = data
            task_images = image_batch.reshape(
                image_batch.shape[0], -1)[:, perm].reshape(image_batch.shape)
            accuracy = compute_accuracy(
                model, task_images, label_batch, samples, rng)
            return carry, accuracy
        _, accuracies = jax.lax.scan(
            f=compute_perm_accuracy, init=(), xs=(batched_permutations))
        return accuracies
    # Split the permutations into smaller chunks
    batched_permutations = jnp.array(
        jnp.split(permutations, len(permutations) // max_perm_parallel)) if len(permutations) > max_perm_parallel else jnp.array([permutations])
    # Compute the accuracy for each chunk of permutations
    accuracies = jax.vmap(test_batch_permutation_fn, in_axes=(None, None, None, 0, None, None))(
        model, image_batch, label_batch, batched_permutations, samples, split(rng)[0])
    # Flatten the results in the first two dimensions
    accuracies = accuracies.reshape(-1)
    return accuracies


@eqx.filter_jit
def test_fn(model: eqx.Module,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            rng,
            max_perm_parallel=None,
            permutations=None,
            test_samples=None):
    if permutations is not None:
        accuracies = jnp.zeros(len(permutations))
        accuracies = jax.vmap(permute_and_test, in_axes=(None, None, 0, 0, None, None, None))(
            model, permutations, images, labels, max_perm_parallel, test_samples, rng).mean(axis=0)
    else:
        accuracies = jnp.array([jax.vmap(compute_accuracy, in_axes=(None, 0, 0, None, None))(
            model, images, labels, test_samples, rng).mean()])
    return accuracies
