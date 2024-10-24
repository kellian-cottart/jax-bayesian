import jax
import jax.numpy as jnp
import equinox as eqx
import functools as ft


@eqx.filter_jit
def test_fn_bayesian(model, images, samples, rng):
    rng = jax.random.split(rng, images.shape[0])
    return jax.vmap(model, in_axes=(0, None, 0))(
        images, samples, rng)


@ eqx.filter_jit
def test_fn_deterministic(model, images):
    return jax.vmap(model)(images)


@ eqx.filter_jit
def compute_accuracy(model, images, labels, samples=None, rng=None):
    if samples is not None:
        predictions = test_fn_bayesian(model, images, samples, rng)
        output = jax.nn.log_softmax(predictions, axis=-1).mean(axis=1)
    else:
        predictions = test_fn_deterministic(model, images)
        output = jax.nn.log_softmax(predictions, axis=-1)
    return (output.argmax(axis=-1) == labels.argmax(axis=-1)).mean(), predictions


@ eqx.filter_jit
def permute_and_test(model, permutations, image_batch, label_batch, max_perm_parallel=25, samples=None, rng=None):
    """ When having too many permutations, we can't vmap everything at once. This function splits the permutations
    into smaller chunks and computes the accuracy for each chunk of max_perm_parallel permutations at a time.
    """
    if not max_perm_parallel > permutations.shape[0]:
        batched_permutations = permutations.reshape(
            max_perm_parallel, permutations.shape[0] // max_perm_parallel, *permutations.shape[1:])
    else:
        batched_permutations = jnp.expand_dims(permutations, 0)

    def test_batch_permutation_fn(model, image_batch, label_batch, batched_permutations, samples=None, rng=None):
        def compute_perm_accuracy(carry, data):
            perm, rng = data
            task_images = image_batch.reshape(
                image_batch.shape[0], -1)[:, perm].reshape(image_batch.shape)
            accuracy, prediction = compute_accuracy(
                model, task_images, label_batch, samples, rng)
            return carry, (accuracy, prediction)
        rng = jax.random.split(rng, len(batched_permutations))
        _, (accuracies, predictions) = jax.lax.scan(
            f=compute_perm_accuracy, init=(), xs=(batched_permutations, rng))
        return accuracies, predictions
    # Compute the accuracy for each chunk of permutations
    rng = jax.random.split(rng, len(batched_permutations))
    accuracies, predictions = jax.vmap(test_batch_permutation_fn, in_axes=(None, None, None, 0, None, 0))(
        model, image_batch, label_batch, batched_permutations, samples, rng)
    # Flatten the results in the first two dimensions
    accuracies = accuracies.reshape(-1)
    predictions = predictions.reshape(-1, *predictions.shape[2:])
    return accuracies, predictions


@ eqx.filter_jit
def test_fn(model: eqx.Module,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            rng,
            max_perm_parallel=None,
            permutations=None,
            test_samples=None):
    if permutations is not None:
        rng = jax.random.split(rng, images.shape[0])

        def scan_fn(carry, data, permutations):
            img, lbl, rng = data
            accuracies, predictions = permute_and_test(
                model, permutations, img, lbl, max_perm_parallel, test_samples, rng)
            return carry, (accuracies, predictions)
        _, (accuracies, predictions) = jax.lax.scan(
            ft.partial(scan_fn, permutations=permutations), init=(), xs=(images, labels, rng))
        accuracies = accuracies.mean(axis=0)
    else:
        accuracies, predictions = jax.vmap(compute_accuracy, in_axes=(
            None, 0, 0, None, None))(model, images, labels, test_samples, rng)

        accuracies = jnp.expand_dims(accuracies.mean(), 0)
        predictions = jnp.expand_dims(predictions, 0)
    predictions = predictions.reshape(
        predictions.shape[1], predictions.shape[0]*predictions.shape[2], *predictions.shape[3:])
    return accuracies, predictions
