import jax.numpy as jnp
import equinox as eqx
from models import GaussianParameter


def histogramWeights(model, path, task, epoch):
    """ Save the weights of the model as a histogram.

    Args:
        model: The model to save the weights from.
        path: The path to save the histogram images.
        task: The task identifier for naming.
        epoch: The epoch number for naming.
    """

    @eqx.filter_jit
    def collect_weights(model):
        weights = []
        # Collect weights and biases from the model
        for layer in model.layers:
            if layer is None:
                continue
            if isinstance(layer.weight, GaussianParameter):
                weights.append(layer.weight.sigma)
                weights.append(layer.weight.mu)
            else:
                weights.append(layer.weight)
            if layer.use_bias:
                if isinstance(layer.bias, GaussianParameter):
                    weights.append(layer.bias.sigma)
                    weights.append(layer.bias.mu)
                else:
                    weights.append(layer.bias)
        return weights

    @eqx.filter_jit
    def process_weight(weight):
        flattened = weight.flatten()
        max_abs_value = jnp.max(jnp.abs(flattened))

        mean = jnp.mean(jnp.abs(flattened))
        std = jnp.std(jnp.abs(flattened))
        save = jnp.array([mean, std])

        # Calculate histogram counts and bin edges
        counts, bin_edges = jnp.histogram(
            flattened, bins=100, range=(-max_abs_value, max_abs_value))
        return counts, save

    # Collect and process weights
    weights = collect_weights(model)
    # Initialize lists to store all counts and parameters
    all_counts = []
    all_params = []

    # Iterate over weights and process them
    for weight in weights:
        counts, save = process_weight(weight)
        all_counts.append(counts)
        all_params.append(save)

    # Convert lists to arrays
    all_counts = jnp.array(all_counts)
    all_params = jnp.array(all_params)

    # Save all counts and parameters in single files
    counts_name = f"{path}/counts-task={task}-epoch={epoch}.npy"
    params_name = f"{path}/params-task={task}-epoch={epoch}.npy"

    with open(counts_name, 'wb') as f:
        jnp.save(f, all_counts)
    with open(params_name, 'wb') as f:
        jnp.save(f, all_params)
