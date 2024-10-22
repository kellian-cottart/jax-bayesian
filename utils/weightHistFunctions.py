import matplotlib.pyplot as plt
from models import *
from jax import jit
import jax.numpy as jnp


def histogramWeights(model, path, task, epoch):
    """ Save the weights of the model as a histogram.

    Args:
        model: The model to save the weights from.
        path: The path to save the histogram images.
        task: The task identifier for naming.
        epoch: The epoch number for naming.
    """
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

    # JIT compile the collect_weights function
    collect_weights_jit = jit(collect_weights)
    weights = collect_weights_jit(model)

    # Create subplots for each weight array
    fig, axes = plt.subplots(len(weights), 1, figsize=(5, 5*len(weights)))
    for i, weight in enumerate(weights):
        flattened = weight.flatten()
        max_abs_value = jnp.max(jnp.abs(flattened))
        # Calculate histogram counts and bin edges
        counts, bin_edges = jnp.histogram(
            flattened, bins=100, range=(-max_abs_value, max_abs_value))
        # Convert counts to percentages
        percentages = counts / jnp.sum(counts) * 100
        # Plotting
        axes[i].bar(bin_edges[:-1], percentages, width=bin_edges[1] -
                    bin_edges[0], align='edge', edgecolor='black')
        axes[i].set_title(
            f'Layer {shape(weight)}, Mean: {jnp.mean(jnp.abs(flattened)):.2f}, Std: {jnp.std(jnp.abs(flattened)):.2f}')
        axes[i].set_xlabel('Weights [-]')
        axes[i].set_ylabel('Percentage of Weights [%]')
        # Center the x-axis around 0 based on the maximum absolute value
        axes[i].set_xlim(-max_abs_value, max_abs_value)
        # Set ylim
        axes[i].set_ylim(0, 50)
        # save the counts
        name = None
        if "Bayesian" in model.__class__.__name__:
            if i % 2 == 0:
                name = f"{path}/counts-sigma-task={task}-epoch={epoch}.npy"
            else:
                name = f"{path}/counts-mu-task={task}-epoch={epoch}.npy"
        else:
            name = f"{path}/counts-task = {task}-epoch = {epoch}.npy"

        with open(name, 'wb') as f:
            jnp.save(f, counts)

    # Save the mean and the std of the weights for each layer
    for i, weight in enumerate(weights):
        flattened = weight.flatten()
        mean = jnp.mean(jnp.abs(flattened))
        std = jnp.std(jnp.abs(flattened))
        save = jnp.array([mean, std])
        name = None
        if "Bayesian" in model.__class__.__name__:
            if i % 2 == 0:
                name = f"{path}/param-sigma-task={task}-epoch={epoch}.npy"
            else:
                name = f"{path}/param-mu-task={task}-epoch={epoch}.npy"
        else:
            name = f"{path}/param-task = {task}-epoch = {epoch}.npy"

        with open(name, 'wb') as f:
            jnp.save(f, save)

    # Adjust layout and save the figure
    plt.tight_layout()
    fig.savefig(f"{path}/weights-task={task}-epoch={epoch}.pdf",
                format='pdf')
    plt.close(fig)
