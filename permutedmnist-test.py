import jax
import jax.numpy as jnp
import equinox as eqx
from tqdm import tqdm
from dataloader import *
from models import *
from datetime import datetime
import os
import json
import functools as ft
from shutil import rmtree
from optimizers import *


@ft.partial(jax.jit, static_argnames=("samples"))
def test_fn(model, images, labels, samples=None, rng=None):
    if samples is not None:
        output = jax.vmap(model, in_axes=(0, None, None))(
            images, samples, rng).mean(axis=1)
    else:
        output = jax.vmap(model)(images)
    pred = jnp.argmax(output, axis=-1)
    labels = jnp.argmax(labels, axis=-1)
    accuracy = jnp.mean(pred == labels)
    return accuracy


@ft.partial(jax.jit, static_argnames=("samples"))
def test_batch_permutation_fn(model, image_batch, label_batch, batched_permutations, samples=None, rng=None):
    accuracies = jnp.zeros(len(batched_permutations))
    for i, perm in enumerate(batched_permutations):
        task_images = image_batch.reshape(
            image_batch.shape[0], -1)[:, perm].reshape(image_batch.shape)
        accuracies = accuracies.at[i].set(
            test_fn(model, task_images, label_batch, samples, rng))
    return accuracies


@ft.partial(jax.jit, static_argnames=("samples", "max_perm_parallel"))
def permute_and_test(model, permutations, image_batch, label_batch, max_perm_parallel=25, samples=None, rng=None):
    """ We can't fit everything in one GPU when using too many permutations, so we must split permutations into batches """
    batched_permutations = jnp.array(
        jnp.split(permutations, len(permutations) // max_perm_parallel)) if len(permutations) > max_perm_parallel else jnp.array([permutations])
    accuracies = jax.vmap(test_batch_permutation_fn, in_axes=(None, None, None, 0, None, None))(
        model, image_batch, label_batch, batched_permutations, samples, rng)
    # Flatten the results in the first two dimensions
    accuracies = accuracies.reshape(
        accuracies.shape[0] * accuracies.shape[1], -1)
    return accuracies


def prepare_data(data, targets, batch_size):
    data = jnp.array(data, dtype=jnp.float32)
    targets = jax.nn.one_hot(
        jnp.array(targets, dtype=jnp.int32), num_classes=num_classes)
    data, targets = data[:len(
        data) - len(data) % batch_size], targets[:len(targets) - len(targets) % batch_size]
    return data.reshape(-1, batch_size, *data.shape[1:]), targets.reshape(-1, batch_size, num_classes)


def configure_networks(configuration, rng):

    print("Configuring network with configuration: ",
          configuration["network"])
    # make a dictionary of maps
    select_network = {
        "bayesianmlp": SmallBayesianNetwork,
        "mlp": SmallNetwork,
    }
    if not "network_params" in configuration:
        raise ValueError("Network parameters not found")
    try:
        model = select_network[configuration["network"]](
            key=rng, **configuration["network_params"])
    except KeyError as e:
        raise KeyError("Error with provided keys: ", e)

    return model


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

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    return optimizer, opt_state


@ft.partial(jax.jit, static_argnames=("samples"))
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
        predictions = jax.vmap(model, in_axes=(0, None, None))(
            images, samples, rng)
        output = jax.nn.log_softmax(
            predictions, axis=-1).mean(axis=1) * labels
    else:
        predictions = jax.vmap(model)(images)
        output = jax.nn.log_softmax(predictions, axis=-1) * labels
    loss = -jnp.sum(jnp.clip(output, -10, -1e-4), axis=-1).sum()
    return loss, predictions


@ ft.partial(jax.jit, static_argnames=("samples"))
def train_fn(carry, data, samples=None):
    """ Training function for models. Receives one batch of data and updates the model by computing the
    loss and its gradients.

    Args:
        carry: tuple containing the model, optimizer state, permutation and random key
        data: tuple containing the images and labels
        samples: number of samples for the model. If None, no sampling is performed.
    """
    model, opt_state, perm, rng = carry
    images, labels = data
    # Compute the loss

    if perm is not None:
        images = images.reshape(
            images.shape[0], -1)[:, perm].reshape(images.shape)
    (loss, predictions), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, images, labels, samples, rng)
    # Update
    model, opt_state = optimizer.update(model, grads, opt_state)
    return (model, opt_state, perm, rng), (loss, predictions)


if __name__ == "__main__":
    N_ITERATIONS = 1
    configurations = [
        {
            "network": "bayesianmlp",
            "network_params": {
                "sigma_init": 0.1,
            },
            "optimizer": "mesu",
            "optimizer_params": {
                "lr_mu": 1,
                "lr_sigma": 1,
                "mu_prior": 0,
                "N_mu": 200_000,
                "N_sigma": 200_000,
                "clamp_grad": 1,
            },
            "task": "PermutedMNIST",
            "n_train_samples": 8,
            "n_test_samples": 8,
            "n_tasks": 100,
            "epochs": 1,
            "train_batch_size": 1,
            "test_batch_size": 128,
            "max_perm_parallel": 25,
            "seed": 0+i
        } for i in range(N_ITERATIONS)
    ]
    # convert all fields to lowercase if string
    for config in configurations:
        config = {k: v.lower() if isinstance(
            v, str) else v for k, v in config.items()}
    # create a timestamp
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S-")
    for k, configuration in enumerate(configurations):
        FOLDER = TIMESTAMP + \
            configuration["task"] + \
            f"- t={configuration['n_tasks']}-e={configuration['epochs']}"
        MAIN_FOLDER = "results"
        SAVE_PATH = os.path.join(MAIN_FOLDER, FOLDER)
        CONFIGURATION_PATH = os.path.join(SAVE_PATH, f"config{k}")
        DATA_PATH = os.path.join(CONFIGURATION_PATH, "data")
        os.makedirs(MAIN_FOLDER, exist_ok=True)
        os.makedirs(CONFIGURATION_PATH, exist_ok=True)
        os.makedirs(SAVE_PATH, exist_ok=True)
        os.makedirs(DATA_PATH, exist_ok=True)
        # save config
        with open(SAVE_PATH + "/config.json", "w") as f:
            json.dump(configuration, f, indent=4)
        try:
            # Load the MNIST dataset
            loader = GPULoading()
            train, test, shape, num_classes = loader.task_selection(
                configuration["task"])
            # Initialize the model
            rng = jax.random.PRNGKey(configuration["seed"])
            # Permutations
            if configuration["task"] == "PermutedMNIST":
                permutations = jnp.array(
                    [jax.random.permutation(key, jnp.array(shape).prod()) for key in jax.random.split(rng, configuration["n_tasks"])])
            model = configure_networks(configuration, rng)
            optimizer, opt_state = configure_optimizer(configuration, model)
            # Prepare datasets
            task_train_images, task_train_labels = prepare_data(
                train.data, train.targets, configuration["train_batch_size"])
            test_train_images, test_train_labels = prepare_data(
                test.data, test.targets, configuration["test_batch_size"])

            train_samples = configuration["n_train_samples"] if "n_train_samples" in configuration else None
            test_samples = configuration["n_test_samples"] if "n_test_samples" in configuration else None
            pbar = tqdm(range(configuration["n_tasks"]), desc="Tasks")
            for task in pbar:
                for epoch in range(configuration["epochs"]):
                    pbar.set_description(
                        f"Task {task+1}/{configuration['n_tasks']} - Epoch {epoch+1}/{configuration['epochs']}")
                    special_perm = permutations[task] if configuration["task"] == "PermutedMNIST" else None
                    (model, opt_state, _, _), (loss, predictions) = jax.lax.scan(f=ft.partial(train_fn, samples=train_samples), init=(
                        model, opt_state, special_perm, rng), xs=(task_train_images, task_train_labels))
                    # Test the model and compute accuracy
                    if epoch % 1 == 0:
                        if configuration["task"] == "PermutedMNIST":
                            accuracies = jnp.zeros(configuration["n_tasks"])
                            accuracies = jax.vmap(permute_and_test, in_axes=(None, None, 0, 0, None, None, None))(
                                model, permutations, test_train_images, test_train_labels, configuration["max_perm_parallel"], test_samples, rng).mean(axis=0)
                        else:
                            accuracies = jnp.array([jax.vmap(test_fn, in_axes=(None, 0, 0, None, None))(
                                model, test_train_images, test_train_labels, test_samples, rng).mean()])
                        tqdm.write("=" * 20)
                        for i, acc in enumerate(accuracies):
                            tqdm.write(f"{acc.item()*100:.2f}%", end="\t" if i % 10 !=
                                       9 and i != len(accuracies) - 1 else "\n")
                        # Save accuracy as jax array
                        with open(os.path.join(DATA_PATH, f"task{task}-epoch{epoch}.npy"), "wb") as f:
                            jnp.save(f, accuracies)
        except (KeyboardInterrupt, SystemExit, Exception) as e:
            print(e)
            rmtree(SAVE_PATH)
