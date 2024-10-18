import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm
from dataloader import *
from models import *
from datetime import datetime
import os
import json
import functools as ft
from shutil import rmtree


@ft.partial(jax.jit, static_argnames=("samples"))
def loss_fn(model, images, labels, perm=None, samples=None, rng=None):
    if perm is not None:
        images = images.reshape(
            images.shape[0], -1)[:, perm].reshape(images.shape)
    if samples is not None:
        key, rng = jax.random.split(rng)
        output = jax.vmap(model, in_axes=(0, None, None))(
            images, samples, key).mean(axis=1)
    else:
        output = jax.vmap(model)(images)
    loss = -jnp.sum(jax.nn.log_softmax(output) * labels)
    return loss


@ft.partial(jax.jit, static_argnames=("samples"))
def train_fn(carry, data, samples=None):
    model, opt_state, perm, rng = carry
    images, labels = data
    # Compute the loss
    loss, params = jax.value_and_grad(loss_fn)(
        model, images, labels, perm, samples, rng)
    # Update
    updates, opt_state = optimizer.update(params, opt_state)
    model = optax.apply_updates(model, updates)
    return (model, opt_state, perm, rng), loss


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


@ eqx.filter_jit
def test_batch_permutation_fn(model, image_batch, label_batch, batched_permutations):
    accuracies = jnp.zeros(len(batched_permutations))
    for i, perm in enumerate(batched_permutations):
        task_images = image_batch.reshape(
            image_batch.shape[0], -1)[:, perm].reshape(image_batch.shape)
        accuracies = accuracies.at[i].set(
            test_fn(model, task_images, label_batch))
    return accuracies


@ eqx.filter_jit
def permute_and_test(model, permutations, image_batch, label_batch, max_perm_parallel=25):
    """ We can't fit everything in one GPU when using too many permutations, so we must split permutations into batches """
    batched_permutations = jnp.array(
        jnp.split(permutations, len(permutations) // max_perm_parallel)) if len(permutations) > max_perm_parallel else jnp.array([permutations])
    accuracies = jax.vmap(test_batch_permutation_fn, in_axes=(None, None, None, 0))(
        model, image_batch, label_batch, batched_permutations)
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
    select_optimizer = {
        "sgd": optax.sgd,
        "adam": optax.adam,
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


if __name__ == "__main__":
    N_ITERATIONS = 1
    configurations = [
        {
            "network": "bayesianmlp",
            "network_params":   {
                "sigma_init": 0.1
            },
            "optimizer": "sgd",
            "optimizer_params": {
                "learning_rate": 0.001
            },
            "task": "MNIST",
            "n_train_samples": 1,
            "n_test_samples": 1,
            "n_tasks": 1,
            "epochs": 100,
            "train_batch_size": 1,
            "test_batch_size": 128,
            "seed": 1000 + i,
            "max_perm_parallel": 25,
            "sigma_init": 0.1,
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

            for task in tqdm(range(configuration["n_tasks"]), desc="Tasks"):
                tqdm.write(f"Task {task+1}/{configuration['n_tasks']}")
                # Train the model
                for epoch in tqdm(range(configuration["epochs"]), desc="Epochs"):

                    special_perm = permutations[task] if configuration["task"] == "PermutedMNIST" else None
                    (model, opt_state, _, _), loss = jax.lax.scan(f=ft.partial(train_fn, samples=train_samples), init=(
                        model, opt_state, special_perm, rng), xs=(task_train_images, task_train_labels))
                    # Test the model and compute accuracy
                    if epoch % 1 == 0:
                        if configuration["task"] == "PermutedMNIST":
                            accuracies = jnp.zeros(configuration["n_tasks"])
                            accuracies = jax.vmap(permute_and_test, in_axes=(None, None, 0, 0, None))(
                                model, permutations, test_train_images, test_train_labels, configuration["max_perm_parallel"]).mean(dim=0)
                        else:
                            accuracies = jnp.mean(jax.vmap(test_fn, in_axes=(None, 0, 0, None, None))(
                                model, test_train_images, test_train_labels, test_samples, rng))
                            accuracies = jnp.array([accuracies])
                        for i, acc in enumerate(accuracies):
                            tqdm.write(f"{acc.item()*100:.2f}%", end="\t" if i % 10 !=
                                       9 and i != len(accuracies) - 1 else "\n")
                        tqdm.write(
                            f"Epoch {epoch+1}/{configuration['epochs']} - Loss: {loss.mean():.2f}")
                        # Save accuracy as jax array
                        with open(os.path.join(DATA_PATH, f"task{task}-epoch{epoch}.npy"), "wb") as f:
                            jnp.save(f, accuracies)
        except KeyboardInterrupt as e:
            rmtree(SAVE_PATH)
        except Exception as e:
            rmtree(SAVE_PATH)
            raise e
