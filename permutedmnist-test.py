import jax
import jax.experimental
import jax.numpy as jnp
import equinox as eqx
from dataloader import *
from models import *
from datetime import datetime
import os
import json
import functools as ft
from shutil import rmtree
from optimizers import *
import traceback
from utils import *


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


if __name__ == "__main__":
    N_ITERATIONS = 1
    configurations = [
        {
            "network": "bayesianmlp",
            "network_params": {
                "sigma_init": 0.2,
            },
            "optimizer": "mesu",
            "optimizer_params": {
                "lr_mu": 1,
                "lr_sigma": 1,
                "mu_prior": 0,
                "N_mu": 200_000,
                "N_sigma": 200_000,
                "clamp_grad": 0.1
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
            f"-t={configuration['n_tasks']}-e={configuration['epochs']}"
        MAIN_FOLDER = "results"
        SAVE_PATH = os.path.join(MAIN_FOLDER, FOLDER)
        CONFIGURATION_PATH = os.path.join(SAVE_PATH, f"config{k}")
        DATA_PATH = os.path.join(CONFIGURATION_PATH, "data")
        WEIGHTS_PATH = os.path.join(CONFIGURATION_PATH, "weights")
        os.makedirs(MAIN_FOLDER, exist_ok=True)
        os.makedirs(CONFIGURATION_PATH, exist_ok=True)
        os.makedirs(SAVE_PATH, exist_ok=True)
        os.makedirs(DATA_PATH, exist_ok=True)
        os.makedirs(WEIGHTS_PATH, exist_ok=True)
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
            # Prepare datasets
            task_train_images, task_train_labels = prepare_data(
                train.data, train.targets, configuration["train_batch_size"])
            test_train_images, test_train_labels = prepare_data(
                test.data, test.targets, configuration["test_batch_size"])
            keys = jax.random.split(
                rng, (configuration["n_tasks"], configuration["epochs"], 2))

            optimizer, opt_state = configure_optimizer(
                configuration, eqx.filter(model, eqx.is_array))
            pbar = tqdm(range(configuration["n_tasks"]), desc="Tasks")
            train_samples = configuration["n_train_samples"] if "n_train_samples" in configuration else None
            test_samples = configuration["n_test_samples"] if "n_test_samples" in configuration else None
            for task in pbar:
                for epoch in range(configuration["epochs"]):
                    rng1 = keys[task, epoch, 0]
                    rng2 = keys[task, epoch, 1]
                    pbar.set_description(
                        f"Task {task+1}/{configuration['n_tasks']} - Epoch {epoch+1}/{configuration['epochs']}")
                    special_perm = permutations[task] if configuration["task"] == "PermutedMNIST" else None

                    dynamic_init_state, static_state = eqx.partition(
                        model, eqx.is_array)

                    @eqx.filter_jit
                    def train_fn(dynamic_model, opt_state, perm, rng, optimizer, images, labels, samples=None):
                        """ Training function for models. Receives one batch of data and updates the model by computing the
                        loss and its gradients.

                        Args:
                            carry: tuple containing the model, optimizer state, permutation and random key
                            data: tuple containing the images and labels
                            samples: number of samples for the model. If None, no sampling is performed.
                        """
                        # Compute the loss
                        if perm is not None:
                            images = images.reshape(
                                images.shape[0], -1)[:, perm].reshape(images.shape)
                        model = eqx.combine(dynamic_model, static_state)

                        (loss, predictions), grads = loss_fn(
                            model, images, labels, samples, rng)
                        # Update
                        dynamic_state, _ = eqx.partition(model, eqx.is_array)
                        dynamic_state, opt_state = optimizer.update(
                            dynamic_state, grads, opt_state)
                        return dynamic_state, opt_state, loss, predictions

                    @eqx.filter_jit
                    def scan_fn(carry, data):
                        dynamic_state, opt_state = carry
                        images, labels = data
                        dynamic_state, opt_state, loss, predictions = train_fn(
                            dynamic_state, opt_state, special_perm, rng1, optimizer, images, labels, train_samples)
                        return (dynamic_state, opt_state), (loss, predictions)

                    (dynamic_init_state, opt_state), (losses, predictions) = jax.lax.scan(
                        f=scan_fn, init=(dynamic_init_state, opt_state), xs=(task_train_images, task_train_labels))
                    model = eqx.combine(dynamic_init_state, static_state)

                    # Test the model and compute accuracy
                    if configuration["task"] == "PermutedMNIST":
                        accuracies = jnp.zeros(configuration["n_tasks"])
                        accuracies = jax.vmap(permute_and_test, in_axes=(None, None, 0, 0, None, None, None))(
                            model, permutations, test_train_images, test_train_labels, configuration["max_perm_parallel"], test_samples, rng2).mean(axis=0)
                    else:
                        accuracies = jnp.array([jax.vmap(compute_accuracy, in_axes=(None, 0, 0, None, None))(
                            model, test_train_images, test_train_labels, test_samples, rng2).mean()])
                    tqdm.write("=" * 20)
                    for i, acc in enumerate(accuracies):
                        tqdm.write(f"{acc.item()*100:.2f}%", end="\t" if i % 10 !=
                                   9 and i != len(accuracies) - 1 else "\n")
                    # Save accuracy as jax array
                    with open(os.path.join(DATA_PATH, f"task{task}-epoch{epoch}.npy"), "wb") as f:
                        jnp.save(f, accuracies)
        except (KeyboardInterrupt, SystemExit, Exception):
            print(traceback.format_exc())
            rmtree(SAVE_PATH)
