import jax
import jax.experimental
import jax.numpy as jnp
import equinox as eqx
from dataloader import *
from models import *
from datetime import datetime
import os
import json
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
                "clamp_grad": 0
            },
            "task": "PermutedMNIST",
            "n_train_samples": 8,
            "n_test_samples": 8,
            "n_tasks": 10,
            "epochs": 1,
            "train_batch_size": 1,
            "test_batch_size": 100,
            "max_perm_parallel": 1,
            "seed": 1000+i
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
            task_test_images, task_test_labels = prepare_data(
                test.data, test.targets, configuration["test_batch_size"])
            optimizer, opt_state = configure_optimizer(
                configuration, eqx.filter(model, eqx.is_array))
            pbar = tqdm(range(configuration["n_tasks"]), desc="Tasks")
            train_samples = configuration["n_train_samples"] if "n_train_samples" in configuration else None
            test_samples = configuration["n_test_samples"] if "n_test_samples" in configuration else None

            # GENERATING A HUGE ARRAY OF KEYS, ASSURING THAT THE KEYS ARE UNIQUE
            training_core_keys = jax.random.split(
                rng, (configuration["n_tasks"], configuration["epochs"]))
            testing_core_keys = jax.random.split(
                rng, (configuration["n_tasks"], configuration["epochs"]))
            for task in pbar:
                for epoch in range(configuration["epochs"]):
                    train_ck = training_core_keys[task, epoch]
                    test_ck = testing_core_keys[task, epoch]
                    pbar.set_description(
                        f"Task {task+1}/{configuration['n_tasks']} - Epoch {epoch+1}/{configuration['epochs']}")
                    special_perm = permutations[task] if configuration["task"] == "PermutedMNIST" else None
                    # Split the model into dynamic and static parts
                    dynamic_init_state, static_state = eqx.partition(
                        model, eqx.is_array)

                    @ eqx.filter_jit
                    def train_fn(dynamic_model, opt_state, perm, keys, optimizer, images, labels, samples=None):
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
                        dynamic_state, _ = eqx.partition(model, eqx.is_array)
                        dynamic_state, opt_state = optimizer.update(
                            dynamic_state, grads, opt_state)
                        return dynamic_state, opt_state, loss

                    @ eqx.filter_jit
                    def scan_fn(carry, data):
                        dynamic_state, opt_state = carry
                        images, labels, key = data
                        # Train the model
                        dynamic_state, opt_state, loss = train_fn(
                            dynamic_state, opt_state, special_perm, key, optimizer, images, labels, train_samples)
                        return (dynamic_state, opt_state), loss
                    train_ck = jax.random.split(
                        train_ck, task_train_images.shape[0])
                    # Use jax.lax.scan to iterate over the batches
                    (dynamic_init_state, opt_state), losses = jax.lax.scan(
                        f=scan_fn, init=(dynamic_init_state, opt_state), xs=(task_train_images, task_train_labels, train_ck))
                    # Combine the dynamic and static parts of the model to recover the activation functions
                    model = eqx.combine(dynamic_init_state, static_state)
                    accuracies, predictions = test_fn(
                        model=model,
                        images=task_test_images,
                        labels=task_test_labels,
                        rng=test_ck,
                        max_perm_parallel=configuration["max_perm_parallel"],
                        permutations=permutations,
                        test_samples=test_samples
                    )
                    tqdm.write("=" * 20)
                    for i, acc in enumerate(accuracies):
                        tqdm.write(f"{acc.item()*100:.2f}%", end="\t" if i % 10 !=
                                   9 and i != len(accuracies) - 1 else "\n")
                    # Save weights histogram
                    # histogramWeights(eqx.filter(
                    #     model, eqx.is_array), WEIGHTS_PATH, task, epoch)
                    # Save accuracy as jax array
                    with open(os.path.join(DATA_PATH, f"task{task}-epoch{epoch}.npy"), "wb") as f:
                        jnp.save(f, accuracies)
        except (KeyboardInterrupt, SystemExit, Exception):
            print(traceback.format_exc())
            rmtree(SAVE_PATH)
