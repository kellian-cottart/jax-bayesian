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
import argparse
import json
# argparse allows to load a configuration from a file
CONFIGURATION_LOADING_FOLDER = "configurations"
# first argument is name of config file
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Configuration file name",
                    type=str)
parser.add_argument(
    "--n_iterations", help="Number of iterations to run the config file for", type=int, default=1)
parser.add_argument(
    "-v", "--verbose", help="Whether to display the pbar or not", action="store_true")
args = parser.parse_args()
CONFIG_FILE = json.load(
    open(os.path.join(CONFIGURATION_LOADING_FOLDER, args.config)))
N_ITERATIONS = args.n_iterations
VERBOSE = args.verbose

if __name__ == "__main__":

    # Configurations is an array with N_ITERATIONS times the same config except with seed +=1
    configurations = [{**CONFIG_FILE, "seed": CONFIG_FILE["seed"] + k}
                      for k in range(N_ITERATIONS)]
    # Convert all fields to lowercase if string
    for config in configurations:
        for k, v in config.items():
            if isinstance(v, str):
                config[k] = v.lower()
    # Create a timestamp
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S-")
    MAIN_FOLDER = "results"
    os.makedirs(MAIN_FOLDER, exist_ok=True)
    for k, configuration in enumerate(configurations):
        FOLDER = f"{TIMESTAMP}{configuration['task']}-t={configuration['n_tasks']}-e={configuration['epochs']}-opt={configuration['optimizer']}"
        SAVE_PATH = os.path.join(MAIN_FOLDER, FOLDER)
        CONFIGURATION_PATH = os.path.join(SAVE_PATH, f"config{k}")
        DATA_PATH = os.path.join(CONFIGURATION_PATH, "accuracy")
        WEIGHTS_PATH = os.path.join(CONFIGURATION_PATH, "weights")
        UNCERTAINTY_PATH = os.path.join(CONFIGURATION_PATH, "uncertainty")
        for path in [SAVE_PATH, CONFIGURATION_PATH, DATA_PATH, WEIGHTS_PATH, UNCERTAINTY_PATH]:
            os.makedirs(path, exist_ok=True)
        # save config
        with open(CONFIGURATION_PATH + "/config.json", "w") as f:
            json.dump(configuration, f, indent=4)
        try:
            # Load the MNIST dataset
            loader = GPULoading()
            train, test, shape, num_classes = loader.task_selection(
                configuration["task"])
            # Initialize the model
            rng = jax.random.PRNGKey(configuration["seed"])
            # Permutations
            perm_keys, rng = jax.random.split(rng, 2)
            perm_keys = jax.random.split(perm_keys, configuration["n_tasks"])
            permutations = None
            if configuration["task"] == "permutedmnist":
                permutations = jnp.array(
                    [jax.random.permutation(key, jnp.array(shape).prod()) for key in perm_keys])
            model = configure_networks(configuration, rng)
            # Prepare datasets
            task_train_images, task_train_labels = prepare_data(
                train.data, train.targets, configuration["train_batch_size"], num_classes)
            task_test_images, task_test_labels = prepare_data(
                test.data, test.targets, configuration["test_batch_size"], num_classes)
            optimizer, opt_state = configure_optimizer(
                configuration, eqx.filter(model, eqx.is_array))
            train_samples = configuration["n_train_samples"] if "n_train_samples" in configuration else None
            test_samples = configuration["n_test_samples"] if "n_test_samples" in configuration else None
            ood_key, rng = jax.random.split(rng)
            ood_test_images, ood_test_labels = prepare_data(
                test.data, test.targets, configuration["test_batch_size"], num_classes)
            ood_permutation = jnp.array([jax.random.permutation(
                ood_key, jnp.array(shape).prod())])

            # GENERATING A HUGE ARRAY OF KEYS, ASSURING THAT THE KEYS ARE UNIQUE
            trkey, tekey, ookey, rng = jax.random.split(rng, 4)
            training_core_keys = jax.random.split(
                trkey, (configuration["n_tasks"], configuration["epochs"]))
            testing_core_keys = jax.random.split(
                tekey, (configuration["n_tasks"], configuration["epochs"]))
            ood_core_keys = jax.random.split(
                ookey, (configuration["n_tasks"], configuration["epochs"]))
            if VERBOSE:
                pbar = tqdm(range(configuration["n_tasks"]), desc="Tasks")
            else:
                pbar = range(configuration["n_tasks"])
            for i, task in enumerate(pbar):
                for epoch in range(configuration["epochs"]):
                    if VERBOSE:
                        pbar.set_description(
                            f"Task {task+1}/{configuration['n_tasks']} - Epoch {epoch+1}/{configuration['epochs']}")
                    train_ck = training_core_keys[task, epoch]
                    test_ck = testing_core_keys[task, epoch]
                    special_perm = permutations[task] if permutations is not None else None
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
                    # Save weights histogram
                    histogramWeights(eqx.filter(
                        model, eqx.is_array), WEIGHTS_PATH, task, epoch)
                    # Compute uncertainty
                    ood_k = ood_core_keys[task, epoch]
                    ood_accuracies, ood_predictions = test_fn(
                        model=model,
                        images=ood_test_images,
                        labels=ood_test_labels,
                        rng=ood_k,
                        max_perm_parallel=configuration["max_perm_parallel"],
                        permutations=ood_permutation,
                        test_samples=test_samples
                    )
                    
                    aleatoric_uncertainty, epistemic_uncertainty, aleatoric_uncertainty_ood, epistemic_uncertainty_ood, auc = computeUncertainty(
                        predictions=predictions,
                        ood_predictions=ood_predictions
                    )
                    with open(os.path.join(UNCERTAINTY_PATH, f"aleatoric-task={task}-epoch={epoch}.npy"), "wb") as f:
                        jnp.save(f, aleatoric_uncertainty)
                    with open(os.path.join(UNCERTAINTY_PATH, f"ood-aleatoric-task={task}-epoch={epoch}.npy"), "wb") as f:
                        jnp.save(f, aleatoric_uncertainty_ood)
                    with open(os.path.join(UNCERTAINTY_PATH, f"epistemic-task={task}-epoch={epoch}.npy"), "wb") as f:
                        jnp.save(f, epistemic_uncertainty)
                    with open(os.path.join(UNCERTAINTY_PATH, f"ood-epistemic-task={task}-epoch={epoch}.npy"), "wb") as f:
                        jnp.save(f, epistemic_uncertainty_ood)
                    with open(os.path.join(UNCERTAINTY_PATH, f"roc-auc-task={task}-epoch={epoch}.npy"), "wb") as f:
                        jnp.save(f, auc)
                    with open(os.path.join(DATA_PATH, f"task={task}-epoch={epoch}.npy"), "wb") as f:
                        jnp.save(f, accuracies)
        except (KeyboardInterrupt, SystemExit, Exception):
            print(traceback.format_exc())
            rmtree(SAVE_PATH)
