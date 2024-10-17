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


@jax.jit
def loss_fn(model, images, labels, perm=None):
    if perm is not None:
        images = images.reshape(
            images.shape[0], -1)[:, perm].reshape(images.shape)
    output = jax.vmap(model)(images)
    loss = -jnp.sum(jax.nn.log_softmax(output) * labels)
    return loss


@jax.jit
def train_fn(carry, data):
    model, opt_state, perm = carry
    images, labels = data
    # Compute the loss
    loss, grads = jax.value_and_grad(loss_fn)(model, images, labels, perm)
    # Update
    updates, opt_state = optimizer.update(grads, opt_state)
    model = optax.apply_updates(model, updates)
    return (model, opt_state, perm), loss


@jax.jit
def test_fn(model, images, labels):
    output = jax.vmap(model)(images)
    pred = jnp.argmax(output, axis=-1)
    labels = jnp.argmax(labels, axis=-1)
    accuracy = jnp.mean(pred == labels)
    return accuracy


@jax.jit
def test_batch_permutation_fn(model, image_batch, label_batch, batched_permutations):
    accuracies = jnp.zeros(len(batched_permutations))
    for i, perm in enumerate(batched_permutations):
        task_images = image_batch.reshape(
            image_batch.shape[0], -1)[:, perm].reshape(image_batch.shape)
        accuracies = accuracies.at[i].set(
            test_fn(model, task_images, label_batch))
    return accuracies


@jax.jit
def permute_and_test(model, permutations, image_batch, label_batch):
    """ We can't fit everything in one GPU when using too many permutations, so we must split permutations into batches """
    max_perm_parallel = 25
    batched_permutations = jnp.array(
        jnp.split(permutations, len(permutations) // max_perm_parallel)) if len(permutations) > max_perm_parallel else jnp.array([permutations])
    accuracies = jax.vmap(test_batch_permutation_fn, in_axes=(None, None, None, 0))(
        model, image_batch, label_batch, batched_permutations)
    # Flatten the results in the first two dimensions
    accuracies = accuracies.reshape(
        accuracies.shape[0] * accuracies.shape[1], -1)
    return accuracies


if __name__ == "__main__":

    N_ITERATIONS = 1
    configurations = [
        {
            "task": "PermutedMNIST",
            "n_tasks": 10,
            "epochs": 1,
            "train_batch_size": 1,
            "test_batch_size": 128,
            "seed": 1000 + i
        } for i in range(N_ITERATIONS)
    ]

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

        model = SmallNetwork(rng)
        # Define the optimizer
        optimizer = optax.sgd(learning_rate=0.001)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        # Define the loss function
        train_images = jnp.array(train.data, dtype=jnp.float32)
        train_labels = jnp.array(train.targets, dtype=jnp.int32)
        train_labels = jax.nn.one_hot(x=train_labels, num_classes=num_classes)
        test_images = jnp.array(test.data, dtype=jnp.float32)
        test_labels = jnp.array(test.targets, dtype=jnp.int32)
        test_labels = jax.nn.one_hot(x=test_labels, num_classes=num_classes)
        # Drop last elements if not divisible by batch_size to speed up training
        train_images = train_images[:len(
            train_images) - len(train_images) % configuration["train_batch_size"]]
        train_labels = train_labels[:len(
            train_labels) - len(train_labels) % configuration["train_batch_size"]]
        test_images = test_images[:len(
            test_images) - len(test_images) % configuration["test_batch_size"]]
        test_labels = test_labels[:len(
            test_labels) - len(test_labels) % configuration["test_batch_size"]]
        # Split the datasets into batches
        task_train_images = train_images.reshape(
            -1, configuration["train_batch_size"], *train_images.shape[1:])
        task_train_labels = train_labels.reshape(
            -1, configuration["train_batch_size"], num_classes)
        test_train_images = test_images.reshape(
            -1, configuration["test_batch_size"], *test_images.shape[1:])
        test_train_labels = test_labels.reshape(
            -1, configuration["test_batch_size"], num_classes)

        for task in tqdm(range(configuration["n_tasks"]), desc="Tasks"):
            tqdm.write(f"Task {task+1}/{configuration['n_tasks']}")
            # Train the model
            for epoch in tqdm(range(configuration["epochs"]), desc="Epochs"):
                special_perm = permutations[task] if configuration["task"] == "PermutedMNIST" else None
                (model, opt_state, _), loss = jax.lax.scan(f=train_fn, init=(
                    model, opt_state, special_perm), xs=(task_train_images, task_train_labels))
                # Test the model and compute accuracy
                if epoch % 1 == 0:
                    if configuration["task"] == "PermutedMNIST":
                        accuracies = jnp.zeros(configuration["n_tasks"])
                        accuracies = jax.vmap(permute_and_test, in_axes=(None, None, 0, 0))(
                            model, permutations, test_train_images, test_train_labels).mean(axis=0)
                    else:
                        accuracies = jnp.mean(jax.vmap(test_fn, in_axes=(None, 0, 0))(
                            model, test_train_images, test_train_labels))
                    if isinstance(accuracies, jnp.ndarray):
                        for i, acc in enumerate(accuracies):
                            # print ten task acc max per line using \t
                            tqdm.write(f"t{i}:{acc.item()*100:.2f}%", end="\t" if i % 10 !=
                                       9 and i != len(accuracies) - 1 else "\n")
                    else:
                        tqdm.write(f"{accuracies.item()*100:.2f}%")
                    tqdm.write(
                        f"Epoch {epoch+1}/{configuration['epochs']} - Loss: {loss.mean():.2f}")
                    # save accuracy as jax array
                    with open(os.path.join(DATA_PATH, f"task{task}-epoch{epoch}.npy"), "wb") as f:
                        jnp.save(f, accuracies)
