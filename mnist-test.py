import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm
from dataloader import *
EPOCHS = 100
TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 128
SEED = 1000

# Load the MNIST dataset
loader = GPULoading()
train, test, shape, classes = loader.task_selection("mnist")

# Define a simple model using Equinox


class SimpleModel(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, key):
        key1, key2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(784, 50, use_bias=False, key=key1)
        self.linear2 = eqx.nn.Linear(50, 10, use_bias=False, key=key2)

    def __call__(self, x):
        x = jnp.reshape(x, (-1))
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)
        return x


@jax.jit
def loss_fn(model, images, labels):
    output = jax.vmap(model)(images)
    loss = -jnp.sum(jax.nn.log_softmax(output) * labels)
    return loss


@jax.jit
def train_fn(carry, data):
    model, opt_state = carry
    images, labels = data
    # Compute the loss
    loss, grads = jax.value_and_grad(loss_fn)(model, images, labels)
    # Update
    updates, opt_state = optimizer.update(grads, opt_state)
    model = optax.apply_updates(model, updates)
    return (model, opt_state), loss


def test_fn(model, images, labels):
    output = jax.vmap(model)(images)
    pred = jnp.argmax(output, axis=-1)
    labels = jnp.argmax(labels, axis=-1)
    accuracy = jnp.mean(pred == labels)
    return accuracy


if __name__ == "__main__":
    # Initialize the model
    rng = jax.random.PRNGKey(SEED)
    model = SimpleModel(rng)
    # Define the optimizer
    optimizer = optax.sgd(learning_rate=0.001)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    # static number of class
    num_classes = train.targets.max() + 1
    # Define the loss function
    train_images = jnp.array(train.data, dtype=jnp.float32)
    train_labels = jnp.array(train.targets, dtype=jnp.int32)
    train_labels = jax.nn.one_hot(x=train_labels, num_classes=num_classes)
    test_images = jnp.array(test.data, dtype=jnp.float32)
    test_labels = jnp.array(test.targets, dtype=jnp.int32)
    test_labels = jax.nn.one_hot(x=test_labels, num_classes=num_classes)
    # Drop last elements if not divisible by BATCH_SIZE
    train_images = train_images[:len(
        train_images) - len(train_images) % TRAIN_BATCH_SIZE]
    train_labels = train_labels[:len(
        train_labels) - len(train_labels) % TRAIN_BATCH_SIZE]
    test_images = test_images[:len(
        test_images) - len(test_images) % TEST_BATCH_SIZE]
    test_labels = test_labels[:len(
        test_labels) - len(test_labels) % TEST_BATCH_SIZE]
    # Split into batches
    task_train_images = jnp.array(
        jnp.split(train_images, len(train_images) // TRAIN_BATCH_SIZE))
    task_train_labels = jnp.array(
        jnp.split(train_labels, len(train_labels) // TRAIN_BATCH_SIZE))
    test_train_images = jnp.array(
        jnp.split(test_images, len(test_images) // TEST_BATCH_SIZE))
    test_train_labels = jnp.array(
        jnp.split(test_labels, len(test_labels) // TEST_BATCH_SIZE))
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        (model, opt_state), loss = jax.lax.scan(f=train_fn, init=(
            model, opt_state), xs=(task_train_images, task_train_labels))
        # Test the model and compute accuracy
        if epoch % 5 == 0:
            accuracy = jnp.mean(jax.vmap(test_fn, in_axes=(None, 0, 0))(
                model, test_train_images, test_train_labels))
            tqdm.write(
                f"Epoch {epoch}, Loss: {loss.mean():.3f}, Accuracy: {accuracy*100:.2f}%")
