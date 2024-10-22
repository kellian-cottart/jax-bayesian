
import jax.numpy as jnp
import jax
from models import *
from optimizers import *


def prepare_data(data, targets, batch_size, num_classes):
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
        "bgd": bgd
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
