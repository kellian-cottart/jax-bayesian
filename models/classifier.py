# Define a simple model using Equinox
from equinox.nn import Linear
from equinox import Module
from jax.random import split
from jax.nn import relu
from jax.numpy import ravel


class SmallNetwork(Module):
    layers: list[Linear]

    def __init__(self, key):

        key1, key2 = split(key)
        self.layers = [
            ravel,
            Linear(784, 50, use_bias=False, key=key1),
            relu,
            Linear(50, 10, use_bias=False, key=key2)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
