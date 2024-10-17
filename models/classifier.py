# Define a simple model using Equinox
from equinox.nn import Linear
from equinox import Module
from jax.random import split
from jax.nn import relu
from jax.numpy import reshape


class SmallNetwork(Module):
    linear1: Linear
    linear2: Linear

    def __init__(self, key):
        key1, key2 = split(key)
        self.linear1 = Linear(784, 50, use_bias=False, key=key1)
        self.linear2 = Linear(50, 10, use_bias=False, key=key2)

    def __call__(self, x):
        x = reshape(x, (-1))
        x = relu(self.linear1(x))
        x = self.linear2(x)
        return x
