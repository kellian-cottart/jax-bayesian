from equinox.nn import Linear
from equinox import Module
from jax.random import split, normal
from jax.nn import relu
from jax.numpy import reshape, ndarray


class BayesianLinear(Module):
    """ Bayesian Linear Layer

    Weights have two parameters: mean and variance.
    They are sampled using the reparameterization trick w = mean + std * eps, where eps ~ N(0, 1)

    Args:
        in_features: int, number of input features
        out_features: int, number of output features
        use_bias: bool, whether to use bias
        key: PRNGKey, random key

    """
    weight: ndarray
    bias: ndarray

    def __init__(self, in_features, out_features, use_bias, key):
        self.weight.mu = normal(key, (out_features, in_features))
        self.weight.sigma = normal(key, (out_features, in_features))
        self.bias.mu = normal(key, (out_features,)) if use_bias else None
        self.bias.sigma = normal(key, (out_features,)) if use_bias else None

    def __call__(self, x, samples, key):
        """ Forward pass sampled `samples` times

        Args:
            x: , input tensor
            samples: int, number of samples to take
        """
        if samples == 0:
            return x @ self.weight.mu.T + self.bias.mu
        else:
            weight = self.weight.mu + self.weight.sigma * \
                normal(key, (samples, *self.weight.mu.shape))
            bias = self.bias.mu + self.bias.sigma * \
                normal(key, (samples, *self.bias.mu.shape))
            return x @ weight.T + bias


class SmallBayesianNetwork(Module):
    linear1: Linear
    linear2: Linear

    def __init__(self, key):
        key1, key2 = split(key)
        self.linear1 = BayesianLinear(784, 50, use_bias=False, key=key1)
        self.linear2 = BayesianLinear(50, 10, use_bias=False, key=key2)

    def __call__(self, x):
        x = reshape(x, (-1))
        x = relu(self.linear1(x))
        x = self.linear2(x)
        return x
