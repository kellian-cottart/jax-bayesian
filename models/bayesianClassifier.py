from equinox import Module, field
from equinox import _misc
from jax.random import split, normal, uniform
from jax.nn import relu
from jax.numpy import shape, broadcast_to, einsum, dot, expand_dims,  reshape, ones, ravel
from typing import Literal, Union
from jaxtyping import PRNGKeyArray, Array
from math import sqrt
from models.gaussianParameter import *


class BayesianLinear(Module, strict=True):
    """Performs a linear transformation."""

    weight: dict[str, Array]
    bias: dict[str, Array]
    in_features: Union[int, Literal["scalar"]] = field(static=True)
    out_features: Union[int, Literal["scalar"]] = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
        sigma_init: float = 0.1,
    ):
        """ Initialises the Bayesian Linear Layer

        Args:
            in_features: The input size. The input to the layer should be a vector of
                shape `(in_features,)`
            out_features: The output size. The output from the layer will be a vector
                of shape `(out_features,)`.
            use_bias: Whether to add on a bias as well.
            dtype: The dtype to use for the weight and the bias in this layer.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
                on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for GaussianParameter
                initialisation. (Keyword only argument.)
        """
        dtype = _misc.default_floating_dtype() if dtype is None else dtype
        wkey, bkey = split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 2 / sqrt(in_features_)
        wshape = (out_features_, in_features_)
        bshape = (out_features_,)

        # Replace the placeholder with the following code
        self.weight = GaussianParameter(
            mu=uniform(wkey, wshape, minval=-lim, maxval=lim),
            sigma=ones(wshape, dtype) * sigma_init,
        )
        self.bias = GaussianParameter(
            mu=uniform(bkey, bshape, minval=-lim,
                       maxval=lim) if use_bias else None,
            sigma=ones(bshape, dtype) * sigma_init if use_bias else None,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(self, x: Array, samples: int, *, key: PRNGKeyArray) -> Array:
        """ Call function for bayesian linear layer
        `samples` forward passes using weights reparametrization w = mu + sigma * epsilon, epsilon ~ N(0, 1)

        Args:
            x: input tensor
            samples: number of samples
            rng: random key
        """
        if samples < 0:
            raise ValueError(
                "Number of samples must be a positive integer. Specify 0 for a deterministic forward pass.")
        if samples == 0:
            # dot product between input and weights
            output = dot(self.weight.mu, x)
            if self.use_bias:
                output = output + self.bias.mu
            return output
        if len(shape(x)) == 1:
            x = broadcast_to(x, (samples, *shape(x)))
        wkey, bkey = split(key, 2)
        weights = self.weight.mu + self.weight.sigma * \
            normal(wkey, (samples, *shape(self.weight.mu)))
        output = einsum("sp, sop -> so", x, weights)
        if self.use_bias:
            biases = self.bias.mu + self.bias.sigma * \
                normal(bkey, (samples, *shape(self.bias.mu)))
            output = output + biases
        return output


class SmallBayesianNetwork(Module):
    layers: list[BayesianLinear]

    def __init__(self, key, sigma_init):
        key1, key2 = split(key)
        self.layers = [ravel,
                       BayesianLinear(
                           784, 50, use_bias=False, key=key1, sigma_init=sigma_init),
                       relu,
                       BayesianLinear(
                           50, 10, use_bias=False, key=key2, sigma_init=sigma_init)]

    def __call__(self, x, samples, key):
        keys = split(key, len(self.layers))
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BayesianLinear):
                x = layer(x, samples=samples, key=keys[i])
            else:   # activation function
                x = layer(x)
        return x if samples != 0 else expand_dims(x, 0)
