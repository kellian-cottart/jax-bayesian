from equinox import Module, field, filter_jit
from equinox import _misc
from jax.random import split, normal, uniform
from jax.nn import relu
from jax.numpy import reshape, ones
from jax import jit, vmap
from jax.numpy import dot, shape, broadcast_to, einsum
from typing import Literal, Optional, Union
from jaxtyping import PRNGKeyArray, Array
from math import sqrt
from functools import partial


class BayesianLinear(Module, strict=True):
    """Performs a linear transformation."""

    weight_mu: Array
    weight_sigma: Array
    bias_mu: Optional[Array]
    bias_sigma: Optional[Array]
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
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)
        """
        dtype = _misc.default_floating_dtype() if dtype is None else dtype
        wkey, bkey = split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 1 / sqrt(in_features_)
        wshape = (out_features_, in_features_)
        # initalizing mu as 1/sqrt(in_features) uniformly
        self.weight_mu = uniform(wkey, wshape, minval=-lim, maxval=lim)
        self.weight_sigma = ones(wshape, dtype)*sigma_init
        bshape = (out_features_,)
        self.bias_mu = uniform(bkey, bshape, minval=-lim,
                               maxval=lim) if use_bias else None
        self.bias_sigma = ones(bshape, dtype)*sigma_init if use_bias else None
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
        # we need to do
        # broadcast x if needed, weights and sigma
        if len(shape(x)) == 1:
            # repeat the content of x samples times
            x = broadcast_to(x, (samples, *shape(x)))
        mu = broadcast_to(self.weight_mu, (samples, *shape(self.weight_mu)))
        sigma = broadcast_to(
            self.weight_sigma, (samples, *shape(self.weight_sigma)))
        weights = mu + sigma * normal(key, shape(mu))

        output = vmap(lambda w, x: dot(w, x), in_axes=(0, 0))(weights, x)
        if self.use_bias:
            biases = vmap(sample_fn, in_axes=(0, 0, 0))(
                self.bias_mu, self.bias_sigma, split(key, samples))
            output = output + biases
        return output


def sample_fn(mu, sigma, key):
    return mu + sigma * normal(key, shape(mu))


class SmallBayesianNetwork(Module):
    linear1: BayesianLinear
    linear2: BayesianLinear

    def __init__(self, key, sigma_init):
        key1, key2 = split(key)
        self.linear1 = BayesianLinear(
            784, 50, use_bias=False, key=key1, sigma_init=sigma_init)
        self.linear2 = BayesianLinear(
            50, 10, use_bias=False, key=key2, sigma_init=sigma_init)

    def __call__(self, x, samples, key):
        x = reshape(x, (-1))
        x = relu(self.linear1(x, samples=samples, key=key))
        x = self.linear2(x, samples=samples, key=key)
        return x
