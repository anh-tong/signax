from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util
from numpy.random import default_rng

from signax import logsignature, signature

rng = default_rng()

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def test_signature_1d_path():
    depth = 3
    length = 100
    path = rng.standard_normal((length, 1))
    signature(path, depth)


@pytest.mark.parametrize("flatten", [True, False])
def test_signature_size(flatten):
    length, channels = 4, 3
    depth = 3
    corpus = jnp.ones((length, channels))
    sig = signature(corpus, depth=depth, flatten=flatten)
    if flatten:
        assert sig.shape == ((channels ** (depth + 1) - 1) / (channels - 1) - 1,)
    else:
        assert len(sig) == depth
        for i, s in enumerate(sig):
            assert s.shape == (channels,) * (i + 1)


@pytest.mark.parametrize("flatten", [True, False])
def test_signature_size_batch(flatten):
    batch, length, channels = 2, 4, 3
    depth = 3
    corpus = rng.standard_normal((batch, length, channels))
    sig = signature(corpus, depth=depth, flatten=flatten)
    if flatten:
        # shape of sig is a single tensor (batch, channels + channels ** 2 + channels **3)
        assert len(sig) == batch
        assert sig.shape == (batch, (channels ** (depth + 1) - 1) / (channels - 1) - 1)
    else:
        # sig is a list of tensors
        #   (batch, channels), (batch, channels, channels), (batch, channels, channels)
        assert len(sig) == depth
        for i, s in enumerate(sig):
            assert s.shape == (batch,) + (channels,) * (i + 1)


def test_signature_flatten():
    length, channels = 4, 3
    corpus = rng.standard_normal((length, channels))
    sig = signature(corpus, depth=6, flatten=True)
    sig2 = signature(corpus, depth=6, flatten=False)
    tree = flatten_util.ravel_pytree(sig2)[0]
    assert jnp.all(sig == tree)


def test_signature_flatten_batch():
    batch, length, channels = 2, 4, 3
    corpus = rng.standard_normal((batch, length, channels))
    sig = signature(corpus, depth=6, flatten=True)
    sig2 = signature(corpus, depth=6, flatten=False)
    tree = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])(sig2).reshape(batch, -1)
    assert jnp.all(sig == tree)


def test_invalid_path_shape():
    with pytest.raises(ValueError, match="Path must be of shape"):
        signature(jnp.ones((10, 10, 10, 10)), 2)
    with pytest.raises(ValueError, match="Path must be of shape"):
        signature(jnp.ones((10,)), 2)

    with pytest.raises(ValueError, match="Path must be of shape"):
        logsignature(jnp.ones((10, 10, 10, 10)), 2)
    with pytest.raises(ValueError, match="Path must be of shape"):
        logsignature(jnp.ones((10,)), 2)
