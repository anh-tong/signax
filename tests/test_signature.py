from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util
from numpy.random import default_rng

from signax import signature

rng = default_rng()

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def test_signature_1d_path():
    depth = 3
    length = 100
    path = rng.standard_normal((length, 1))
    signature(path, depth)


def test_signature_size():
    length, channels = 4, 3
    depth = 3
    corpus = jnp.ones((length, channels))
    sig = signature(corpus, depth=depth)
    assert len(sig) == depth
    for i, s_i in enumerate(sig):
        assert s_i.shape == (channels,) * (i + 1)


def test_signature_size_batch():
    batch, length, channels = 2, 4, 3
    depth = 3
    corpus = rng.standard_normal((batch, length, channels))
    sig = signature(corpus, depth=depth)
    assert len(sig) == batch
    for s in sig:
        assert len(s) == depth
        for i, s_i in enumerate(s):
            assert s_i.shape == (channels,) * (i + 1)


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
    tree = flatten_util.ravel_pytree(sig2)[0].reshape(batch, -1)
    assert jnp.all(sig == tree)


def test_invalid_path_shape():
    with pytest.raises(ValueError, match="Path must be of shape"):
        signature(jnp.ones((10, 10, 10, 10)), 2)
    with pytest.raises(ValueError, match="Path must be of shape"):
        signature(jnp.ones((10,)), 2)
