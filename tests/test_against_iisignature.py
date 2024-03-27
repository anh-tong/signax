from __future__ import annotations

import iisignature
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.random import default_rng

from signax import logsignature, signature

rng = default_rng()

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("depth", "length", "dim", "stream"),
    [(1, 2, 2, False), (3, 100, 3, False), (2, 10, 2, True)],
)
def test_signature(depth, length, dim, stream):
    batch_size = 10
    path = rng.standard_normal((batch_size, length, dim))
    jax_signature = signature(path, depth=depth, stream=stream)
    iis_signature = (
        iisignature.sig(np.asarray(path), depth)
        if not stream
        else iisignature.sig(np.asarray(path), depth, 2)
    )
    iis_signature = jnp.asarray(iis_signature)
    assert jnp.allclose(jax_signature, iis_signature)


@pytest.mark.parametrize(
    ("depth", "length", "dim", "stream"), [(1, 2, 2, False), (3, 3, 5, False)]
)
def test_logsignature(depth, length, dim, stream):
    batch_size = 10
    path = rng.standard_normal((batch_size, length, dim))
    jax_logsignature = logsignature(path, depth=depth, stream=stream, flatten=True)
    s = iisignature.prepare(dim, depth, "O")
    iis_logsignature = iisignature.logsig(np.asarray(path), s)
    iis_logsignature = jnp.asarray(iis_logsignature)
    assert jnp.allclose(jax_logsignature, iis_logsignature, atol=5e-1, rtol=5e-1)
