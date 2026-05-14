from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from numpy.random import default_rng

from signax import signature
from signax.utils import index_select, term_at, unravel_signature

jax.config.update("jax_platform_name", "cpu")

rng = default_rng(0)


def test_index_select():
    # first test
    dim = 4
    a = jnp.arange(0, dim**2).reshape((dim, dim))
    indices = jnp.array([[0, 0], [0, 1], [3, 3]])
    true_output = jnp.array([0, 1, 15])
    assert jnp.allclose(true_output, index_select(a, indices))

    # second test
    dim = 3
    a = jnp.arange(0, dim**3).reshape((dim, dim, dim))
    indices = jnp.array([[0, 0, 0], [0, 1, 1], [2, 2, 2]])
    true_output = jnp.array([0, 4, 26])
    assert jnp.allclose(index_select(a, indices), true_output)


@pytest.mark.parametrize(("dim", "depth"), [(3, 3), (2, 4), (4, 2)])
def test_term_at(dim, depth):
    length = 10
    path = rng.standard_normal((length, dim))
    sig = signature(path, depth, flatten=True)
    terms = unravel_signature(sig, dim, depth)
    for i in range(depth):
        assert jnp.allclose(term_at(sig, dim, i), terms[i])
