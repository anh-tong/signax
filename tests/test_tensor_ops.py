from __future__ import annotations

import jax
import jax.numpy as jnp

# need to install torch and signatory for testing
from numpy.random import default_rng

from signax.tensor_ops import (
    addcmul,
    otimes,
)

rng = default_rng()


jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def test_otimes():
    # 1D x 1D
    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])

    true_output = jnp.array([[3.0, 4.0], [6.0, 8.0]])
    assert jnp.allclose(true_output, otimes(x, y), rtol=1e-3, atol=1e-5)

    # 1D x 2D
    x = jnp.array([1.0, 2.0])
    y = jnp.array([[3.0, 4.0], [5.0, 6.0]])
    true_output = jnp.array(
        [
            [[3.0, 4.0], [5.0, 6.0]],
            [[6.0, 8.0], [10.0, 12.0]],
        ]
    )
    assert jnp.allclose(true_output, otimes(x, y), rtol=1e-3, atol=1e-5)

    # 2D x 1D
    true_output = jnp.array(
        [
            [[3.0, 6.0], [4.0, 8.0]],
            [[5.0, 10.0], [6.0, 12.0]],
        ]
    )

    assert jnp.allclose(true_output, otimes(y, x), rtol=1e-3, atol=1e-5)


def test_addcmul():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([1.0, 2.0])
    z = jnp.array([3.0, 4.0])
    true_output = jnp.array([[4.0, 6.0], [9.0, 12.0]])
    assert jnp.allclose(true_output, addcmul(x, y, z), rtol=1e-3, atol=1e-5)
