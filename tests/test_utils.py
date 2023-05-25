import jax.numpy as jnp
from signax.utils import index_select


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
