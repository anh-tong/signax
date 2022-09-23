import jax
import jax.numpy as jnp
from jax.random import PRNGKey


def get_bm_noise(
    random_key: PRNGKey,
    n_points: int,
    num_samples: int = 1000,
):
    """Generate examples of a Brownian motion."""
    random_keys = jax.random.split(random_key, num_samples)
    paths = jnp.array(
        [gen_bm_noise(random_keys[i], n_points) for i in range(num_samples)]
    )
    return paths


def gen_bm_noise(
    random_key: PRNGKey,
    n_points: int = 100,
):
    dt = 1 / jnp.sqrt(n_points)
    nd = jax.random.normal(random_key, (n_points - 1,)).cumsum()
    bm = dt * jnp.r_[0.0, nd]
    timeline = jnp.linspace(0, 1, n_points)
    return jnp.c_[timeline, bm].T
