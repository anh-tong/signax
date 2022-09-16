import jax
import jax.numpy as jnp
from jax._src.random import PRNGKey


def get_bm_noise(num_samples: int = 1000, **kwargs):
    """Generate examples of a Brownian motion."""

    paths = jnp.array([gen_bm_noise(**kwargs) for _ in range(num_samples)])
    y = jnp.zeros_like(paths[:, 0, :-1])
    return paths, y


def gen_bm_noise(random_key: PRNGKey, n_points: int = 100):
    dt = 1 / jnp.sqrt(n_points)
    nd = jax.random.normal(random_key, (n_points - 1,)).cumsum()
    bm = dt * jnp.r_[0.0, nd]
    timeline = jnp.linspace(0, 1, n_points)
    return jnp.c_[timeline, bm].T
