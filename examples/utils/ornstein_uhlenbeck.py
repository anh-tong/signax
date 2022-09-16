from functools import partial

import jax
from jax import numpy as jnp
from jax import random as jrandom


def get_ou_signal(
    random_key: jrandom.PRNGKey, num_samples: int = 1000, n_points: int = 100
):
    """Generate examples of an Ornstein-Uhlenbeck process."""

    keys = jax.random.split(random_key, num_samples)
    return jax.vmap(lambda key: gen_ou_data(key, n_points))(keys)


def gen_ou_data(random_key: jrandom.PRNGKey, n_points: int = 100):
    """Generate an Ornstein-Uhlenbeck process."""

    timeline = jnp.linspace(0, 1, n_points)
    values = ornstein_uhlenbeck_process(random_key, steps=timeline).flatten()
    path = jnp.c_[timeline, values]
    return path.T


@partial(jax.jit, static_argnames=["dt", "mu", "tau", "sigma"])
def ornstein_uhlenbeck_process(
    random_key,
    steps: jnp.ndarray,
    dt: float = 0.1,
    mu: float = 0,
    tau: float = 2,
    sigma: float = 1,
):
    num_steps = len(steps)
    noise = jax.random.normal(random_key, (num_steps,))

    def ou_step(t, val):
        dx = -(val[t - 1] - mu) / tau * dt + sigma * jnp.sqrt(2 / tau) * noise[
            t
        ] * jnp.sqrt(dt)
        return val.at[t].set(val[t - 1] + dx)

    return jax.lax.fori_loop(1, num_steps + 1, ou_step, steps)
