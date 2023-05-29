from __future__ import annotations

import jax
from jax import numpy as jnp


def scalar_orders(
    dim: int,
    order: int,
):
    """The order of the scalar basis elements
    as one moves along the signature."""

    orders = [jnp.array([0])]
    cum_dim = dim
    for idx in range(order):
        orders.append(jnp.ones((cum_dim,)) * (idx + 1))
        cum_dim *= dim

    return jnp.concatenate(orders)


def psi(x, M=4, a=1):
    """Psi function, as defined in the following paper:

    Chevyrev, I. and Oberhauser, H., 2018. Signature moments to
    characterize laws of stochastic processes. arXiv preprint arXiv:1810.10971.

    """

    return jax.lax.cond(
        x > M,
        lambda: x,
        lambda: M + M ** (1 + a) * (M ** (-a) - x ** (-a)) / a,
    )


def normalize_signature(x, order):
    """Normalise signature, following the paper

    Chevyrev, I. and Oberhauser, H., 2018. Signature moments to
    characterize laws of stochastic processes. arXiv preprint arXiv:1810.10971.

    """

    x = jnp.concatenate([jnp.array([1.0]), x])

    a = x**2
    a.at[0].set(a[0] - psi(jnp.linalg.norm(x)))

    moments = jnp.ones_like(x)
    polx0 = jnp.dot(a, moments)

    d_moments = jnp.arange(0, 2 * len(x), 2)
    d_polx0 = jnp.dot(a, d_moments)

    x1 = 1 - polx0 / d_polx0
    x1 = jax.lax.cond(
        x1 < 0.2,
        lambda: 1.0,
        lambda: x1,
    )

    lambda_ = jnp.array([x1**t for t in scalar_orders(2, order)])

    return lambda_ * x
