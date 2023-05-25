from __future__ import annotations

from functools import partial

import jax

from .signature import logsignature as _logsignature
from .signature import signature as _signature
from .signature import signature_combine as _signature_combine
from .signature import signature_to_logsignature as _signature_to_logsignature
from .utils import flatten, unravel_signature


@partial(jax.jit, static_argnames="depth")
def signature(path: jax.Array, depth: int) -> jax.Array:
    """
    Compute the signature of a path

    Args:
        path: size (length, dim)
        depth: signature is truncated at this depth
    Returns:
        A list of `jnp.ndarray` in a form
        (dim +  dim * dim + dim * dim * dim, ...)
    """

    return flatten(_signature(path, depth))


def logsignature(path, depth):
    return flatten(_logsignature(path, depth))


def signature_to_logsignature(signature: jax.Array, dim: int, depth: int) -> jax.Array:
    """
    Compute logsignature from signature

    This function ONLY supports the compression of expanded logsignature
    in Lyndon basis

    Args:
        signature: A list of `jnp.ndarray`
        The list is like [(dim,), (dim, dim), (dim, dim, dim),...]
    Returns:
        A list of `jnp.ndarray` [(dim1, ), (dim2,), (dim3,), ...]
        `dim1`, `dim2`, `dim3` are determined of how to
    """

    unraveled = unravel_signature(signature, dim, depth)
    return flatten(_signature_to_logsignature(unraveled))


def signature_combine(
    signature1: jax.Array, signature2: jax.Array, dim: int, depth: int
) -> jax.Array:
    sig1 = unravel_signature(signature1, dim, depth)
    sig2 = unravel_signature(signature2, dim, depth)
    return flatten(_signature_combine(sig1, sig2))
