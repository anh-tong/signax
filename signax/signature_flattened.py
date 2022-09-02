from functools import partial
from typing import List

import jax
import jax.numpy as jnp

from .signature import signature as _signature, signature_combine as _signature_combine
from .utils import flatten, unsqueeze_signature


@partial(jax.jit, static_argnames="depth")
def signature(path: jnp.ndarray, depth: int) -> jnp.ndarray:
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
    # TODO: implement logsignature flattenning method
    raise NotImplementedError()


def signature_to_logsignature(
        signature: List[jnp.ndarray],
) -> List[jnp.ndarray]:
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
    # TODO: implement logsignature flattenning method
    raise NotImplementedError()


def signature_combine(signature1: jnp.ndarray, signature2: jnp.ndarray, dim: int, depth: int) -> jnp.ndarray:
    sig1 = unsqueeze_signature(signature1, dim, depth)
    sig2 = unsqueeze_signature(signature2, dim, depth)
    return flatten(_signature_combine(sig1, sig2))
