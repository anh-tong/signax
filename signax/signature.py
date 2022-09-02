from functools import partial
from typing import List

import jax
import jax.numpy as jnp

from .tensor_ops import log, mult, mult_fused_restricted_exp, restricted_exp
from .utils import compress, lyndon_words


@partial(jax.jit, static_argnames="depth")
def signature(path: jnp.ndarray, depth: int) -> List[jnp.ndarray]:
    """
    Compute the signature of a path

    Args:
        path: size (length, dim)
        depth: signature is truncated at this depth
    Returns:
        A list of `jnp.ndarray` in a form
        [(dim, ), (dim, dim), (dim, dim, dim), ...]
    """

    path_increments = jnp.diff(path, axis=0)
    exp_term = restricted_exp(path_increments[0], depth=depth)

    def _body(i, val):
        ret = mult_fused_restricted_exp(path_increments[i], val)
        ret = [x.squeeze() for x in ret]
        return ret

    exp_term = jax.lax.fori_loop(
        lower=1,
        upper=path_increments.shape[0],
        body_fun=_body,
        init_val=exp_term,
    )

    return exp_term


def logsignature(path, depth):
    pass


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
    depth = len(signature)
    dim = signature[0].shape[0]

    indices = lyndon_words(depth, dim)

    # compute Lyndon words given `depth` and `dim`
    expanded_logsignature = log(signature)

    # compress using the information of Lyndon words
    log_sig = compress(expanded_logsignature, indices)
    return log_sig


def signature_combine(signatures):
    combination = signatures[0]

    for i in range(1, len(signatures)):
        combination = mult(combination, signatures[i])

    return combination
