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
        return ret

    exp_term = jax.lax.fori_loop(
        lower=1,
        upper=path_increments.shape[0],
        body_fun=_body,
        init_val=exp_term,
    )

    return exp_term


def signature_batch(path: jnp.ndarray, depth: int, n_chunks: int):
    length, dim = path.shape
    chunk_length = int((length - 1) / n_chunks)
    remainder = (length - 1) % n_chunks
    bulk_length = length - remainder

    path_bulk = path[1:bulk_length]
    path_bulk = jnp.reshape(path_bulk, (n_chunks, chunk_length, dim))
    basepoints = jnp.roll(path_bulk[:, -1], shift=1, axis=0)
    basepoints = basepoints.at[0].set(path[0])
    path_bulk = jnp.concatenate([basepoints[:, None, :], path_bulk], axis=1)
    path_remainder = path[bulk_length:]

    def _signature(path):
        return signature(path, depth)

    # this will return a list of [(b, n), (b, n, n), ...]
    multi_signatures = jax.vmap(_signature)(path_bulk)
    bulk_signature = multi_signature_combine(multi_signatures)

    if remainder != 0:
        # compute the signature of the remainder chunk
        remainder_signature = signature(path_remainder, depth)
        # combine with the bulk signature
        return mult(bulk_signature, remainder_signature)
    else:
        # no remainder, just return the bulk
        return bulk_signature


def logsignature(path, depth):
    return signature_to_logsignature(signature(path, depth))


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


def signature_combine(
    signature1: List[jnp.ndarray],
    signature2: List[jnp.ndarray],
):
    return mult(signature1, signature2)


@jax.jit
def multi_signature_combine(signatures: List[jnp.ndarray]):
    """
    Combine multiple signatures.

    The input of this function is the output of `jax.vmap` version of
    signature function.

    Args:
        signatures: size [(b, n), (b, n, n), (b, n, n) ...]
    Returns:
        size [(n,), (n, n), (n, n, n), ...]
    """
    batch_size = signatures[0].shape[0]

    init_val = [x[0] for x in signatures]

    def _body_fn(i, val):
        current = [x[i] for x in signatures]
        ret = mult(val, current)
        return ret

    combined = jax.lax.fori_loop(
        lower=1, upper=batch_size, body_fun=_body_fn, init_val=init_val
    )
    return combined
