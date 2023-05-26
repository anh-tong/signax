from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .tensor_ops import log, mult, mult_fused_restricted_exp, restricted_exp
from .utils import compress, lyndon_words


@partial(jax.jit, static_argnames="depth")
def signature(path: jax.Array, depth: int) -> list[jax.Array]:
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
        return mult_fused_restricted_exp(path_increments[i], val)

    exp_term = jax.lax.fori_loop(
        lower=1,
        upper=path_increments.shape[0],
        body_fun=_body,
        init_val=exp_term,
    )

    return exp_term


@partial(jax.jit, static_argnames=["depth", "n_chunks"])
def signature_batch(path: jax.Array, depth: int, n_chunks: int):
    """Compute signature for a long path

    The path will be divided into chunks. The numbers of chunks
    is set manually.

    Args:
        path: size (length, dim)
        depth: signature depth
        n_chunks:
    Returns:
        signature in a form of [(n,), (n,n), ...]
    """
    length, dim = path.shape
    chunk_length = int((length - 1) / n_chunks)
    remainder = (length - 1) % n_chunks
    bulk_length = length - remainder

    path_bulk = path[1:bulk_length]
    path_bulk = jnp.reshape(path_bulk, (n_chunks, chunk_length, dim))
    basepoints = jnp.roll(path_bulk[:, -1], shift=1, axis=0)
    basepoints = basepoints.at[0].set(path[0])
    path_bulk = jnp.concatenate([basepoints[:, None, :], path_bulk], axis=1)
    path_remainder = path[bulk_length - 1 :]

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

    # no remainder, just return the bulk
    return bulk_signature


def logsignature(path, depth):
    return signature_to_logsignature(signature(path, depth))


def signature_to_logsignature(
    signature: list[jax.Array],
) -> list[jax.Array]:
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
    return compress(expanded_logsignature, indices)


def signature_combine(
    signature1: list[jax.Array],
    signature2: list[jax.Array],
):
    return mult(signature1, signature2)


@jax.jit
def multi_signature_combine(signatures: list[jax.Array]) -> list[jax.Array]:
    """
    Combine multiple signatures.

    The input of this function is the output of `jax.vmap` version of
    signature function.

    Args:
        signatures: size [(b, n), (b, n, n), (b, n, n) ...]
    Returns:
        size [(n,), (n, n), (n, n, n), ...]
    """
    result = jax.lax.associative_scan(
        fn=jax.vmap(signature_combine),
        elems=signatures,
    )
    # return the last index after associative scan
    result = jax.tree_map(lambda x: x[-1], result)

    return result
