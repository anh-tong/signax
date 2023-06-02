from __future__ import annotations

__all__ = (
    "signature",
    "logsignature",
    "signature_batch",
    "signature_combine",
    "signature_to_logsignature",
    "multi_signature_combine",
)

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from signax.tensor_ops import log, mult, mult_fused_restricted_exp, restricted_exp
from signax.utils import compress, lyndon_words


@partial(jax.jit, static_argnames=["depth", "stream"])
def signature(
    path: Float[Array, "path_len dim"],
    depth: int,
    stream: bool = False,
) -> list[Array]:
    """
    Compute the signature of a path

    Args:
        path: size (length, dim)
        depth: signature is truncated at this depth
        stream: whether to handle `path` as a stream. Default is False
    Returns:
        If `stream` is `True`, this will return a list of `Array` in a form
            [(path_len - 1, dim), (path_len - 1, dim, dim), (path_len - 1, dim, dim, dim), ...]
        If `stream` is `False`, this will return a list of `Array` in a form
            [(dim, ), (dim, dim), (dim, dim, dim), ...]
    """

    path_increments = jnp.diff(path, axis=0)
    exp_term = restricted_exp(path_increments[0], depth=depth)

    def _body(carry, path_inc):
        ret = mult_fused_restricted_exp(path_inc, carry)
        return ret, ret

    carry, stacked = jax.lax.scan(f=_body, init=exp_term, xs=path_increments[1:])

    if stream:
        return [
            jnp.concatenate([first[None, ...], rest], axis=0)
            for first, rest in zip(exp_term, stacked)
        ]

    return carry


@partial(jax.jit, static_argnames=["depth", "n_chunks", "stream"])
def signature_batch(
    path: Float[Array, "path_len dim"], depth: int, n_chunks: int, stream: bool = False
) -> list[Array]:
    """Compute signature for a long path

    The path will be divided into chunks to compute signatures. Then, obtained signatures are combined (using Chen's identity).
    The numbers of chunks is set manually.

    Args:
        path: Input path
        depth: signature depth
        n_chunks: number of chunks
        stream: whether to handle `path` as a stream
    Returns:
        If `stream` is `True`, this will return a list of `Array` in a form
            [(path_len - 1, dim), (path_len - 1, dim, dim), (path_len - 1, dim, dim, dim), ...]
        If `stream` is `False`, this will return a list of `Array` in a form
            [(dim, ), (dim, dim), (dim, dim, dim), ...]
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

    multi_signatures = jax.vmap(partial(signature, depth=depth, stream=stream))(
        path_bulk
    )

    if stream:
        # `multi_signature` in a form of [(chunk, chunk_len, dim), (chunk, chunk_len, dim, dim), ...]
        def scan_fn(last_sig, current_stream_signature):
            # combine the last signature of the previous chunk with every signature in the current stream
            combined = jax.vmap(partial(signature_combine, last_sig))(
                current_stream_signature
            )
            # return `carry` as the last signature of the combined signatures
            last_sig = [x[-1, ...] for x in combined]
            return last_sig, combined

        # initial value is the last of the stream in  the first chunk
        init = [sig[0, -1, ...] for sig in multi_signatures]
        last_sig, bulk_signature = jax.lax.scan(
            f=scan_fn, init=init, xs=[sig[1:] for sig in multi_signatures]
        )

        bulk_signature = [
            jnp.concatenate([x[0][None, ...], y])
            for x, y in zip(multi_signatures, bulk_signature)
        ]
        bulk_signature = [
            jnp.reshape(sig, (sig.shape[0] * sig.shape[1],) + sig.shape[2:])
            for sig in bulk_signature
        ]

        if remainder != 0:
            remainder_signature = signature(path_remainder, depth, stream)
            combined = jax.vmap(partial(signature_combine, last_sig))(
                remainder_signature
            )
            return [
                jnp.concatenate([bulk, rest], axis=0)
                for bulk, rest in zip(bulk_signature, combined)
            ]

        return bulk_signature

    # this is the case when `stream`=False
    # `multi_signature` in a form of [(chunk, dim), (chunk, dim, dim), ...]
    bulk_signature = multi_signature_combine(multi_signatures)

    if remainder != 0:
        # compute the signature of the remainder chunk
        remainder_signature = signature(path_remainder, depth, stream)
        # combine with the bulk signature
        return signature_combine(bulk_signature, remainder_signature)

    # no remainder, just return the bulk
    return bulk_signature


def logsignature(
    path: Float[Array, "path_len dim"], depth: int, stream: bool = False
) -> list[Array]:
    sig = signature(path, depth, stream)
    if stream:
        return jax.vmap(signature_to_logsignature)(sig)
    return signature_to_logsignature(sig)


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
    signature1: list[Array],
    signature2: list[Array],
) -> list[Array]:
    return mult(signature1, signature2)


@jax.jit
def multi_signature_combine(signatures: list[Array]) -> list[Array]:
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
