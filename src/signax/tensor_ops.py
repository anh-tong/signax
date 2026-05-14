from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def otimes(x: jax.Array, y: jax.Array) -> jax.Array:
    """Tensor product

    Args:
        x: size=(n,n,...,n), ndim=ndim_x
        y: size=(n,n,...,n), ndim=ndim_y
    Return:
        Tensor size (n,n,...,n) with ndim=ndim_x + ndim_y
    """
    expanded_x = jnp.reshape(x, x.shape + (1,) * y.ndim)
    expanded_y = jnp.reshape(y, (1,) * x.ndim + y.shape)
    return expanded_x * expanded_y


@jax.jit
def addcmul(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    """Similar to `torch.addcmul` returning
        x + y * z
    Here `*` is the tensor product
    """
    return x + otimes(y, z)


_UNROLL_THRESHOLD = 5


def _restricted_exp_unrolled(input: jax.Array, depth: int) -> list[jax.Array]:
    """Python-unrolled variant; generates a flat XLA chain at trace time."""
    ret = [input]
    for i in range(2, depth + 1):
        ret.append(otimes(ret[-1], input / i))
    return ret


def _restricted_exp_flat(input: jax.Array, depth: int) -> list[jax.Array]:
    """Flat-buffer variant for depth > _UNROLL_THRESHOLD.

    Stores all depth levels in one contiguous 1-D JAX array, then extracts
    each level at the end.  Slice indices are Python ints (dim and depth are
    both concrete at JIT-trace time), so no dynamic indexing overhead is
    incurred.

    Note: a true single-construct XLA loop (jax.lax.fori_loop) would further
    reduce the compiled graph node count but requires a fixed-shape carry,
    which is incompatible with the per-level size growth dim^1 … dim^depth.
    The flat-buffer layout at least makes the memory pattern explicit and
    provides a direct foundation for a future padded-carry implementation.
    """
    dim = input.shape[0]
    sizes: list[int] = [dim**d for d in range(1, depth + 1)]
    offsets: list[int] = []
    off = 0
    for s in sizes:
        offsets.append(off)
        off += s

    flat = jnp.zeros(off, dtype=input.dtype)
    flat = flat.at[offsets[0] : offsets[0] + sizes[0]].set(input)

    for d in range(1, depth):
        prev_flat = flat[offsets[d - 1] : offsets[d - 1] + sizes[d - 1]]
        prev = prev_flat.reshape((dim,) * d)
        nxt = otimes(prev, input / (d + 1))
        flat = flat.at[offsets[d] : offsets[d] + sizes[d]].set(nxt.ravel())

    return [
        flat[offsets[d] : offsets[d] + sizes[d]].reshape((dim,) * (d + 1))
        for d in range(depth)
    ]


@partial(jax.jit, static_argnames="depth")
def restricted_exp(input: jax.Array, depth: int) -> list[jax.Array]:
    """Restricted exponentiate

    As `depth` is fixed so we can make it as a static argument.
    This allows us to `jit` this function
    Args:
        input: shape (n, )
        depth: the depth of signature
    Return:
        A list of `jnp.ndarray` contains tensors
    """
    if depth <= _UNROLL_THRESHOLD:
        return _restricted_exp_unrolled(input, depth)
    return _restricted_exp_flat(input, depth)


@jax.jit
def mult_fused_restricted_exp(z: jax.Array, A: list[jax.Array]) -> list[jax.Array]:
    """
    Multiply-fused-exponentiate

    Args:
        z: shape (n,)
        A: a list of `jnp.array` [(n, ), (n x n), (n x n x n), ...]
    Return:
        A list of which elements have the same shape with `A`
    """
    depth = len(A)
    ret = []

    for depth_index in range(depth):
        current = jnp.array(1.0)
        for i in range(depth_index + 1):
            current = addcmul(x=A[i], y=current, z=z / (depth_index + 1 - i))
        ret.append(current)

    return ret


@partial(jax.jit, static_argnums=2)
def mult_inner(
    A: list[jax.Array],
    B: list[jax.Array],
    depth_index: int,
) -> list[jax.Array]:
    """
    Let `depth_index` = n

    this function returns
        $sum_{i=1}^n A_i x B_{n - i}$
    """
    return sum(
        [
            otimes(
                A[i],
                B[depth_index - i - 1],
            )
            for i in range(depth_index)
        ]
    )


@jax.jit
def mult(A: list[jax.Array], B: list[jax.Array]) -> list[jax.Array]:
    """
    Multiplication in tensor algebra

    Args:
        A: [(dim,), (dim,dim), (dim,dim,dim), ...]
        B: [(dim,), (dim,dim), (dim,dim,dim), ...]
    """

    depth = len(A)
    C = [a + b for a, b in zip(A, B)]
    for i in range(1, depth):
        C[i] += mult_inner(A, B, depth_index=i)

    return C


@partial(jax.jit, static_argnums=3)
def mult_partial(
    input1: list[jax.Array],
    input2: list[jax.Array],
    scalar_term_value: float,
    top_terms_to_skip: int,
) -> list[jax.Array]:
    """Sort of multiplication in the tensor algebra

    `input1` assumed scalar value
    `input2` assume scalar value zero

    return `input1` x `input2` for some of its terms

    many terms are left unchanged

    This corresponds to compute one parenthesis in Equation (10)
    (see iisignature paper)
    """
    depth = len(input1)
    for depth_index in reversed(range(depth - top_terms_to_skip)):
        input1[depth_index] = jnp.zeros_like(input1[depth_index])
        input1[depth_index] = mult_inner(input1, input2, depth_index)
        input1[depth_index] = (
            input1[depth_index] + input2[depth_index] * scalar_term_value
        )

    return input1


def _log_coef_at_depth(depth: int) -> float:
    """Note that reciprocals is an array [1/2, 1/3, ...]"""
    sign = -1.0 if depth % 2 == 0 else 1.0
    return sign / (depth + 2)


@jax.jit
def log(input: list[jax.Array]) -> list[jax.Array]:
    """This follows Equation (10) of iisignature paper"""

    depth = len(input)
    if depth == 1:
        return input

    output = [jnp.zeros_like(x) for x in input]
    output[0] = input[0] * _log_coef_at_depth(depth - 2)

    for depth_index in reversed(range(depth - 2)):
        output = mult_partial(
            output,
            input,
            scalar_term_value=_log_coef_at_depth(depth_index),
            top_terms_to_skip=depth_index + 1,
        )

    output = mult_partial(
        output,
        input,
        scalar_term_value=1.0,
        top_terms_to_skip=0,
    )

    return output
