from functools import partial
from typing import List

import jax
import jax.numpy as jnp


@jax.jit
def otimes(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
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
def addcmul(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray):
    """Similar to `torch.addcmul` returning
        x + y * z
    Here `*` is the tensor product
    """
    return x + otimes(y, z)


@partial(jax.jit, static_argnames="depth")
def restricted_exp(input: jnp.ndarray, depth: int) -> List[jnp.ndarray]:
    """Restricted exponentiate

    As `depth` is fixed so we can make it as a static argument.
    This allows us to `jit` this function
    Args:
        input: shape (n, )
        depth: the depth of signature
    Return:
        A list of `jnp.ndarray` contains tensors
    """
    ret = [input]
    for i in range(2, depth + 1):
        ret.append(otimes(ret[-1], input / i))
    return ret


@jax.jit
def mult_fused_restricted_exp(
    z: jnp.ndarray, A: List[jnp.ndarray]
) -> List[jnp.ndarray]:
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
        current = 1.0
        for i in range(depth_index + 1):
            current = addcmul(x=A[i], y=current, z=z / (depth_index + 1 - i))
        ret.append(current)

    return ret


@partial(jax.jit, static_argnums=2)
def mult_inner(
    A: List[jnp.ndarray],
    B: List[jnp.ndarray],
    depth_index: int,
) -> List[jnp.ndarray]:
    """
    Let `depth_index` = n

    this function returns
        $sum_{i=1}^n A_i x B_{n - i}$


    Note this is hard to convert to `jax.lax.fori_loop`.
    I don't know if it's possible. Several attempts but
    `TracerIntergerConversionError` is encountered because
    getting index of a list (it's okay to get index of ndarray
    but not for lists)
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
def mult(A: List[jnp.ndarray], B: List[jnp.ndarray]) -> List[jnp.ndarray]:
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
    input1: List[jnp.ndarray],
    input2: List[jnp.ndarray],
    scalar_term_value: float,
    top_terms_to_skip: int,
) -> List[jnp.ndarray]:
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
def log(input: List[jnp.ndarray]) -> List[jnp.ndarray]:
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
