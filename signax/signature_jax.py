from functools import partial
from typing import List

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames="depth")
def restricted_exp(input: jnp.ndarray, depth: int):
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
        last = jnp.expand_dims(ret[-1], axis=-1) * input[None, :] / i
        ret += [last]
    return ret


@jax.jit
def mult_fused_restricted_exp(z: jnp.ndarray, A: List[jnp.ndarray]):
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
        last = 1.0
        for i in range(depth_index + 1):
            current = addcmul(A[i], last, z=z / (depth_index + 1 - i))
            last = current
        ret.append(last)

    return ret


# def mult_inner(tensor_at_depth: jnp.ndarray):
#     pass


def addcmul(A, prev, z):
    return A + jnp.expand_dims(prev, axis=-1) * z[None, :]


@partial(jax.jit, static_argnames="depth")
def compute_signature(x, depth):
    diff_x = jnp.diff(x, axis=0)
    exp_term = restricted_exp(diff_x[0], depth=depth)
    fused = mult_fused_restricted_exp(diff_x[1], exp_term)
    return fused