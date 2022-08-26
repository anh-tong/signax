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
        cur = 1.0
        for i in range(depth_index + 1):
            cur = addcmul(A[i], cur, z=z / (depth_index + 1 - i))
        ret.append(cur)

    return ret


# def mult_inner(tensor_at_depth: jnp.ndarray):
#     pass

def addcmul(A, prev, z):
    return A + jnp.expand_dims(prev, axis=-1) * z[None, :]


@partial(jax.jit, static_argnames="depth")
def compute_signature(path, depth):
    path_increments = jnp.diff(path, axis=0)
    exp_term = restricted_exp(path_increments[0], depth=depth)

    def _body(i, val):
        ret = mult_fused_restricted_exp(path_increments[i], val)
        return [x.squeeze() for x in ret]

    return jax.lax.fori_loop(
        lower=1,
        upper=path_increments.shape[0],
        body_fun=_body,
        init_val=exp_term,
    )


def mult(sig1, sig2):
    depth = len(sig1)
    res = sig1.copy()
    for i in range(depth - 1, -1, -1):
        res[i] = sig1[i]
        for j in range(0, i):
            res[i] += jnp.expand_dims(sig1[i - 1 - j], -2) * jnp.expand_dims(sig2[j], -1)
        res[i] += sig2[i]

    return res


def combine_signatures(signatures):
    out = signatures[0]

    for i in range(1, len(signatures)):
        out = mult(signatures[i], out)

    return out
