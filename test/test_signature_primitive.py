import timeit
from functools import partial
from timeit import default_timer
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import signatory
import torch

from signax.signature_primitive_no_gpu import signature

x, y, z = jnp.array([2, 0.5, 1], dtype=jnp.float32)
source = [
    [[1., 2.], [3., 4.]],
    [[5., 6.], [7., 8.]],
]

in_arr = jnp.array(source, dtype=jnp.float32, )
in_arr = jnp.swapaxes(in_arr, 0,1)
in_tensor = torch.tensor(source)

starttime = default_timer()
print("The start time is :", starttime)
signature(in_arr, 4)
print("The time difference is :", default_timer() - starttime)

print("The start time is :", starttime)
signatory_lib_output = signatory.signature(in_tensor, 4)
print("The time difference is :", default_timer() - starttime)


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
    last = input
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


np.random.seed(0)
x = np.array([
    [0.0, 0.0], [1, 2], [2, 4]
])

depth = 10
jnp_x = jnp.array(x)
torch_x = torch.as_tensor(x)[None, :]
# make sure to compile first
_ = compute_signature(jnp_x, depth)

print("The start time is :", starttime)
fused = compute_signature(jnp_x, depth)
[f.block_until_ready() for f in fused]
print("The time difference is :", default_timer() - starttime)
