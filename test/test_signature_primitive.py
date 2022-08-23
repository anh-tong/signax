from timeit import default_timer

import jax
import jax.numpy as jnp
import numpy as np
import signatory
import torch

from signax.signature_jax import compute_signature
from signax.signature_primitive_no_gpu import signature


def timeis(fun):
    def wrap(*args, **kwargs):
        start_time = default_timer()
        output = fun(*args, **kwargs)
        end_time = default_timer()
        print(f"{fun.__name__}: {end_time - start_time}")
        return output

    return wrap


depth = 10
x, y, z = jnp.array([2, 0.5, 1], dtype=jnp.float32)

source = np.random.randn(10, 10, 3)
# source = [
#     [[1., 2.], [3., 4.]],
#     [[5., 6.], [7., 8.]],
# ]

in_arr = jnp.array(source, dtype=jnp.float32, )
in_arr = jnp.swapaxes(in_arr, 0, 1)
in_tensor = torch.tensor(source)
jnp_x = jnp.array(source)


@timeis
def benchmark_signature_cpp():
    signature(in_arr, depth)


@timeis
def benchmark_signatory():
    signatory_lib_output = signatory.signature(in_tensor, depth)


_ = jax.vmap(lambda _: compute_signature(_, depth))(jnp_x)


@timeis
def benchmark_pure_jax():
    fused = jax.vmap(lambda x: compute_signature(x, depth))(jnp_x)
    [f.block_until_ready() for f in fused]


benchmark_signature_cpp()
benchmark_signatory()
benchmark_pure_jax()
