from functools import wraps
from timeit import default_timer

import jax
import jax.numpy as jnp
import numpy as np
import signatory
import torch

from signax.signature_jax import compute_signature, combine_signatures


def timeis(iterations=1):
    def wrapper_outer(fun):
        @wraps(fun)
        def wrapper_inner(*args, **kwargs):
            output = None
            start_time = default_timer()
            for _ in range(iterations):
                output = fun(*args, **kwargs)
            end_time = default_timer()
            elapsed_time = end_time - start_time
            print(f"{fun.__name__} launched {iterations} times:\n"
                  f"Average time:\t{elapsed_time / iterations}\n"
                  f"Elapsed time:\t{elapsed_time}\n")
            return output

        return wrapper_inner

    return wrapper_outer


benchmark_runs = 100
depth = 2
np.random.seed(0)
batch_size, path_len, features_num = 2, 2, 2
source = np.random.randn(batch_size, path_len, features_num)

in_tensor = torch.tensor(source).requires_grad_(True)
in_array = jnp.array(source)


@timeis(benchmark_runs)
def benchmark_signatory():
    return signatory.signature(in_tensor, depth)


_ = jax.vmap(lambda _: compute_signature(_, depth))(in_array)


@timeis(benchmark_runs)
def benchmark_pure_jax():
    return jax.vmap(lambda x: compute_signature(x, depth))(in_array)


@timeis(benchmark_runs)
def benchmark_pure_jax_vjp():
    def loss(x):
        signatures = compute_signature(x, depth)
        return sum(map(jnp.sum, signatures))

    def batch_loss(batch_val):
        return jnp.sum(jax.vmap(loss)(batch_val))

    grad, val = jax.value_and_grad(batch_loss)(in_array)
    val.block_until_ready()
    grad.block_until_ready()


@timeis(benchmark_runs)
def benchmark_signatory_backprop():
    def loss(x):
        signatures = signatory.signature(x, depth)
        loss_val = torch.sum(signatures)
        loss_val.backward()
        return loss_val

    grad = loss(in_tensor)


source_continuation = np.hstack((
    source[:, -1:, :],
    np.random.randn(batch_size, path_len - 1, features_num)
))

in_array_continuation = jnp.array(source_continuation)

sig_jax_1 = jax.vmap(lambda x: compute_signature(x, depth))(in_array)
sig_jax_2 = jax.vmap(lambda x: compute_signature(x, depth))(in_array_continuation)

in_tensor_continuation = torch.tensor(source_continuation)
sig_signatory_1 = signatory.signature(in_tensor, depth)
sig_signatory_2 = signatory.signature(in_tensor_continuation, depth)


@timeis(benchmark_runs)
def benchmark_signatory_combination():
    return signatory.signature_combine(
        sig_signatory_1, sig_signatory_2,
        features_num,
        depth
    )


@timeis(benchmark_runs)
def benchmark_jax_combination():
    return combine_signatures((sig_jax_1, sig_jax_2))


# benchmark_signatory()
# benchmark_pure_jax()
# benchmark_signatory_backprop()
# benchmark_pure_jax_vjp()
print(benchmark_signatory_combination())
print(benchmark_jax_combination())
