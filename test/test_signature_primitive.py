import jax.numpy as jnp
import torch
import signatory
from timeit import default_timer

from signax.signature_primitive_no_gpu import signature

x, y, z = jnp.array([2, 0.5, 1], dtype=jnp.float32)
source = [
    [[1., 2.], [3., 4.]],
    [[5., 6.], [7., 8.]],
]

in_arr = jnp.array(source, dtype=jnp.float32, )
in_tensor = torch.tensor(source)

starttime = default_timer()
print("The start time is :", starttime)
signature(in_arr, 4)
print("The time difference is :", default_timer() - starttime)

print("The start time is :", starttime)
signatory.signature(in_tensor, 4)
print("The time difference is :", default_timer() - starttime)
