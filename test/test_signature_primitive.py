import jax.numpy as jnp
from signax.signature_primitive_no_gpu import signature


x, y, z = jnp.array([2, 0.5, 1], dtype=jnp.float32)
out_arr = jnp.array([2], dtype=jnp.float32)
in_arr = jnp.array(
    [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ],
    dtype=jnp.float32,
)

print(signature(in_arr, 4))
