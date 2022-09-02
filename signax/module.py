from typing import List

import equinox as eqx
import jax.numpy as jnp

from signax.signature import signature


class SignatureTransform(eqx.Module):
    depth: int
    tensor_algebra: List[jnp.ndarray]

    def __init__(self, depth: int):
        self.depth = depth
        self.tensor_algebra = []

    def __call__(self, path) -> List[jnp.ndarray]:
        self.tensor_algebra = signature(path, self.depth)
        print(self.tensor_algebra)
        return self.tensor_algebra
