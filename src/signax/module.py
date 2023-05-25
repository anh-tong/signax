from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from signax.signature_flattened import signature, signature_combine


class SignatureTransform(eqx.Module):
    depth: int

    def __init__(self, depth: int):
        self.depth = depth

    def __call__(
        self,
        path: jnp.ndarray,
        *,
        key: Optional["jax.random.PRNGKey"] = None,  # noqa: F821
    ) -> jnp.ndarray:
        return signature(path, self.depth)


class SignatureCombine(eqx.Module):
    dim: int
    depth: int

    def __init__(self, dim: int, depth: int):
        self.dim = dim
        self.depth = depth

    def __call__(self, signature1: jnp.ndarray, signature2: jnp.ndarray):
        return signature_combine(signature1, signature2, self.dim, self.depth)
