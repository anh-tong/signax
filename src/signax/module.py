from __future__ import annotations

import equinox as eqx
import jax

from signax.signature_flattened import signature, signature_combine


class SignatureTransform(eqx.Module):
    depth: int

    def __init__(self, depth: int):
        self.depth = depth

    def __call__(
        self,
        path: jax.Array,
    ) -> jax.Array:
        return signature(path, self.depth)


class SignatureCombine(eqx.Module):
    dim: int
    depth: int

    def __init__(self, dim: int, depth: int):
        self.dim = dim
        self.depth = depth

    def __call__(self, signature1: jax.Array, signature2: jax.Array):
        return signature_combine(signature1, signature2, self.dim, self.depth)
