from __future__ import annotations

import equinox as eqx
import jax

from signax import signature, signature_combine
from signax.utils import flatten, unravel_signature


class SignatureTransform(eqx.Module):
    depth: int

    def __init__(self, depth: int):
        self.depth = depth

    def __call__(
        self,
        path: jax.Array,
    ) -> jax.Array:
        return flatten(signature(path, self.depth))


class SignatureCombine(eqx.Module):
    dim: int
    depth: int

    def __init__(self, dim: int, depth: int):
        self.dim = dim
        self.depth = depth

    def __call__(self, signature1: jax.Array, signature2: jax.Array):
        sig1 = unravel_signature(signature1, self.dim, self.depth)
        sig2 = unravel_signature(signature2, self.dim, self.depth)
        return flatten(signature_combine(sig1, sig2))
