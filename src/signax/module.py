from __future__ import annotations

from typing import Any

__all__ = ("SignatureTransform",)

import equinox as eqx
from jaxtyping import Array, Float

from signax.signatures import logsignature, signature
from signax.utils import flatten


class SignatureTransform(eqx.Module):
    depth: int
    stream: bool

    def __init__(self, depth: int, stream: bool) -> None:
        self.depth = depth
        self.stream = stream

    def __call__(
        self,
        path: Float[Array, "path_len dim"],
        *,
        key: Any | None = None,
    ) -> Array:
        return flatten(signature(path, self.depth, self.stream))


class LogSignatureTransform(eqx.Module):
    depth: int
    stream: bool

    def __init__(self, depth: int, stream: bool) -> None:
        self.depth = depth
        self.stream = stream

    def __call__(
        self,
        path: Float[Array, "path_len dim"],
        *,
        key: Any | None = None,
    ) -> Array:
        return flatten(logsignature(path, self.depth, self.stream))
