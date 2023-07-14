from __future__ import annotations

from typing import Any

__all__ = ("SignatureTransform", "LogSignatureTransform")

import equinox as eqx
from jaxtyping import Array, Float

from signax.signatures import logsignature, signature


class SignatureTransform(eqx.Module):
    depth: int
    stream: bool
    num_chunks: int = 1

    def __init__(self, depth: int, stream: bool = False) -> None:
        self.depth = depth
        self.stream = stream

    def __call__(
        self,
        path: Float[Array, "path_len dim"],
        *,
        key: Any | None = None,
    ) -> Array:
        return signature(
            path, self.depth, self.stream, flatten=True, num_chunks=self.num_chunks
        )


class LogSignatureTransform(eqx.Module):
    depth: int
    stream: bool
    num_chunks: int = 1

    def __init__(self, depth: int, stream: bool = False) -> None:
        self.depth = depth
        self.stream = stream

    def __call__(
        self,
        path: Float[Array, "path_len dim"],
        *,
        key: Any | None = None,
    ) -> Array:
        return logsignature(
            path, self.depth, self.stream, flatten=True, num_chunks=self.num_chunks
        )
