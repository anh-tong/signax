from __future__ import annotations

import pytest
from numpy.random import default_rng

from signax.module import LogSignatureTransform, SignatureTransform

rng = default_rng()


@pytest.mark.parametrize("stream", [True, False])
def test_signature_module(stream):
    depth = 3
    length = 100
    dim = 5
    x = rng.standard_normal((length, dim))

    model = SignatureTransform(depth=depth, stream=stream)
    model(x)


@pytest.mark.parametrize("stream", [True, False])
def test_logsignature_module(stream):
    depth = 3
    length = 100
    dim = 5
    x = rng.standard_normal((length, dim))

    model = LogSignatureTransform(depth=depth, stream=stream)
    model(x)
