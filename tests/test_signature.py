from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import signatory
import torch
from numpy.random import default_rng

from signax import (
    logsignature,
    multi_signature_combine,
    signature,
    signature_batch,
)

rng = default_rng()

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def test_signature_1d_path():
    depth = 3
    length = 100
    path = rng.standard_normal((length, 1))
    signature(path, depth)

    path = rng.standard_normal(length)
    signature(path, depth)


def test_multi_signature_combine():
    batch_size = 5
    dim = 5
    signatures = [
        rng.standard_normal((batch_size, dim)),
        rng.standard_normal((batch_size, dim, dim)),
        rng.standard_normal((batch_size, dim, dim, dim)),
    ]

    jax_signatures = [jnp.array(x) for x in signatures]

    jax_output = multi_signature_combine(jax_signatures)
    jax_output = jnp.concatenate([jnp.ravel(x) for x in jax_output])

    torch_signatures = []
    for i in range(batch_size):
        tensors = [torch.tensor(x[i]) for x in signatures]
        current = torch.cat([t.flatten() for t in tensors])
        current = current[None, :]
        torch_signatures.append(current)

    torch_output = signatory.multi_signature_combine(
        torch_signatures, input_channels=dim, depth=len(signatures)
    )
    torch_output = jnp.array(torch_output.numpy())
    assert jnp.allclose(jax_output, torch_output)


@pytest.mark.parametrize("stream", [True, False])
def test_signature_batch(stream):
    depth = 3

    # no remainder case
    length = 1001
    dim = 5
    n_chunks = 10

    path = rng.standard_normal((length, dim))
    jax_signature = signature_batch(path, depth, n_chunks, stream=stream)
    ravel_fn = jax.vmap(jnp.ravel) if stream else jnp.ravel
    jax_signature = jnp.concatenate([ravel_fn(x) for x in jax_signature], axis=-1)

    torch_path = torch.tensor(path)
    torch_signature = signatory.signature(
        torch_path[None, ...], depth=depth, stream=stream
    )
    torch_signature = jnp.array(torch_signature.numpy())

    assert jnp.allclose(jax_signature, torch_signature)

    # has remainder case
    length = 1005
    path = rng.standard_normal((length, dim))

    jax_signature = signature_batch(path, depth, n_chunks, stream)
    jax_signature = jnp.concatenate([ravel_fn(x) for x in jax_signature], axis=-1)

    torch_path = torch.tensor(path)
    torch_signature = signatory.signature(
        torch_path[None, ...], depth=depth, stream=stream
    )
    torch_signature = jnp.array(torch_signature.numpy())

    assert jnp.allclose(jax_signature, torch_signature)


@pytest.mark.parametrize("stream", [True, False])
def test_signature(stream):
    depth = 3
    length = 100
    dim = 5

    path = rng.standard_normal((length, dim))
    jax_signature = signature(path, depth=depth, stream=stream)
    ravel_fn = jax.vmap(jnp.ravel) if stream else jnp.ravel
    jax_signature = jnp.concatenate([ravel_fn(x) for x in jax_signature], axis=-1)

    torch_path = torch.tensor(path)
    torch_signature = signatory.signature(
        torch_path[None, ...], depth=depth, stream=stream
    )
    torch_signature = jnp.array(torch_signature.numpy())

    assert jnp.allclose(jax_signature, torch_signature)


@pytest.mark.parametrize("stream", [True, False])
def test_logsignature(stream):
    depth = 3
    length = 100
    dim = 5

    path = rng.standard_normal((length, dim))
    jax_signature = logsignature(path, depth=depth, stream=stream)
    ravel_fn = jax.vmap(jnp.ravel) if stream else jnp.ravel
    jax_signature = jnp.concatenate([ravel_fn(x) for x in jax_signature], axis=-1)

    torch_path = torch.tensor(path)
    torch_signature = signatory.logsignature(
        torch_path[None, ...], depth=depth, stream=stream
    )
    torch_signature = jnp.array(torch_signature.numpy())

    assert jnp.allclose(jax_signature, torch_signature)
