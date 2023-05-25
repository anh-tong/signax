import jax
import jax.numpy as jnp
import numpy as np
import signatory
import torch
from signax.signature import (  # noqa: E501
    multi_signature_combine,
    signature,
    signature_batch,
)


jax.config.update("jax_platform_name", "cpu")


def test_signature_1d_path():
    depth = 3
    length = 100
    path = np.random.randn(length, 1)
    signature(path, depth)

    path = np.random.randn(length)
    signature(path, depth)


def test_multi_signature_combine():
    batch_size = 5
    dim = 5
    signatures = [
        np.random.randn(batch_size, dim),
        np.random.randn(batch_size, dim, dim),
        np.random.randn(batch_size, dim, dim, dim),
    ]

    jax_signatures = [jnp.array(x) for x in signatures]

    jax_output = multi_signature_combine(jax_signatures)
    jax_sum = sum(jnp.sum(x) for x in jax_output)

    torch_signatures = []
    for i in range(batch_size):
        tensors = [torch.tensor(x[i]) for x in signatures]
        current = torch.cat([t.flatten() for t in tensors])
        current = current[None, :]
        torch_signatures.append(current)

    torch_output = signatory.multi_signature_combine(
        torch_signatures, input_channels=dim, depth=len(signatures)
    )
    torch_sum = torch_output.sum().item()
    assert jnp.allclose(jax_sum, torch_sum, rtol=1e-2, atol=1e-1)


def test_signature_batch():
    depth = 3

    # no remainder case
    length = 1001
    dim = 100
    n_chunks = 10

    path = np.random.randn(length, dim)
    jax_signature = signature_batch(path, depth, n_chunks)
    jax_sum = sum(jnp.sum(x) for x in jax_signature)

    torch_path = torch.tensor(path)
    torch_signature = signatory.signature(torch_path[None, ...], depth=depth)
    torch_sum = torch_signature.sum().item()

    # TODO: this has a low precision error
    assert jnp.allclose(jax_sum, torch_sum, rtol=1e-2, atol=1e-1)

    # has remainder case
    length = 1005
    path = np.random.randn(length, dim)

    jax_signature = signature_batch(path, depth, n_chunks)
    jax_sum = sum(jnp.sum(x) for x in jax_signature)

    torch_path = torch.tensor(path)
    torch_signature = signatory.signature(torch_path[None, ...], depth=depth)
    torch_sum = torch_signature.sum().item()

    # TODO: this has a low precision error
    assert jnp.allclose(jax_sum, torch_sum, rtol=1e-2, atol=1e-1)
