import jax
import jax.numpy as jnp
import numpy as np
import signatory

# need to install torch and signatory for testing
import torch
from signax.signature import signature, signature_to_logsignature
from signax.tensor_ops import (
    addcmul,
    mult,
    mult_fused_restricted_exp,
    otimes,
    restricted_exp,
)


jax.config.update("jax_platform_name", "cpu")


def test_otimes():

    # 1D x 1D
    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])

    true_output = jnp.array([[3.0, 4.0], [6.0, 8.0]])
    assert jnp.allclose(true_output, otimes(x, y))

    # 1D x 2D
    x = jnp.array([1.0, 2.0])
    y = jnp.array([[3.0, 4.0], [5.0, 6.0]])
    true_output = jnp.array(
        [
            [[3.0, 4.0], [5.0, 6.0]],
            [[6.0, 8.0], [10.0, 12.0]],
        ]
    )
    assert jnp.allclose(true_output, otimes(x, y))

    # 2D x 1D
    true_output = jnp.array(
        [
            [[3.0, 6.0], [4.0, 8.0]],
            [[5.0, 10.0], [6.0, 12.0]],
        ]
    )

    assert jnp.allclose(true_output, otimes(y, x))


def test_addcmul():

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([1.0, 2.0])
    z = jnp.array([3.0, 4.0])
    true_output = jnp.array([[4.0, 6.0], [9.0, 12.0]])
    assert jnp.allclose(true_output, addcmul(x, y, z))


def test_restricted_exp():

    depth = 4
    length, dim = 2, 3
    path = np.random.randn(length, dim)

    signatory_output = (
        signatory.signature(
            torch.tensor(path)[None, ...],
            depth=depth,
        )
        .sum()
        .item()
    )
    jax_output = restricted_exp(jnp.diff(path, axis=0), depth=depth)
    jax_output = sum([jnp.sum(x) for x in jax_output])
    # only check the sum of output in two cases are the same
    assert jnp.allclose(signatory_output, jax_output)


def test_mult_fused_restricted_exp():

    depth = 4
    length, dim = 3, 3
    path = np.random.randn(length, dim)

    # re-test restricted_exp() to make sure it run correctly
    test_restricted_exp()

    signatory_output = (
        signatory.signature(
            torch.tensor(path)[None, ...],
            depth=depth,
        )
        .sum()
        .item()
    )

    # our computation
    increments = jnp.diff(path, axis=0)
    exp_term = restricted_exp(increments[0], depth)
    jax_output = mult_fused_restricted_exp(increments[1], exp_term)
    jax_output = sum([jnp.sum(x) for x in jax_output])
    # again, just check the sum
    assert jnp.allclose(signatory_output, jax_output)


def test_mult():

    depth = 4
    length, dim = 3, 4
    path = np.random.randn(length, dim)

    # use our implementation, need to compute exp first
    increments = jnp.diff(path, axis=0)
    exp1 = restricted_exp(increments[0], depth)
    exp2 = restricted_exp(increments[1], depth)
    combine = mult(exp1, exp2)
    jax_output = sum(jnp.sum(x) for x in combine)
    jax_output = jax_output.item()

    # use signatory
    exp1 = torch.tensor(
        np.array(jnp.concatenate([x.ravel() for x in exp1])),
    )[None, :]
    exp2 = torch.tensor(
        np.array(jnp.concatenate([x.ravel() for x in exp2])),
    )[None, :]
    signatory_output = signatory.signature_combine(exp2, exp1, dim, depth)
    signatory_output = signatory_output.sum().item()

    assert jnp.allclose(signatory_output, jax_output, rtol=1e-3)


def test_log():
    """Test log via signature_to_logsignature"""
    depth = 4
    length, dim = 3, 2
    path = np.random.randn(length, dim)
    jax_path = jnp.array(path)
    jax_signature = signature(jax_path, depth)
    jax_logsignature = signature_to_logsignature(jax_signature)

    jax_output = sum(jnp.sum(x) for x in jax_logsignature)
    jax_output = jax_output.item()

    torch_signature = signatory.signature(
        torch.tensor(path)[None, ...],
        depth,
    )
    torch_logsignature = signatory.signature_to_logsignature(
        torch_signature,
        dim,
        depth,
    )

    torch_output = torch_logsignature.sum().item()

    assert jnp.allclose(torch_output, jax_output)
