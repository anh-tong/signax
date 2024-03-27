from __future__ import annotations

import iisignature
import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import default_rng

from signax.tensor_ops import (
    addcmul,
    mult,
    mult_fused_restricted_exp,
    otimes,
    restricted_exp,
)

rng = default_rng()


jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def test_otimes():
    # 1D x 1D
    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])

    true_output = jnp.array([[3.0, 4.0], [6.0, 8.0]])
    assert jnp.allclose(true_output, otimes(x, y), rtol=1e-3, atol=1e-5)

    # 1D x 2D
    x = jnp.array([1.0, 2.0])
    y = jnp.array([[3.0, 4.0], [5.0, 6.0]])
    true_output = jnp.array(
        [
            [[3.0, 4.0], [5.0, 6.0]],
            [[6.0, 8.0], [10.0, 12.0]],
        ]
    )
    assert jnp.allclose(true_output, otimes(x, y), rtol=1e-3, atol=1e-5)

    # 2D x 1D
    true_output = jnp.array(
        [
            [[3.0, 6.0], [4.0, 8.0]],
            [[5.0, 10.0], [6.0, 12.0]],
        ]
    )

    assert jnp.allclose(true_output, otimes(y, x), rtol=1e-3, atol=1e-5)


def test_addcmul():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([1.0, 2.0])
    z = jnp.array([3.0, 4.0])
    true_output = jnp.array([[4.0, 6.0], [9.0, 12.0]])
    assert jnp.allclose(true_output, addcmul(x, y, z), rtol=1e-3, atol=1e-5)


def test_restricted_exp():
    depth = 4
    length, dim = 2, 3
    path = rng.standard_normal((length, dim))
    iis_signature = jnp.asarray(iisignature.sig(np.asarray(path), depth))
    jax_output = restricted_exp(jnp.diff(path, axis=0), depth=depth)
    jax_output = jnp.concatenate([jnp.ravel(x) for x in jax_output])
    assert jnp.allclose(iis_signature, jax_output, rtol=1e-3, atol=1e-5)


def test_mult_fused_restricted_exp():
    depth = 4
    length, dim = 3, 3
    path = rng.standard_normal((length, dim))

    # re-test restricted_exp() to make sure it run correctly
    test_restricted_exp()

    iis_signature = jnp.asarray(iisignature.sig(np.asarray(path), depth))

    # our computation
    increments = jnp.diff(path, axis=0)
    exp_term = restricted_exp(increments[0], depth)
    jax_output = mult_fused_restricted_exp(increments[1], exp_term)
    jax_output = jnp.concatenate([jnp.ravel(x) for x in jax_output])

    assert jnp.allclose(iis_signature, jax_output)


def test_mult():
    depth = 4
    length, dim = 3, 4
    path = rng.standard_normal((length, dim))

    # use our implementation, need to compute exp first
    increments = jnp.diff(path, axis=0)
    exp1 = restricted_exp(increments[0], depth)
    exp2 = restricted_exp(increments[1], depth)
    combine = mult(exp1, exp2)
    jax_output = jnp.concatenate([jnp.ravel(x) for x in combine])
    iis_signature = jnp.asarray(
        iisignature.sigcombine(
            np.asarray(jnp.concatenate([a.reshape(-1) for a in exp1])),
            np.asarray(jnp.concatenate([a.reshape(-1) for a in exp2])),
            dim,
            depth,
        )
    )
    assert jnp.allclose(iis_signature, jax_output)


# def test_log():
#     """Test log via signature_to_logsignature"""
#     depth = 4
#     length, dim = 3, 2
#     path = rng.standard_normal((length, dim))
#     jax_path = jnp.array(path)
#     jax_signature = signature(jax_path, depth, flatten=False)
#     jax_logsignature = signature_to_logsignature(jax_signature)
#     jax_output = jnp.concatenate([jnp.ravel(x) for x in jax_logsignature])

#     torch_signature = signatory.signature(
#         torch.tensor(path)[None, ...],
#         depth,
#     )
#     torch_logsignature = signatory.signature_to_logsignature(
#         torch_signature,
#         dim,
#         depth,
#     )

#     torch_output = jnp.array(torch_logsignature.numpy())

#     assert jnp.allclose(torch_output, jax_output)
