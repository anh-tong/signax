from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import lax


def autocovariance(hurst: float, n: int):
    ns_2h = jnp.arange(n + 1) * (2**hurst)
    ret = 0.5 * (ns_2h[:-2] - 2 * ns_2h[1:-1] + ns_2h[2:])
    ret = jnp.insert(ret, 0, 1)
    return ret


def sqrt_eigenvals(hurst: float, n: int):
    ifft = jnp.fft.irfft(autocovariance(hurst, n))[:n]
    return jnp.sqrt(ifft)


@partial(jax.jit, static_argnames=["t0", "t1", "dt"])
def fbm_noise(hurst, t0=0, t1=1, dt=1e-2, *, key):
    """Generate Fractional Brownian noise"""
    n = int((t1 - t0) / dt)
    scale = dt**hurst
    m = 2 ** (n - 2).bit_length() + 1
    sq_eig = sqrt_eigenvals(hurst, m)
    scale *= 2**0.5 * (m - 1)

    real_key, imag_key = jrandom.split(key)
    w_real = jrandom.normal(key=real_key, shape=(m,)) * scale
    w_real = w_real.at[0].set(w_real[0] * 2**0.5)
    w_real = w_real.at[-1].set(w_real[-1] * 2**0.5)
    w_imag = jrandom.normal(key=imag_key, shape=(m,)) * scale
    w = lax.complex(w_real, w_imag)
    return jnp.fft.irfft(sq_eig * w)[:n]


def generate_fbm(
    hurst, n_paths, t0: float = 0.0, t1: float = 1.0, dt: float = 1e-2, *, key
):
    keys = jrandom.split(key, num=n_paths)
    if hurst == 0.5:
        # Generate Brownian motion noise
        def fn(_key):
            gaussian_noise = jrandom.normal(
                key=_key,
                shape=(int((t1 - t0) / dt),),
            )
            return gaussian_noise * (dt**0.5)

    else:
        # Generate FBM noise
        def fn(_key):
            return fbm_noise(
                hurst=hurst,
                t0=t0,
                t1=t1,
                dt=dt,
                key=_key,
            )

    delta = jax.vmap(fn)(keys)
    return delta.cumsum(axis=-1)
