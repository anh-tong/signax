from __future__ import annotations

from collections import namedtuple
from functools import partial

import einops
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
from jaxtyping import Array, Float

from signax import signature


def split_key(key):
    return None if key is None else jrandom.split(key, 1)[0]


class TensorLinear(eqx.Module):
    lin: nn.Linear
    n_heads: int = eqx.static_field()

    def __init__(
        self,
        in_features,
        out_features,
        order,
        n_heads=1,
        use_bias=True,
        *,
        key: PRNGKey,
    ):
        assert order >= 1
        self.n_heads = n_heads
        self.lin = nn.Linear(
            in_features**order,
            n_heads * out_features**order,
            use_bias=use_bias,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, " dim"],
        *,
        key: PRNGKey | None = None,
    ) -> Float[Array, " out_dim"]:
        x = self.lin(x, key=key)

        x = einops.rearrange(
            x, "(n_heads out_dim) -> n_heads out_dim", n_heads=self.n_heads
        )
        return x


class TensorLinearOutput(eqx.Module):
    lin: nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        order,
        n_heads=1,
        use_bias=True,
        *,
        key: PRNGKey,
    ):
        self.lin = nn.Linear(
            n_heads * in_features**order,
            out_features**order,
            use_bias=use_bias,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, "n_heads dim"],
        *,
        key: PRNGKey | None = None,  # noqa: ARG002
    ) -> Float[Array, " out_dim"]:
        x = einops.rearrange(x, "... -> (...)")
        x = self.lin(x)

        return x


class SelfAttentionAtDepth(eqx.Module):
    query_proj: TensorLinear
    key_proj: TensorLinear
    value_proj: TensorLinear
    output_proj: TensorLinear

    dropout: nn.Dropout

    n_heads: int = eqx.static_field()

    def __init__(
        self,
        order: int,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        *,
        key: jrandom.PRNGKey,
    ):
        qkey, kkey, vkey, okey = jrandom.split(key, 4)

        query_size = key_size = value_size = output_size = dim
        self.n_heads = n_heads
        self.query_proj = TensorLinear(
            in_features=query_size,
            out_features=query_size,
            order=order,
            n_heads=n_heads,
            use_bias=False,
            key=qkey,
        )

        self.key_proj = TensorLinear(
            in_features=key_size,
            out_features=key_size,
            order=order,
            n_heads=n_heads,
            use_bias=False,
            key=kkey,
        )

        self.value_proj = TensorLinear(
            in_features=value_size,
            out_features=value_size,
            order=order,
            n_heads=n_heads,
            key=vkey,
        )

        self.output_proj = TensorLinearOutput(
            in_features=output_size,
            out_features=output_size,
            order=order,
            n_heads=n_heads,
            key=okey,
        )

        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        x: Float[Array, "seq_len dim"],
        *,
        key: PRNGKey = None,
    ) -> Float[Array, "seq_len dim"]:
        shape = x.shape
        seq_len = shape[0]
        x = einops.rearrange(x, "seq_len ... -> seq_len (...)")

        q = jax.vmap(self.query_proj)(x)
        k = jax.vmap(self.key_proj)(x)
        v = jax.vmap(self.value_proj)(x)

        mask = jnp.tril(jnp.ones((seq_len, seq_len)))

        attn_fn = partial(
            eqx.nn._attention.dot_product_attention,
            dropout=self.dropout,
            mask=mask,
            inference=None,
        )

        keys = None if key is None else jrandom.split(key, self.n_heads)
        x = jax.vmap(attn_fn, in_axes=1, out_axes=1)(q, k, v, key=keys)
        x = jax.vmap(self.output_proj)(x)

        x = jnp.reshape(x, shape)

        return x


class TensorSelfAttention(eqx.Module):
    all_attn: list[SelfAttentionAtDepth]

    def __init__(
        self,
        order: int,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        *,
        key: jrandom.PRNGKey,
    ) -> None:
        all_attn = []
        for i in range(order):
            attn = SelfAttentionAtDepth(
                order=i + 1,
                dim=dim,
                n_heads=n_heads,
                dropout=dropout,
                key=jrandom.fold_in(key, i + 1),
            )
            all_attn.append(attn)

        self.all_attn = all_attn

    def __call__(
        self,
        x: list[Array],
        *,
        key: jrandom.PRNGKey = None,
    ) -> list[Array]:
        result = []
        for attn, xx in zip(self.all_attn, x):
            key = split_key(key)
            attn_x = attn(x=xx, key=key)
            result.append(attn_x)

        return result


class TensorLayerNorm(eqx.Module):
    norms: list[nn.LayerNorm]

    def __init__(self, dim: int, order: int):
        self.norms = [nn.LayerNorm((dim,) * i) for i in range(1, order + 1)]

    def __call__(self, x: list[Array]) -> list[Array]:
        return [norm(xx) for norm, xx in zip(self.norms, x)]


class TensorDropout(eqx.Module):
    dropout: nn.Dropout

    def __init__(
        self,
        dropout_p=0.0,
    ) -> None:
        self.dropout = nn.Dropout(dropout_p)

    def __call__(self, x: list[Array], *, key=None) -> list[Array]:
        key = [None] * len(x) if key is None else jrandom.split(key, len(x))
        return [self.dropout(xx, key=kk) for xx, kk in zip(x, key)]


class TensorMLP(eqx.Module):
    ff: list[nn.Sequential]

    def __init__(self, dim: int, order: int, d_ff: int, dropout=0.0, *, key: PRNGKey):
        self.ff = [
            nn.Sequential(
                layers=[
                    nn.Linear(dim**i, d_ff, key=jrandom.fold_in(key, i * 2)),
                    nn.Lambda(jax.nn.gelu),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, dim**i, key=jrandom.fold_in(key, i * 2 + 1)),
                ]
            )
            for i in range(1, order + 1)
        ]

    def __call__(
        self,
        x: list[Array],
        *,
        key: PRNGKey = None,
    ) -> list[Array]:
        shapes = [xx.shape for xx in x]
        x = [einops.rearrange(xx, "... -> (...)") for xx in x]
        key = [None] * len(x) if key is None else jrandom.split(key, len(x))
        x = [ff(xx, key=kk) for ff, xx, kk in zip(self.ff, x, key)]
        x = [jnp.reshape(xx, shape) for xx, shape in zip(x, shapes)]
        return x


class TensorAdd(eqx.Module):
    def __call__(self, x: list[Array], y: list[Array]) -> list[Array]:
        return [xx + yy for xx, yy in zip(x, y)]


class TensorFlatten(eqx.Module):
    def __call__(self, x: list[Array]) -> Float[Array, "seq_len dim"]:
        x = [einops.rearrange(xx, "seq ... -> seq (...)") for xx in x]
        x = jnp.concatenate(x, axis=-1)
        return x


def lead_lag(x: Float[Array, "seq_len dim"], n_step_delay=1):
    """
    E.g., given x = [1, 6, 3, 9, 5], and n_step_delay = 1
        return [(1,1), (6,1), (6, 6), (3, 6), (3, 3), (9, 3), (9, 9), (5, 9), (5, 5)]

    Args:
        x : input stream
        n_step_delay (int, optional): number of steps delay. Defaults to 1.
    """

    x_repreated = jnp.repeat(x, n_step_delay + 1, axis=0)
    all = [x_repreated[n_step_delay:]]

    for i in range(n_step_delay - 1, -1, -1):
        all += [x_repreated[i : -(n_step_delay - i)]]

    return jnp.concatenate(all, axis=-1)


class LeadLagSignature(eqx.Module):
    patch_len: int = eqx.static_field()
    signature_depth: int = eqx.static_field()

    def __init__(self, depth, patch_len):
        self.patch_len = patch_len
        self.signature_depth = depth

    def __call__(self, x: Float[Array, "seq_len dim"]):
        seq_len, dim = x.shape

        x = jnp.pad(x, ((self.patch_len - 1, 0), (0, 0)), constant_values=0.0)

        index = jnp.arange(seq_len)

        def _f(carry, i):
            patch = jax.lax.dynamic_slice(x, (i, 0), (self.patch_len, dim))
            lead_lag_patch = lead_lag(patch)
            sig = signature(
                lead_lag_patch,
                depth=self.signature_depth,
                stream=False,
                flatten=False,
            )
            return carry, sig

        # output shape: (number_strides, patch_len, dim)
        _, output = jax.lax.scan(f=_f, init=None, xs=index)
        return output


Config = namedtuple(
    "Config",
    [
        "in_dim",
        "out_dim",
        "dim",
        "num_heads",
        "d_ff",
        "dropout",
        "n_attn_blocks",
        "order",
    ],
)


class Block(eqx.Module):
    attn_block: TensorSelfAttention
    attn_norm: TensorLayerNorm

    mlp_block: TensorMLP
    mlp_norm: TensorLayerNorm

    dropout: TensorDropout

    add_fn: TensorAdd

    def __init__(self, config: Config, *, key: PRNGKey) -> None:
        attn_key, mlp_key = jrandom.split(key)

        dim = config.dim
        order = config.order
        dropout = config.dropout
        n_heads = config.num_heads
        self.attn_block = TensorSelfAttention(
            order=order, dim=dim, dropout=dropout, n_heads=n_heads, key=attn_key
        )
        self.attn_norm = TensorLayerNorm(dim, order)

        self.mlp_block = TensorMLP(
            dim=dim, order=order, d_ff=dim * dim * 4, key=mlp_key
        )
        self.mlp_norm = TensorLayerNorm(dim, order)
        self.dropout = TensorDropout(dropout)
        self.add_fn = TensorAdd()

    def __call__(
        self,
        x: list[Array],
        *,
        key: PRNGKey = None,
    ) -> list[Array]:
        # attention
        resid = x
        x = self.attn_block(x, key=key)
        key = split_key(key)
        x = self.add_fn(resid, self.dropout(x, key=key))
        x = self.attn_norm(x)

        # MLP
        resid = x
        x = jax.vmap(self.mlp_block)(x)
        key = split_key(key)
        x = self.add_fn(resid, self.dropout(x, key=key))
        x = self.mlp_norm(x)

        return x


class SigFormer(eqx.Module):
    config: Config = eqx.static_field()

    project: nn.Linear
    blocks: list[Block]
    readout: nn.Linear
    flatten: TensorFlatten

    def __init__(self, config: Config, *, key: PRNGKey):
        block_key, proj_key, readout_key = jrandom.split(key, 3)
        self.config = config
        in_dim = config.in_dim
        out_dim = config.out_dim
        dim = config.dim

        self.project = nn.Linear(in_dim, dim, key=proj_key)
        blocks = []
        for i in range(config.n_attn_blocks):
            block = Block(config, key=jrandom.fold_in(block_key, i))
            blocks.append(block)
        self.blocks = blocks
        self.flatten = TensorFlatten()
        readout_in_dim = sum(config.dim ** (i + 1) for i in range(config.order))
        self.readout = nn.Linear(readout_in_dim, out_dim, key=readout_key)

    def __call__(
        self,
        x: Float[Array, "seq_len in_dim"],
        *,
        key: PRNGKey,
    ) -> Float[Array, "seq_len out_dim"]:
        x = jax.vmap(self.project)(x)

        # compute signature
        x = jnp.pad(x, ((1, 0), (0, 0)), constant_values=0.0)
        x = signature(x, depth=self.config.order, stream=True, flatten=False)

        for block in self.blocks:
            key = split_key(key)
            x = block(x, key=key)

        x = self.flatten(x)

        x = jax.vmap(self.readout)(x)

        return x
