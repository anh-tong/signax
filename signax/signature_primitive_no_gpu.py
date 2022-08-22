import os
import sys
from functools import partial

import numpy as np
from jax import core, dtypes, lax
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla
from jaxlib import xla_client
# from signax.backend import cpu_ops
import torch
from signax import cpu_ops

for name, value in cpu_ops.registrations().items():
    xla_client.register_cpu_custom_call_target(name, value)


def get_signature_shape(path_shape, sig_depth):
    batches, path_len, features = path_shape
    total_features_size = 0
    acc_features_size = features

    for i in range(sig_depth):
        total_features_size += acc_features_size
        acc_features_size *= features

    return batches, total_features_size


def _signature_abstract(path, depth):
    dtype = dtypes.canonicalize_dtype(path.dtype)
    # return output_shape
    # output_shape = get_signature_shape(path.shape, depth)
    return ShapedArray(flattened_output_shape, dtype)


def _signature_cpu_translation(ctx, path, depth, platform="cpu"):
    path_shape = ctx.get_shape(path)
    path_dims = np.array(path_shape.dimensions())

    dtype = path_shape.element_type()

    if dtype == np.float32:
        cpu_op_name = b"cpu_signature_f32"
    elif dtype == np.float64:
        cpu_op_name = b"cpu_signature_f64"
    else:
        raise NotImplementedError(f"dtype {dtype} is not supported")

    path_dim = xla_client.ops.Constant(ctx, path_dims)
    path_dim_shape = xla_client.Shape.array_shape(
        np.dtype(np.int64), path_dims.shape, (0,)
    )
    depth_shape = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())

    return xla_client.ops.CustomCallWithLayout(
        ctx,
        cpu_op_name,
        operands=(path, path_dim, depth),
        operand_shapes_with_layout=(path_shape, path_dim_shape, depth_shape),
        shape_with_layout=xla_client.Shape.array_shape(
            dtype, flattened_output_shape, (0,)
        ),
        # shape_with_layout=path_shape,
    )


def signature_cpu_batch(vector_arg_values, batch_axes):
    assert batch_axes[0] == 0
    res = lax.map(lambda x: signature(*x), vector_arg_values)
    return res, batch_axes[0]


flattened_output_shape = None
def signature(path, depth):
    global flattened_output_shape
    output_shape = get_signature_shape(path.shape, depth)
    flattened_output_shape = (np.prod(output_shape), )

    flattened_output = _signature_prim.bind(path, depth)
    return flattened_output.reshape(output_shape)


_signature_prim = core.Primitive("signature")
_signature_prim.def_impl(partial(xla.apply_primitive, _signature_prim))
_signature_prim.def_abstract_eval(_signature_abstract)

xla.backend_specific_translations["cpu"][_signature_prim] = partial(
    _signature_cpu_translation, platform="cpu"
)
