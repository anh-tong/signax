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


def _signature_abstract(path, depth):
    dtype = dtypes.canonicalize_dtype(path.dtype)
    output_shape = np.prod(path.shape)
    return ShapedArray((output_shape,), dtype)


def _signature_cpu_translation(ctx, path, depth, platform="cpu"):
    path_shape = ctx.get_shape(path)
    tensor_dimensions = np.array(path_shape.dimensions())

    dtype = path_shape.element_type()

    if dtype == np.float32:
        cpu_op_name = b"cpu_signature_f32"
    elif dtype == np.float64:
        cpu_op_name = b"cpu_signature_f64"
    else:
        raise NotImplementedError(f"dtype {dtype} is not supported")

    path_dim = xla_client.ops.Constant(ctx, tensor_dimensions)
    path_dim_shape = xla_client.Shape.array_shape(
        np.dtype(np.int64), tensor_dimensions.shape, (0,)
    )
    depth_shape = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())

    return xla_client.ops.CustomCallWithLayout(
        ctx,
        cpu_op_name,
        operands=(path, path_dim, depth),
        operand_shapes_with_layout=(path_shape, path_dim_shape, depth_shape),
        shape_with_layout=xla_client.Shape.array_shape(
            dtype, (np.prod(tensor_dimensions),), (0,)
        ),
        # shape_with_layout=path_shape,
    )


def signature_cpu_batch(vector_arg_values, batch_axes):
    assert batch_axes[0] == 0
    res = lax.map(lambda x: signature(*x), vector_arg_values)
    return res, batch_axes[0]


def signature(path, depth):
    flattened_output = _signature_prim.bind(path, depth)
    return flattened_output.reshape(path.shape)


_signature_prim = core.Primitive("signature")
_signature_prim.def_impl(partial(xla.apply_primitive, _signature_prim))
_signature_prim.def_abstract_eval(_signature_abstract)

xla.backend_specific_translations["cpu"][_signature_prim] = partial(
    _signature_cpu_translation, platform="cpu"
)
