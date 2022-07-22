# to understand JAX, see
# https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
#

from jax import core
from jax._src.lib import xla_client
from jax.interpreters import xla
from signax import cpu_ops  # this module in comple in C++ using Pybind11
from signax import gpu_ops  # this module in comple in C++ using Pybind11


signature_p = core.Primitive("signature")

# ===================================================================
# Register custom call target
# ===================================================================
for name, value in cpu_ops.registrations().items():
    xla_client.register_cpu_custom_call_target(name, value)

for name, value in gpu_ops.registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")

# ===================================================================
#  All necessary functions for primitive
# ===================================================================


def signature_impl(
    paths,
    depth,
    stream,
    basepoint,
    inverse,
    initial,
    scalar_term,
):
    raise NotImplementedError


def signature_abstract_eval():
    """This is to validate shape"""
    raise NotImplementedError


def signature_xla_translation_cpu(
    ctx,
    avals_in,
    avals_out,
    paths,
    depth,
    stream,
    basepoint,
    inverse,
    initial,
    scalar_term,
):

    raise NotImplementedError


def signature_xla_translation_gpu(
    ctx,
    avals_in,
    avals_out,
    paths,
    depth,
    stream,
    basepoint,
    inverse,
    initial,
    scalar_term,
):

    raise NotImplementedError


# ===================================================================
#  Ready to register primitive
# ===================================================================
signature_p.def_impl(signature_impl)

xla.backend_specific_translations["cpu"][
    signature_p
] = signature_xla_translation_cpu  # noqa: E501
xla.backend_specific_translations["gpu"][
    signature_p
] = signature_xla_translation_gpu  # noqa: E501
