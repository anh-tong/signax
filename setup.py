import sys

import setuptools
from torch.utils import cpp_extension as cpp

extra_compile_args = []
if not sys.platform.startswith("win"):  # linux or mac
    extra_compile_args.append("-fvisibility=hidden")
if sys.platform.startswith("win"):  # windows
    extra_compile_args.append("/openmp")
else:  # linux or mac
    extra_compile_args.append("-fopenmp")

ext_modules = [
    cpp.CppExtension(
        name="cpu_ops",
        sources=[
            "signax/backend/cpu_ops.cc",
            "signax/backend/logsignature.cpp",
            "signax/backend/lyndon.cpp",
            "signax/backend/misc.cpp",
            "signax/backend/signature.cpp",
            "signax/backend/tensor_algebra_ops.cpp",
        ],
        depends=[
            "signax/backend/pybind11_helpers.h",
            "signax/backend/logsignature.hpp",
            "signax/backend/lyndon.hpp",
            "signax/backend/misc.hpp",
            "signax/backend/signature.hpp",
            "signax/tensor_algebra_ops.hpp",
        ],
        extra_compile_args=extra_compile_args,
    )
]

python_requires = "~=3.7"
install_requires = ["jax>=0.3.10", "pybind11>=2.6", "cmake"]

setuptools.setup(
    name='signax',
    version="1.0",
    author="signax authors",
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp.BuildExtension},
    python_requires=python_requires,
    install_requires=install_requires,
)
