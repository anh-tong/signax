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

metadata = {
    "name": "signax",
    "version": "1.0",
    "author": "signax authors"
}


ext_modules = [
    cpp.CppExtension(
        name="cpu_ops",
        sources=[
            "signax/backend/signatory/logsignature.cpp",
            "signax/backend/signatory/lyndon.cpp",
            "signax/backend/signatory/misc.cpp",
            "signax/backend/signatory/signature.cpp",
            "signax/backend/signatory/tensor_algebra_ops.cpp",
            "signax/backend/cpu_ops.cc",
        ],
        depends=[
            "signax/backend/pybind11_helpers.h",
            "signax/backend/signatory/logsignature.hpp",
            "signax/backend/signatory/lyndon.hpp",
            "signax/backend/signatory/misc.hpp",
            "signax/backend/signatory/signature.hpp",
            "signax/backend/signatory/tensor_algebra_ops.hpp",
        ],
        extra_compile_args=extra_compile_args,
    )
]

python_requires = "~=3.7"
install_requires = ["jax>=0.3.10", "pybind11>=2.6", "cmake"]

setuptools.setup(
    name=metadata["name"],
    version=metadata["version"],
    author=metadata["author"],
    ext_modules=ext_modules,
    packages=[metadata["name"]],
    ext_package=metadata["name"],
    cmdclass={"build_ext": cpp.BuildExtension},
    python_requires=python_requires,
    install_requires=install_requires,
)
