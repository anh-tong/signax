# import os
import pathlib
import re

# import subprocess
import sys

import setuptools


# from setuptools.command.build_ext import build_ext


try:
    import torch.utils.cpp_extension as cpp
except ImportError:
    raise ImportError("need to install Pytorch")

here = pathlib.Path(__file__).resolve().parent
name = "signax"

with open(here / name / "__init__.py") as f:
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if match:
        version = match.group(1)
    else:
        RuntimeError("Can not determine version")


# class CMakeBuildExt(build_ext):
#     def build_extensions(self):
#         import platform
#         import sys
#         from distutils import sysconfig

#         import pybind11

#         if "windows" == platform.system():
#             raise RuntimeError("Not support Windows")
#         else:
#             cmake_python_library = "{}/{}".format(
#                 sysconfig.get_config_var("LIBDIR"),
#                 sysconfig.get_config_var("INSTSONAME"),
#             )
#         cmake_python_include_dir = sysconfig.get_python_inc()
#         install_dir = os.path.abspath(
#             os.path.dirname(self.get_ext_fullpath("dummy")),
#         )
#         os.makedirs(install_dir, exist_ok=True)

#         prefix_paths = [pybind11.get_cmake_dir(
#         )] +
# ["/home/anhth/anaconda3/lib/python3.9/site-packages/torch/share/cmake/Torch/"]
#         prefix_paths = ";".join(prefix_paths)

#         cmake_args = [
#             f"-DCMAKE_INSTALL_PREFIX={install_dir}",
#             f"-DPython_EXECUTABLE={sys.executable}",
#             f"-DPython_LIBRARIES={cmake_python_library}",
#             f"-DPython_INCLUDE_DIRS={cmake_python_include_dir}",
#             "-DCMAKE_BUILD_TYPE={}".format(
#                 "Debug" if self.debug else "Release",
#             ),
#             f"-DCMAKE_PREFIX_PATH={prefix_paths}",
#         ]

#         cmake_args.append(f"-DCUDNN_LIBRARY_PATH=/usr/local/cudnn-9.2-v7.1/")
#         cmake_args.append(
#             f"-DCUDNN_INCLUDE_PATH=/usr/local/cudnn-9.2-v7.1/include")

#         if os.environ.get("BUILD_WITH_CUDA", "no") == "yes":
#             cmake_args.append("-DBUILD_WITH_CUDA=yes")

#         os.makedirs(self.build_temp, exist_ok=True)
#         subprocess.check_call(
#             ["cmake", here] + cmake_args,
#             cwd=self.build_temp,
#         )

#         super().build_extensions()

#         subprocess.check_call(
#             ["cmake", "--build", ".", "--target", "install"],
#             cwd=self.build_temp,
#         )

#     def build_extension(self, ext) -> None:
#         target_name = ext.name.split(".")[-1]
#         subprocess.check_call(
#             ["cmake", "--build", ".", "--target", target_name],
#             cwd=self.build_temp,
#         )


# extensions = [
#     setuptools.Extension(
#         "signax.cpu_ops",
#         ["signax/backend/cpu_ops.cc"],
#     )
# ]
# if os.environ.get("BUILD_WITH_CUDA", "no") == "yes":
#     extensions.append(
#         setuptools.Extension(
#             "signax.cpu_ops",
#             [
#                 "signax/backend/gpu_ops.cc",
#                 "signax/backend/cuda_kernels.cc.cu",
#             ],
#         )
#     )


extra_compile_args = []

if not sys.platform.startswith("win"):  # linux or mac
    extra_compile_args.append("-fvisibility=hidden")

if sys.platform.startswith("win"):  # windows
    extra_compile_args.append("/openmp")
else:  # linux or mac
    extra_compile_args.append("-fopenmp")

ext_modules = [
    cpp.CppExtension(
        name="signax.cpu_ops",
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
    name=name,
    version=version,
    author="signax authors",
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp.BuildExtension},
    python_requires=python_requires,
    install_requires=install_requires,
)
