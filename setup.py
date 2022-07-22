import os
import pathlib
import re
import subprocess

import setuptools
from setuptools.command.build_ext import build_ext


here = pathlib.Path(__file__).resolve().parent
name = "signax"

with open(here / name / "__init__.py") as f:
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if match:
        version = match.group(1)
    else:
        RuntimeError("Can not determine version")


class CMakeBuildExt(build_ext):
    def build_extensions(self):
        import platform
        import sys
        from distutils import sysconfig

        import pybind11

        if "windows" == platform.system():
            raise RuntimeError("Not support Windows")
        else:
            cmake_python_library = "{}/{}".format(
                sysconfig.get_config_var("LIBDIR"),
                sysconfig.get_config_var("INSTSONAME"),
            )
        cmake_python_include_dir = sysconfig.get_python_inc()
        install_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath("dummy")),
        )
        os.makedirs(install_dir, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_LIBRARIES={cmake_python_library}",
            f"-DPython_INCLUDE_DIRS={cmake_python_include_dir}",
            "-DCMAKE_BUILD_TYPE={}".format(
                "Debug" if self.debug else "Release",
            ),
            f"-DCMAKE_PREFIX_PATH={pybind11.get_cmake_dir()}",
        ]
        if os.environ.get("BUILD_WITH_CUDA", "no") == "yes":
            cmake_args.append("-DBUILD_WITH_CUDA=yes")

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(
            ["cmake", here] + cmake_args,
            cwd=self.build_temp,
        )

        super().build_extensions()

        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"],
            cwd=self.build_temp,
        )

    def build_extension(self, ext) -> None:
        target_name = ext.name.split(".")[-1]
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", target_name],
            cwd=self.build_temp,
        )


extensions = [
    setuptools.Extension(
        "signax.cpu_ops",
        ["signax/backend/cpu_ops.cc"],
    )
]
if os.environ.get("BUILD_WITH_CUDA", "no") == "yes":
    extensions.append(
        setuptools.Extension(
            "signax.cpu_ops",
            [
                "signax/backend/gpu_ops.cc",
                "signax/backend/cuda_kernels.cc.cu",
            ],
        )
    )

python_requires = "~=3.7"
install_requires = ["jax>=0.3.10", "pybind11>=2.6", "cmake"]

setuptools.setup(
    name=name,
    version=version,
    author="signax authors",
    ext_modules=extensions,
    cmdclass={"build_ext": CMakeBuildExt},
    python_requires=python_requires,
    install_requires=install_requires,
)
