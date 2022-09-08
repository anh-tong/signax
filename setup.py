import pathlib

import setuptools


HERE = pathlib.Path(__file__).resolve().parent

metadata = {
    "name": "signax",
    "version": "0.1.0",
    "author": "signax authors",
    "author_email": "anh.h.tong@gmail.com",
}

python_requires = "~=3.7"
install_requires = ["jax>=0.3.10", "equinox"]

license = "MIT"

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
]

description = "Signax: Signature computation in JAX"
url = "https://github.com/anh-tong/signax"

with open(HERE / "README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    name=metadata["name"],
    version=metadata["version"],
    author=metadata["author"],
    author_email=metadata["author_email"],
    maintainer=metadata["author"],
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=url,
    license=license,
    classifiers=classifiers,
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    packages=setuptools.find_packages(exclude=["examples", "test"]),
)
