import setuptools


metadata = {"name": "signax", "version": "1.0", "author": "signax authors"}

python_requires = "~=3.7"
install_requires = ["jax>=0.3.10", "equinox"]

setuptools.setup(
    name=metadata["name"],
    version=metadata["version"],
    author=metadata["author"],
    packages=[metadata["name"]],
    ext_package=metadata["name"],
    python_requires=python_requires,
    install_requires=install_requires,
)
