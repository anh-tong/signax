
Signax - Signature computation in JAX
=====================================


Introduction
------------
Signax is a JAX library for signature computation.


Installation
------------

Install via pip

.. code-block:: bash

   python3 -m pip install signax


Install via source

.. code-block:: bash

   git clone https://github.com/anh-tong/signax.git
   cd signax
   python3 -m pip install -v -e .


Get Started
-----------

.. code-block:: python

    import jax
    import jax.random as jrandom
    import signax


    key = jrandom.PRNGKey(0)
    depth = 3

    # compute signature for a single path
    length = 100
    dim = 20
    path = jrandom.normal(shape=(length, dim), key=key)
    output = signax.signature(path, depth)
    # output is a list of array representing tensor algebra

    # compute signature for batches (multiple) of paths
    # this is done via `jax.vmap`
    batch_size = 20
    path = jrandom.normal(shape=(batch_size, length, dim), key=key)
    output = jax.vmap(lambda x: signax.signature(x, depth))(path)


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples
   :glob:

   examples/inversion
   examples/generative_model
   examples/estimate_hurst
