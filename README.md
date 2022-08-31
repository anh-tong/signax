
## Goal

To have a library that supports signature computation in JAX. See [this paper](https://arxiv.org/abs/1905.08494) to see how to adopt signatures in machine learning.

This implementation is inspired by [patrick-kidger/signatory](https://github.com/patrick-kidger/signatory).


## Examples

<!-- TODO: example with equinox -->

## Parallelism 

This implementation makes use of `jax.vmap` to perform the parallelism over batch dimension. 

Signatory allows dividing a path into chunks and performing asynchronous multithread computation over chunks. 
<!-- TODO: This implementation allow perform  -->

## Why is using pure JAX good enough?

JAX make use of just-in-time (JIT) compilations. 

We observe that the performance of this implementation is similar to Signatory in CPU and slightly better in GPU. It could be because of the optimized operators of XLA in JAX. As mentioned in the paper, signatory is not fully optimized for CUDA but relies on LibTorch.