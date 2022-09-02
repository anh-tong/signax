import jax
import numpy as np
import signatory
import torch

from signax.signature_flattened import signature, signature_combine

np.random.seed(0)

n_batch = 4
length = 3
dim = 2
x1 = np.random.rand(n_batch, length, dim)
x2 = np.hstack((
    x1[:, -1:, :].copy(),
    np.random.randn(n_batch, length - 1, dim)
))

depth = 3

torch_x1 = torch.as_tensor(x1).requires_grad_(True)
torch_x2 = torch.as_tensor(x2).requires_grad_(True)

sig_x = jax.vmap(lambda in_: signature(in_, depth))(x1)


def combine_batch(path1, path2):
    def combine(_path1, _path2):
        return signature_combine(
            signature(_path1, depth),
            signature(_path2, depth),
            dim,
            depth)

    return jax.vmap(combine)(path1, path2)


combination = combine_batch(x1, x2)

torch_combination = signatory.signature_combine(signatory.signature(torch_x1, depth),
                                                signatory.signature(torch_x2, depth), dim, depth)
