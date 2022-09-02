import jax
import numpy as np
import signatory
import torch

from signax.signature import signature
from signax.utils import flatten, term_at

np.random.seed(0)

n_batch = 4
length = 3
dim = 2
x = np.random.rand(n_batch, length, dim)

depth = 3

torch_x = torch.as_tensor(x).requires_grad_(True)

sig_x = jax.vmap(lambda in_: signature(in_, depth))(x)
flattened_sig_x = jax.vmap(flatten)(sig_x)
sig_torch_x = signatory.signature(torch_x, depth)

sig_at_1 = jax.vmap(lambda x: term_at(x, dim, 1))(flattened_sig_x)
