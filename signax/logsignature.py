from functools import partial
from typing import List

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=2)
def mult_inner(
    A: List[jnp.ndarray],
    B: List[jnp.ndarray],
    depth_index: int,
):
    """
    Let `depth_index` = n

    this function returns
        $sum_{i=1}^n A_i x B_{n - i}$


    Note this is hard to convert to `jax.lax.fori_loop`.
    I don't know if it's possible. Several attempts but
    `TracerIntergerConversionError` is encountered because
    getting index of a list (it's okay to get index of ndarray
    but not for lists)
    """

    return sum(
        [
            A[i][..., None] * B[depth_index - i - 1][None, ...]
            for i in range(depth_index)
        ]
    )


@partial(jax.jit, static_argnums=3)
def mult_partial(
    input1: List[jnp.ndarray],
    input2: List[jnp.ndarray],
    scalar_term_value: float,
    top_terms_to_skip: int,
):
    """Sort of multiplication in the tensor algerbra

    `input1` assumed scalar value
    `input2` assume scalar value zero

    return `input1` x `input2` for some of its terms

    many terms are left unchanged

    This corresponds to compute one parenthesis in Equation (10)
    (see iisignature paper)
    """

    depth = len(input1)
    for depth_index in reversed(range(depth - top_terms_to_skip)):
        input1[depth_index] = jnp.zeros_like(input1[depth_index])
        input1[depth_index] = mult_inner(input1, input2, depth_index)
        input1[depth_index] = (
            input1[depth_index] + input2[depth_index] * scalar_term_value
        )

    return input1


def log_coef_at_depth(depth: int):
    """Note that reciprocals is an array [1/2, 1/3, ...]"""
    sign = -1.0 if depth % 2 == 0 else 1.0
    return sign / (depth + 2)


@jax.jit
def log(input: List[jnp.ndarray]):
    """This follows Equation (10) of iisignature paper"""

    depth = len(input)
    if depth == 1:
        return input

    output = [jnp.zeros_like(x) for x in input]
    output[0] = input[0] * log_coef_at_depth(depth - 2)

    for depth_index in reversed(range(depth - 2)):
        output = mult_partial(
            output,
            input,
            scalar_term_value=log_coef_at_depth(depth_index),
            top_terms_to_skip=depth_index + 1,
        )

    output = mult_partial(
        output,
        input,
        scalar_term_value=1.0,
        top_terms_to_skip=0,
    )

    return output


def signature_to_logsignature(
    signature: List[jnp.ndarray],
    dim: int,
    depth: int,
):

    # compute Lyndon words given `depth` and `dim`
    expanded_logsignature = log(signature)

    # compress using the information of Lyndon words

    return expanded_logsignature


if __name__ == "__main__":

    import signatory
    import torch

    dim = 3

    input1 = [
        jnp.ones((dim,)),
        jnp.ones((dim, dim)),
        jnp.ones((dim, dim, dim)),
    ]

    output = log(input1)
    print(output)

    signature = torch.ones((1, dim + dim * dim + dim * dim * dim))
    log_signature = signatory.signature_to_logsignature(
        signature=signature, depth=3, channels=dim
    )
    print(log_signature)
