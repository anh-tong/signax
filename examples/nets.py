from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from signax.module import SignatureTransform
from signax.signature import signature, signature_combine
from signax.utils import flatten


def _make_convs(input_size: int, layer_sizes, kernel_size, *, key):
    keys = jrandom.split(key, num=len(layer_sizes))
    convs = []

    first_conv = eqx.nn.Conv1d(
        in_channels=input_size,
        out_channels=layer_sizes[0],
        kernel_size=kernel_size,
        key=keys[0],
    )
    convs += [first_conv]
    last_conv_size = layer_sizes[0]
    for i, layer_size in enumerate(layer_sizes[1:]):
        conv = eqx.nn.Conv1d(
            in_channels=last_conv_size,
            out_channels=layer_size,
            kernel_size=1,
            key=keys[i + 1],
        )
        convs += [conv]
        last_conv_size = layer_size
    return convs


class Augment(eqx.nn.Sequential):
    """A stack of Conv1D, first Conv1D has kernel_size as input
    The remaining Conv1D has kernel_size = 1

    This allows to add original input and time dimension to the output
    """

    activation: Callable
    include_original: bool
    include_time: bool
    kernel_size: int

    def __init__(
        self,
        layers,
        include_original=True,
        include_time=True,
        kernel_size=3,
        activation=jax.nn.relu,
    ):
        self.layers = layers
        self.include_original = include_original
        self.include_time = include_time
        self.kernel_size = kernel_size
        self.activation = activation

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ):
        """x size (length, dim)"""
        length, _ = x.shape
        ret = []
        if self.include_original:
            start_index = self.kernel_size - 1
            truncated_x = x[start_index:]
            ret.append(truncated_x)
        if self.include_time:
            time = jnp.linspace(0, 1, length - self.kernel_size + 1)
            time = time[:, None]
            ret.append(time)
        augmented_x = self.layers[0](x.transpose())
        for layer in self.layers[1:]:
            augmented_x = self.activation(augmented_x)
            augmented_x = layer(augmented_x)
        ret.append(augmented_x.transpose())
        return jnp.concatenate(ret, axis=-1)


class Window(eqx.Module):
    """
    JAX version of
    https://github.com/patrick-kidger/Deep-Signature-Transforms/blob/master/packages/candle/recurrent.py

    Here we directly apply signature transform
    """

    stride: int
    window_len: int
    signature_depth: int

    def __init__(self, stride, window_len, signature_depth=2) -> None:
        self.stride = stride
        self.window_len = window_len
        self.signature_depth = signature_depth

    def __call__(self, x, *, key):
        """
        Example:

        >>> x = [0., 1., 2., 3., 4., 5., 6.]
        >>> window = Window(stride=2, window_len=3)
        >>> window(x)
        Output: [[0., 1., 2.], [2., 3., 4.], [4., 5., 6.]]

        Args:
            x : size (path_length, dim)

        Returns:
            size (n, window_len, dim)
            where n = floor((path_len - window_len)/stride)
        """
        path_length, dim = x.shape
        n_strides = int((path_length - self.window_len) / self.stride)
        index = jnp.arange(n_strides + 1) * self.stride

        def _f(carry, i):
            """Must use dynamic_slice here
            because jax.numpy slice x[i:(i + window_len)] does not work with
            dynamic index
            """
            return carry, jax.lax.dynamic_slice(
                x,
                (i, 0),
                (self.window_len, dim),
            )

        _, output = jax.lax.scan(f=_f, init=None, xs=index)

        # output is a tensor algebra which is a list of `jnp.ndarray`
        # size of output: [(n, dim), (n, dim, dim,), (n, dim, dim, dim), ...]

        def _signature(x):
            ta = signature(x, self.signature_depth)
            return flatten(ta)

        output = jax.vmap(_signature)(output)

        return output


class WindowAdjusted(eqx.Module):
    length: int
    adjusted_length: int
    signature_depth: int

    def __init__(self, length, adjusted_length, signature_depth=2) -> None:
        assert adjusted_length > 0
        self.length = length
        self.adjusted_length = adjusted_length
        self.signature_depth = signature_depth

    def __call__(self, x, *, key=None):
        """x size: (path_len, dim)"""
        path_length, dim = x.shape[0], x.shape[1]

        # this can miss the last index
        index = jnp.arange(
            start=self.length, stop=path_length, step=self.adjusted_length
        )
        init = signature(x[: self.length], self.signature_depth)

        def f(carry, i):
            current_x = jax.lax.dynamic_slice(
                x,
                start_indices=(i - 1, 0),
                slice_sizes=(self.adjusted_length + 1, dim),
            )
            sig = signature(current_x, self.signature_depth)
            out = signature_combine(carry, sig)
            return out, flatten(out)

        _, ret = jax.lax.scan(f, init=init, xs=index)

        return ret


class RecurrentNet(eqx.Module):
    """
    Given input [x1, x2, ..., x_n]
    y_{t+1}, h_{t+1} = f(h_t, x_t)
    """

    memory_size: int
    output_size: int
    mlp: eqx.nn.MLP
    intermediate_outputs: bool

    def __init__(
        self,
        input_size,
        memory_size,
        output_size,
        hidden_size=32,
        depth=2,
        intermediate_outputs=True,
        *,
        key,
    ):
        self.mlp = eqx.nn.MLP(
            in_size=input_size + memory_size,
            width_size=hidden_size,
            depth=depth,
            out_size=memory_size + output_size,
            key=key,
        )

        self.memory_size = memory_size
        self.output_size = output_size
        self.intermediate_outputs = intermediate_outputs

    def __call__(self, input, *, key):
        """
        Args:
            input: size (seq_length, dim)
        Returns:
            output: size (seq_lenth, dim)
        """
        memory = jnp.zeros((self.memory_size))

        def f(carry, inp):
            # carry = memory
            x = jnp.concatenate([carry, inp])
            out = self.mlp(x)
            carry, out = jnp.split(
                out,
                [
                    self.memory_size,
                ],
            )
            return carry, out

        _, output = jax.lax.scan(f, memory, input)
        return output


class Sigmoid(eqx.Module):
    def __call__(self, x, *, key):
        return jax.nn.sigmoid(x)


def signature_dim(n_channels, depth):
    return sum([n_channels ** (i + 1) for i in range(depth)])


def create_simple_net(
    dim=1,
    signature_depth=4,
    augment_layer_size=(8, 8, 2),
    augmented_kernel_size=1,
    augmented_include_original=True,
    augmented_include_time=True,
    mlp_width=32,
    mlp_depth=2,
    output_size=1,
    final_activation=jax.nn.sigmoid,
    *,
    key,
):
    augment_key, mlp_key = jrandom.split(key)

    # create Convolutional augmented layers
    convs = _make_convs(
        input_size=dim,
        layer_sizes=augment_layer_size,
        kernel_size=augmented_kernel_size,
        key=augment_key,
    )
    augment = Augment(
        layers=convs,
        include_original=augmented_include_original,
        include_time=augmented_include_time,
        kernel_size=augmented_kernel_size,
    )

    signature = SignatureTransform(depth=signature_depth)

    # calculate output dimension of Agument
    last_dim = augment_layer_size[-1]
    if augmented_include_original:
        last_dim += dim
    if augmented_include_time:
        last_dim += 1
    # the output dimension of signature
    mlp_input_dim = signature_dim(n_channels=last_dim, depth=signature_depth)
    mlp = eqx.nn.MLP(
        in_size=mlp_input_dim,
        width_size=mlp_width,
        depth=mlp_depth,
        out_size=output_size,
        final_activation=final_activation,
        key=mlp_key,
    )
    layers = [augment, signature, mlp]

    return eqx.nn.Sequential(layers=layers)


def create_deep_recurrence(
    dim=2,
    signature_depth=4,
    lengths=(5, 5, 10),
    strides=(1, 1, 5),
    memory_sizes=(8, 8, 8),
    output_sizes=(8, 8, 1),
    *,
    key,
):
    layers = []
    current_n_channels = dim
    zipped = zip(lengths, strides, memory_sizes, output_sizes)
    for i, (length, stride, memory_size, output_size) in enumerate(zipped):
        key, _ = jrandom.split(key)

        layers.append(
            Window(
                stride=stride,
                window_len=length,
                signature_depth=signature_depth,
            )
        )
        # input of recurrent model is the ouput dim of signature
        input_size = signature_dim(current_n_channels, signature_depth)
        layers.append(
            RecurrentNet(
                input_size=input_size,
                memory_size=memory_size,
                output_size=output_size,
                intermediate_outputs=i != len(lengths) - 1,
                key=key,
            )
        )
        # update input dim
        current_n_channels = output_size

        if i == len(lengths) - 1:
            layers.append(Sigmoid())

    model = eqx.nn.Sequential(layers)
    return model


def create_generative_net(dim, *, key):
    augment_in_key, augment_out_key = jrandom.split(key, num=2)
    convs_in = _make_convs(
        input_size=dim,
        layer_sizes=(8, 8, 2),
        kernel_size=1,
        key=augment_in_key,
    )
    augment_in = Augment(
        layers=convs_in,
        include_original=True,
        include_time=False,
        kernel_size=1,
    )

    convs_out = _make_convs(
        input_size=84,
        layer_sizes=(1,),
        kernel_size=1,
        key=augment_out_key,
    )
    augment_out = Augment(
        layers=convs_out,
        include_original=False,
        include_time=False,
        kernel_size=1,
    )

    layers = (augment_in, WindowAdjusted(2, 1, 3), augment_out)
    return eqx.nn.Sequential(layers)


if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    key = jrandom.PRNGKey(0)
    model = create_simple_net(dim=2, key=key)

    x = jnp.ones((100, 2))
    output = model(x)
