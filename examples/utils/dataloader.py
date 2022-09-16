from typing import Sequence

import jax.random as jrandom
from jax import numpy as jnp


class DataLoader:
    def __init__(
        self, data: Sequence[jnp.ndarray], batch_size: int = 1, random_key=None
    ):
        self.dataset_size = data[0].shape[0]
        assert all(datum.shape[0] == self.dataset_size for datum in data)

        self.data = data
        self.batch_size = batch_size
        self.random_key = random_key

    def __iter__(self):
        return DataLoaderIterator(self)


class DataLoaderIterator:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader

        self.indices = jnp.arange(self.dataloader.dataset_size)
        if self.dataloader.random_key is not None:
            self.indices = jrandom.permutation(
                self.dataloader.random_key,
                self.indices,
            )

        self.cur_index = 0

    def __next__(self):
        end = self.cur_index + self.dataloader.batch_size
        if end > self.dataloader.dataset_size:
            raise StopIteration()
        start = self.cur_index
        indices = self.indices[start:end]
        result = (datum[indices] for datum in self.dataloader.data)

        self.cur_index += 1

        return result
