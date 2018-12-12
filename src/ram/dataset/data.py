from typing import NamedTuple

import numpy as np
import tensorflow as tf


class Data(NamedTuple):
    """represents a Tensorflow Dataset object and the number of samples in it"""
    dataset: tf.data.Dataset
    num_samples: int
    sample_inds: np.ndarray

    def __repr__(self) -> str:
        return (f'ram.dataset.Data object with fields:\n\tdataset={self.dataset}\n\tnum_samples={self.num_samples}'
                f'\n\tsample_inds={self.sample_inds}')
