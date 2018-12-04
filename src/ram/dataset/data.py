from typing import NamedTuple

import numpy as np
import tensorflow as tf


class Data(NamedTuple):
    """represents a Tensorflow Dataset object and the number of samples in it"""
    dataset: tf.data.Dataset
    num_samples: int
    sample_inds: np.ndarray

    def __repr__(self) -> str:
        return f'<Data {self.dataset}, samples={self.num_samples}>'
