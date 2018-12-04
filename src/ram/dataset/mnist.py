#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Modified from the original code for purposes of this project
"""tf.data.Dataset interface to the MNIST dataset."""
import gzip
import os
import shutil
import tempfile
import urllib.request
import struct
from typing import NamedTuple

import numpy as np
import tensorflow as tf

from .data import Data


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_images, unused
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))
        if rows != 28 or cols != 28:
            raise ValueError(
                'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                (f.name, rows, cols))


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_items, unused
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))


def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, \
            tf.gfile.Open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


def fetch_images(images_file):
    """Fetch MNIST images"""
    with open(images_file, 'rb') as fd:
        magic, size, h, w = struct.unpack('>iiii', fd.read(4 * 4))
        images = np.frombuffer(fd.read(), 'u1').reshape(size, h, w, 1)
    return images


def fetch_labels(labels_file):
    """Fetch MNIST labels data"""
    with open(labels_file, 'rb') as fd:
        magic, size = struct.unpack('>ii', fd.read(2 * 4))
        labels = np.frombuffer(fd.read(), 'u1')
    # cast from uint8 to int32
    return labels.astype(np.int32)


def normalize(images):
    """
    Normalize images to [0.0,1.0]
    """
    images = tf.cast(images, tf.float32)
    images /= 255.
    return images


class Data(NamedTuple):
    """represents a Tensorflow Dataset object and the number of samples in it"""
    dataset: tf.data.Dataset
    num_samples: int
    sample_inds: np.ndarray

    def __repr__(self) -> str:
        return f'<Data {self.dataset}, samples={self.num_samples}>'


def _dataset(directory, images_file, labels_file, num_samples=None):
    """Helper function that downloads (if necessary) and parses MNIST dataset.
    Instead of calling this directly, call the train or test functions.

    Parameters
    ----------
    directory : str
        Directory where raw MNIST files exist or should be downloaded
    images_file : str
        one of {'train-images-idx3-ubyte', 't10k-images-idx3-ubyte'}
    labels_file : str
        one of {'train-labels-idx1-ubyte', 't10k-labels-idx1-ubyte'}
    num_samples : int
        number of samples to return from. If None, uses all samples
        in dataset. Default is None.

    Returns
    -------
    ds : tf.data.Dataset
        that yields tuple pairs of:
            img : tf.float32
                images with shape (28, 28, 1) normalized to range [0.0, 1.0]
            lbl : tf.uint8
                labels with values 0-9
    samples : int
        number of samples in dataset
    """
    images_file = download(directory, images_file)
    labels_file = download(directory, labels_file)

    check_image_file_header(images_file)
    check_labels_file_header(labels_file)

    images = fetch_images(images_file)
    labels = fetch_labels(labels_file)

    if num_samples:
        if images.shape[0] < num_samples:
            raise ValueError(f'Number of samples in dataset, {images.shape[0]}, is less than'
                             f'number of samples to draw from that set, {num_samples}')

        sample_inds = np.random.choice(np.arange(images.shape[0]), size=(num_samples,))
        images = images[sample_inds, :, :, :]
        labels = labels[sample_inds]
    else:
        sample_inds = np.arange(len(labels))

    images = normalize(images)

    def gen():
        for image, label in zip(images, labels):
            yield image, label

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((28, 28, 1), []))
    data = Data(dataset=ds, num_samples=len(labels), sample_inds=sample_inds)
    return data


def train(directory, num_samples=None):
    """tf.data.Dataset object for MNIST training data."""
    return _dataset(directory=directory,
                    images_file='train-images-idx3-ubyte',
                    labels_file='train-labels-idx1-ubyte',
                    num_samples=num_samples)


def test(directory, num_samples=None):
    """tf.data.Dataset object for MNIST test data."""
    return _dataset(directory=directory,
                    images_file='t10k-images-idx3-ubyte',
                    labels_file='t10k-labels-idx1-ubyte',
                    num_samples=num_samples)
