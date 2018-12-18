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


def prep(download_dir, val_size=None, random_seed=None,
         train_images_file='train-images-idx3-ubyte', train_labels_file='train-labels-idx1-ubyte',
         test_images_file='t10k-images-idx3-ubyte', test_labels_file='t10k-labels-idx1-ubyte',
         output_dir=None):
    """prepares MNIST dataset:
    downloads files, checks validity of downloaded data, and creates a validation
    set from a subset of the training set if 'val_size' is specified,
    then saves images and labels as numpy arrays.
    Datasets are stratified.

    Parameters
    ----------
    val_size : float or None, optional (default=0.05)
    random_seed : int
        value with which to seed random number generator when initialized.
        Default is None, in which case no seed is used.
    output_dir : str
        path to directory where .json file containing filenames split into
        train, validation, and test sets should be saved. Default is None,
        in which case file is saved in current directory.

    Other Parameters
    ----------------
    train_images_file : str
        Name of file with MNIST training images. Default is 'train-images-idx3-ubyte'.
    train_labels_file : str
        Name of file with MNIST training labels. Default is 'train-labels-idx1-ubyte'.
    test_images_file : str
        Name of file with MNIST test images. Default is 't10k-images-idx3-ubyte'.
    test_labels_file : str
        Name of file with MNIST test labels. Default is 't10k-labels-idx1-ubyte'.

    Returns
    -------
    filenames : dict
        with keys 'train', 'val', and 'test',
        and value for each key being absolute paths
        to files that contain that key's dataset
    """
    datasets = {}

    if val_size is not None:
        if val_size <= 0 or val_size >= 1.0:
            raise ValueError('val_size must be >0.0 and <1.0')

    train_images_file = download(download_dir, train_images_file)
    train_labels_file = download(download_dir, train_labels_file)

    check_image_file_header(train_images_file)
    check_labels_file_header(train_labels_file)

    datasets['train'] = {
        'images': fetch_images(train_images_file),
        'labels': fetch_labels(train_labels_file),
        'sample_inds': np.arange(len(fetch_labels(train_labels_file))),
    }

    test_images_file = download(download_dir, test_images_file)
    test_labels_file = download(download_dir, test_labels_file)

    check_image_file_header(test_images_file)
    check_labels_file_header(test_labels_file)

    datasets['test'] = {
        'images': fetch_images(test_images_file),
        'labels': fetch_labels(test_labels_file),
        'sample_inds': np.arange(len(fetch_labels(test_labels_file))),
    }

    if val_size:
        rng = np.random.RandomState()
        rng.seed(random_seed)
        val_ind = np.floor(val_size * len(datasets['train']['labels'])).astype(int)

        new_train_img = []
        new_train_lbl = []
        new_train_inds = []
        val_img = []
        val_lbl = []
        val_inds = []

        for digit_class in range(10):
            this_class_inds = np.where(datasets['train']['labels'] == digit_class)[0]
            rng.shuffle(this_class_inds)

            this_class_val_inds = this_class_inds[:val_ind]
            val_img.extend(datasets['train']['images'][this_class_val_inds])
            val_lbl.extend(datasets['train']['labels'][this_class_val_inds])
            val_inds.extend(datasets['train']['sample_inds'][this_class_val_inds])

            this_class_train_inds = this_class_inds[val_ind:]
            new_train_img.extend(datasets['train']['images'][this_class_train_inds])
            new_train_lbl.extend(datasets['train']['labels'][this_class_train_inds])
            new_train_inds.extend(datasets['train']['sample_inds'][this_class_train_inds])

        datasets['val'] = {
            'images': np.asarray(val_img),
            'labels': np.asarray(val_lbl),
            'sample_inds': np.asarray(val_inds),
        }

        datasets['train'] = {
            'images': np.asarray(new_train_img),
            'labels': np.asarray(new_train_lbl),
            'sample_inds': np.asarray(new_train_inds),
        }

    splits = ['train', 'test']
    if val_size:
        splits.append('val')
    paths_dict = {split: {} for split in splits}
    for split, data_dict in datasets.items():
        img_path = os.path.join(output_dir, f'MNIST_{split}_images.npy')
        np.save(img_path, data_dict['images'])
        paths_dict[split]['images'] = img_path

        lbl_path = os.path.join(output_dir, f'MNIST_{split}_labels.npy')
        np.save(lbl_path, data_dict['labels'])
        paths_dict[split]['labels'] = lbl_path

        ind_path = os.path.join(output_dir, f'MNIST_{split}_indices.npy')
        np.save(ind_path, data_dict['sample_inds'])
        paths_dict[split]['sample_inds'] = ind_path

    return paths_dict


def normalize(images):
    """
    Normalize images to [0.0,1.0]
    """
    images = tf.cast(images, tf.float32)
    images /= 255.
    return images


def _dataset(images_file, labels_file, sample_inds_file):
    """Helper function that packs a split of MNIST dataset into a Dataset object.
    Instead of calling this directly, call the `get_split` function.

    Parameters
    ----------
    images_file : str
        one of {'train-images-idx3-ubyte', 't10k-images-idx3-ubyte'}
    labels_file : str
        one of {'train-labels-idx1-ubyte', 't10k-labels-idx1-ubyte'}
    sample_inds_file : str

    Returns
    -------
    ram.data.Data named tuple with the following fields:
        dataset : tf.data.Dataset
            that yields tuple pairs of:
                img : tf.float32
                    images with shape (28, 28, 1) normalized to range [0.0, 1.0]
                lbl : tf.uint8
                    labels with values 0-9
                sample_inds : tf.int32
                    indices of samples from original MNIST dataset
        num_samples : int
            number of samples in dataset
        sample_inds : np.ndarray
            sample indices from original MNIST dataset (mostly important when
            a validation set is used, meaning the MNIST training set was split
            into training and validation subsets).
    """
    images = np.load(images_file)
    labels = np.load(labels_file)
    sample_inds = np.load(sample_inds_file)

    images = normalize(images)

    def gen():
        for image, label, batch_sample_inds in zip(images, labels, sample_inds):
            yield image, label, batch_sample_inds

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32, tf.int32), ((28, 28, 1), [], []))
    data = Data(dataset=ds, num_samples=len(labels), sample_inds=sample_inds)
    return data


def get_split(paths_dict, setname='train'):
    """get data set from files

    Parameters
    ----------
    paths_dict : dict
        containing full paths to files containing images, labels, and sample indices as
        numpy arrays. Will have keys 'train', 'test', and optionally 'val'.
        returned by ram.datasets.mnist.prep function
    setname : str or list
        One of {'train', 'val', 'test'}, or a list with some combination of the three strings.
        'val' returns the MNIST training data split into train and validation sets.

    Returns
    -------
    dataset : ram.dataset.Dataset
        NamedTuple with two fields:
            dataset : tensorflow.data.Dataset
                images loaded from file names in the split specified
            num_samples : int
                number of samples in dataset
            sample_inds : np.ndarray
                indices of samples used for dataset

    Examples
    --------
    >>> from ram.dataset.mnist import get_split
    >>> train_data, val_data = get_split(paths_dict=paths_dict, setname=['train', 'val'])
    >>> test_data = get_split(paths_dict=paths_dict, setname='test')
    >>> ram.Trainer(config=config, train_data=train_data, val_data=val_data)
    """
    if type(setname) == str:
        setname = [setname]  # so we can iterate over list (not over str which would cause an error)
    valid_setnames = {'train', 'val', 'test'}
    if not all(name in valid_setnames for name in setname):
        raise ValueError(f"invalid dataset name in setname: '{setname}'.\n"
                         "Must be 'train', 'val', or 'test'.")
    data_obj_list = []
    for name in setname:
        data_obj_list.append(_dataset(images_file=paths_dict[name]['images'],
                                      labels_file=paths_dict[name]['labels'],
                                      sample_inds_file=paths_dict[name]['sample_inds'])
                             )
    return tuple(data_obj_list[:])
