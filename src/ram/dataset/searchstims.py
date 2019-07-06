"""tf.data.Dataset interface to searchstims dataset."""
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .data import Data


def split(stim_json_filename,
          stim_type,
          test_size=None,
          val_size=None,
          train_size=None,
          random_seed=None,
          output_dir=None):
    """splits searchstims dataset into training, validation, and test sets,
    then saves filenames for each set in text files.
    Datasets are stratified, so that the number of examples of each "set size"
    of visual search stimuli is proportional to the number in the total training set.

    Parameters
    ----------
    stim_json_filename : str
        path to .json file containing paths to image files, i.e. visual search stimuli,
        and related metadata
    stim_type : str, list
        visual search stimulus type(s) to use. Will be key(s) in dictionary loaded
        from stim_json_filename.
    test_size : float or None, optional (default=0.25)
        If float, should be between 0.0 and 1.0, and represent
        the proportion of the dataset to include in the test split.
        If None, set to (1.0 - train_size + validation_size).
        If train_size is also None, set to default of 0.25,
        and train_size defaults to 1.0 - test_size.
    val_size : float or None, optional (default=0.05)
    train_size : float or None, optional
    random_seed : int
        value with which to seed random number generator when initialized.
        Default is None, in which case no seed is used.
    output_dir : str
        path to directory where .json files containing filenames split into
        train, validation, and test sets should be saved. Default is None,
        in which case file is saved in current directory.

    Returns
    -------
    split_filenames : dict
        with keys 'train', 'val', and 'test',
        and value for each key being list of absolute paths
        to files in that key's dataset
    """
    if train_size is not None:
        if train_size <= 0 or train_size >= 1.0:
            raise ValueError('train_size must be >0.0 and <1.0')
    if val_size is not None:
        if val_size <= 0 or val_size >= 1.0:
            raise ValueError('val_size must be >0.0 and <1.0')
    if test_size is not None:
        if test_size <= 0 or test_size >= 1.0:
            raise ValueError('test_size must be >0.0 and <1.0')

    if val_size is None:
        val_size = 0.05

    if test_size is None:
        if train_size is not None:
            test_size = 1.0 - train_size - val_size
        else:
            test_size = 0.25
            train_size = 1.0 - test_size - val_size
    else:  # if test_size is not None
        if train_size is None:
            train_size = 1.0 - test_size - val_size

    train_size = np.around(train_size, decimals=2)
    test_size = np.around(test_size, decimals=2)
    val_size = np.around(val_size, decimals=2)

    size_sum = sum([train_size, test_size, val_size])
    if size_sum > 1.0:
        raise ValueError('sum of train_size, test_size, and val_size cannot be greater than 1.0, '
                         f' but was {size_sum}.\n'
                         f'Train size is {train_size}, test_size is {test_size}, and val_size is {val_size}.')

    rng = np.random.RandomState()
    rng.seed(random_seed)

    train_fnames = []
    val_fnames = []
    test_fnames = []

    with open(stim_json_filename) as json_file:
        searchstims_dict = json.load(json_file)
    stim_dict = searchstims_dict[stim_type]
    for set_size, present_absent_dict in stim_dict.items():
        for present_or_absent, stim_info_list in present_absent_dict.items():
            fname_list = [stim_info_dict['filename'] for stim_info_dict in stim_info_list]
            fname_list = [
                str(Path(stim_json_filename).parent.joinpath(fname))
                for fname in fname_list
            ]
            num_fname = len(fname_list)
            num_train = np.ceil(train_size * num_fname).astype(int)
            num_test = np.floor(test_size * num_fname).astype(int)
            num_val = np.floor(val_size * num_fname).astype(int)
            num_samples = num_train + num_val + num_test
            if num_samples != num_fname:
                raise ValueError("split failed; total number of training, test, and validation "
                                 f"samples, {num_samples}, did not equal to total number of files, {num_fname}")
            fname_arr = np.asarray(fname_list)
            shuffle_inds = rng.permutation(np.arange(num_fname))
            train_fnames.extend(fname_arr[shuffle_inds[:num_train]])
            test_fnames.extend(fname_arr[shuffle_inds[num_train:num_train + num_test]])
            val_fnames.extend(fname_arr[shuffle_inds[num_train + num_test:num_train+num_test + num_val]])
    # need to shuffle after we get filenames out of target present and target absent sets
    # so each batch has both classes
    for fnames_list in [train_fnames, test_fnames, val_fnames]:
        # shuffle list in place with our random number generator
        rng.shuffle(fnames_list)
    paths_dict = {}
    for split, filenames in zip(['train', 'val', 'test'],
                                [train_fnames, val_fnames, test_fnames]):
        split_json_filename = split + '_filename.json'
        if output_dir:
            split_json_filename = os.path.join(output_dir, split_json_filename)
        split_json = json.dumps(filenames, indent=4)
        with open(split_json_filename, 'w') as json_output:
            print(split_json, file=json_output)
        paths_dict[split] = split_json_filename
    return paths_dict


def prep(download_dir, stim_type, train_size=None, val_size=None, random_seed=None,
         output_dir=None):
    """prepare searchstims dataset

    Parameters
    ----------
    download_dir
    train_size
    val_size
    random_seed
    output_dir

    Returns
    -------
    paths_dict : dict
        with keys 'train', 'test', and 'val', and the corresponding values being
        paths to .json files containing a list of filenames, the files being images
        that belong to each split of the original data set
    """
    stim_json_filename = os.path.join(download_dir, f'{stim_type}.json')
    return split(stim_json_filename, stim_type, val_size=val_size, train_size=train_size,
                 random_seed=random_seed, output_dir=output_dir)


def normalize(images):
    """
    Normalize images to [0.0,1.0]
    """
    images = tf.cast(images, tf.float32)
    images = ((images - 255. / 2.) / 255.)
    return images


def load(x):
    x = tf.read_file(x)
    x = tf.image.decode_png(x)
    x = tf.cast(x, tf.float32)
    return x


def preprocess(x):
    return normalize(x)


def _generic_dataset(x, y, sample_inds, preprocess_func):
    if type(x) == list:
        x = np.asarray(x)
    x_ds = tf.data.Dataset.from_tensor_slices(x)
    x_ds = x_ds.map(load, num_parallel_calls=24)
    x_ds = x_ds.map(preprocess_func, num_parallel_calls=24)
    y_ds = tf.data.Dataset.from_tensor_slices(y)
    sample_inds = tf.data.Dataset.from_tensor_slices(sample_inds)

    ds = tf.data.Dataset.zip((x_ds, y_ds, sample_inds))
    return ds


def _make_dataset(filename_list, shard=False, num_shards=None):
    """Returns Tensorflow dataset objects with training, validation,
    and/or test images and "labels" (whether target is present or absent)

    Parameters
    ----------
    filename_list : list
        list of filenames of visual search images

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
    labels = []
    print('loading images')
    for filename in tqdm(filename_list):
        if 'present' in filename:
            labels.append(1)
        elif 'absent' in filename:
            labels.append(0)
        else:
            raise ValueError(f"Filename does not contain 'present' or 'absent': {filename}")

    labels = np.asarray(labels)
    # can use sample_inds to index into filenames_list when needing to recover images
    # e.g. for plotting plot of a gaze on a specific image
    sample_inds = np.arange(len(labels))

    tf_ds = _generic_dataset(x=filename_list,
                             y=labels,
                             sample_inds=sample_inds,
                             preprocess_func=preprocess)

    dataset = Data(dataset=tf_ds, num_samples=len(labels), sample_inds=sample_inds)
    return dataset


def get_split(paths_dict, setname='train'):
    """get data set from files

    Parameters
    ----------
    paths_dict : dict
        containing full paths to .json files that have list of images that should be in each split
        of dataset.
    setname : str or list
        One of {'train', 'val', 'test'}, or a list with some combination of the three strings.

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
    >>> from ram.dataset.searchstims import prep, get_split
    >>> paths_dict = prep(download_dir='/home/you/data/', val_size=0.1)
    >>> train_data, val_data = get_split(paths_dict=paths_dict, setname=['train', 'val'])
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
        with open(paths_dict[name]) as split_json:
            filename_list = json.load(split_json)
        data_obj_list.append(_make_dataset(filename_list))

    if len(data_obj_list) == 1:
        return data_obj_list[0]
    else:
        return tuple(data_obj_list[:])
