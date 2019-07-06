"""utility functions"""
import os
import numpy as np


def get_example_arrs(suffix='_epoch_test', glimpses_filename=None, fixations_filename=None,
                     locations_filename=None, images_filename=None, labels_filename=None,
                     predictions_filename=None, examples_dir=None, num_examples_to_show=None):
    """load saved examples and return as arrays

    Parameters
    ----------
    suffix : str
        suffix applied to each saved array file.
        Can be epoch from which examples were saved, e.g. '_epoch_100'.
        If plotting test examples, use the string '_epoch_test' instead of an epoch number.
        Examples saved by runner should have suffix '_from_run'.
    glimpses_filename : str
        filename with glimpses from saved examples. Default is None, in which case the default
        'glimpses{suffix}.npy' is used.
    fixations_filename : str
        filename with fixations (pixel co-ordinates of glimpse location) from saved examples.
        Default is None, in which case the default 'fixations{suffix}.npy' is used.
    locations_filename : str
        filename with locations (co-ordinates of glimpse location, normalized to [-1, 1])
        from saved examples. Default is None, in which case the default 'locations{suffix}.npy' is used.
    images_filename : str
        filename with images from from saved examples. Default is None, in which case the default
        'images{suffix}.npy' is used.
    labels_filename : str
        filename with labels from from saved examples. Default is None, in which case the default
        'labels{suffix}.npy' is used.
    predictions_filename : str
        filename with predictions from saved examples. Default is None, in which case the default
        'predictions{suffix}.npy' is used.
    examples_dir : str
        path to directory where examples are svaed
    num_examples_to_show : int
        number of examples to show. Default is None, in which case *all* examples will be returned.

    Returns
    -------
    glimpses : numpy.ndarray
        containing glimpses extracted by RAM
    fixations : numpy.ndarray
        containing fixations of RAM
    images : numpy.ndarray
        containing images shown to RAM
    predictions : numpy.ndarray
        containing predictions made by RAM
    """
    if glimpses_filename is None:
        glimpses_filename = f"glimpses{suffix}.npy"
    if fixations_filename is None:
        fixations_filename = f"fixations{suffix}.npy"
    if locations_filename is None:
        locations_filename = f"locations{suffix}.npy"
    if images_filename is None:
        images_filename = f"images{suffix}.npy"
    if labels_filename is None:
        labels_filename = f"labels{suffix}.npy"
    if predictions_filename is None:
        predictions_filename = f"predictions{suffix}.npy"

    if examples_dir:
        glimpses_filename = os.path.join(examples_dir, glimpses_filename)
        fixations_filename = os.path.join(examples_dir, fixations_filename)
        locations_filename = os.path.join(examples_dir, fixations_filename)
        images_filename = os.path.join(examples_dir, images_filename)
        labels_filename = os.path.join(examples_dir, labels_filename)
        predictions_filename = os.path.join(examples_dir, predictions_filename)

    glimpses = np.load(glimpses_filename)
    fixations = np.load(fixations_filename)
    locations = np.load(locations_filename)
    images = np.load(images_filename)
    labels = np.load(labels_filename)
    predictions = np.load(predictions_filename)

    if num_examples_to_show:
        glimpses = glimpses[:num_examples_to_show]
        fixations = fixations[:num_examples_to_show]
        locations = locations[:num_examples_to_show]
        images = images[:num_examples_to_show]
        labels = labels[:num_examples_to_show]
        predictions = predictions[:num_examples_to_show]

    return glimpses, fixations, locations, images, labels, predictions


def denormalize_loc(locations, img_height, img_width):
    """convert locations from normalized co-ordinates in range [0,1] back into pixel co-ordinates

    Parameters
    ----------
    locations : numpy.ndarray
        with two columns (y, x) (not (x, y), because of how array indexing works)
    img_height : int
        image height in pixels
    img_width : int
        image width in pixels

    Returns
    -------
    locations : numpy.ndarray
        with values converted from range [0, 1] to range [0, img_height] in first column
        and range [0, img_width] in second column.
    """
    for glimpse in range(locations.shape[0]):
        locations[glimpse, :, 0] = ((locations[glimpse, :, 0] + 1) / 2) * img_height
        locations[glimpse, :, 1] = ((locations[glimpse, :, 1] + 1) / 2) * img_width
    return locations


def denormalize_img(images):
    """convert images back to RGB [0 to 255] from normalized image shown to network

    Parameters
    ----------
    images : numpy.ndarray
        array of images with dimensions (number of images, height, width) with values
        zero centered and on Z scale

    Returns
    -------
    images : numpy.ndarray
        array of images with dimensions (number of images, height, width) with values
        ranging from 0 to 255
    """
    images = ((images * 255) + 255. / 2.)
    return images


def compute_patch_sizes(g_w=8, k=3, s=2):
    """computes size of patches that make up a glimpse, given size of first patch, scaling parameter, and
    number of patches

    Parameters
    ----------
    g_w : int
        width of a single patch, in pixels. Default is 8. Patches are square, so height == width
    k : int
        number of "patches" (concentric squares) in a glimpse. Default is 3.
    s : int
        scaling term. Default is 2. The size of each consecutive concentric patch is
        g_w * (s ** k_n), where k_n is some integer (0, 1, ..., k-1).

    Returns
    -------
    patch_sizes : list
        of length k. For the default values of g_w, k, and s, patch_sizes is [8, 16, 32].
    """
    patch_sizes = []
    for patch_num in range(k):
        size = g_w * (s ** patch_num)
        size = (size, size)
        patch_sizes.append(size)
    return patch_sizes
