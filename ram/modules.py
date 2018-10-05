import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.nn import rnn_cell

from .decorators import define_scope

def _weight_variable(self, shape, name=None):
    """convenience function to return tf.Variable for weights
    Uses truncated normal to avoid values to far from mean of 0"""
    initial = tf.truncated_normal(shape, stddev=1.0 / shape[0])
    return tf.Variable(initial, name=name)


class GlimpseSensor:
    """glimpse sensor, returns retina-like representation
    of a region of an image x, given a location l to 'fixate'.
    """

    def __init__(self, g_w, k, s):
        """__init__ for GlimpseSensor

        Parameters
        ----------
        g_w : int
            length of one side of square patches in glimpes extracted by glimpse sensor.
            Default is 8.
        k : int
            number of patches that the retina encoding rho(x,l) extracts
            at location l from image x. Default is 3.
        s : int
            scaling factor, controls size of successive patches.
        """

        self.g_w = g_w
        self.k = k
        self.s = s

    def _denormalize(self, loc_normd, img_H, img_W):
        """

        Parameters
        ----------
        loc_normd
        img_H
        img_W

        Returns
        -------

        """

        return loc

    def glimpse(self, img, loc):
        """take a "glimpse" of a batch of images.
        Returns patches from each image.

        Parameters
        ----------
        img : tf.Tensor
            with shape (B, H, W, C). Minibatch of images.
        loc : tf.Tensor
            with shape (B, 2). Location of retina "fixation",
            in normalized co-ordinates where center of image is (0,0),
            upper left corner is (-1,-1), and lower right corner is (1,1).

        Returns
        -------

        """

        batch_size, img_H, img_W, C = img.shape
        # convert image co-ordinates from normalized to co-ordinates within
        # the specific size of the images
        loc[:, 0] = ((loc[:, 0] + 1)/2) + img_H
        loc[:, 1] = ((loc[:, 1] + 1) / 2) + img_W

        # compute top left corner of patches
        patch_x = loc[:, 0] - (img_H // 2)
        patch_y = loc[:, 1] - (img_W // 2)

        patches = []

        # process each image individually
        for ind in range(batch_size):
            img_patches = []
            one_img = img[ind, :, :, :]
            max_radius = self.g_w * (2 ** (self.k - 1))
            offset = max_radius

            # pad image with zeros
            one_img = tf.image.pad_to_bounding_box(one_img, offset, offset,
                                                   max_radius * 2 + mnist_size,
                                                   max_radius * 2 + mnist_size)

            for i in range(self.k):
                r = int(self.min_radius * (2 ** (i - 1)))

                d_raw = 2 * r
                d = tf.constant(d_raw, shape=[1])

                d = tf.tile(d, [2])

                loc_k = loc[k, :]
                adjusted_loc = offset + loc_k - r

                one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value,
                                                one_img.get_shape()[1].value))

                # crop image to (d x d)
                patch = tf.slice(one_img2, adjusted_loc, d)

                # resize cropped image to (sensorBandwidth x sensorBandwidth)
                patch = tf.image.resize_bilinear(tf.reshape(patch, (1, d_raw, d_raw, 1)), (sensorBandwidth, sensorBandwidth))
                patch = tf.reshape(patch, (sensorBandwidth, sensorBandwidth))
                img_patches.append(patch)

            patches.append(tf.stack(img_patches))

        patches = tf.stack(patches)

        # tf.summary.image(name='patches', tensor=zooms)

        return patches


def glimpse_network(self, loc):
    glimpse_input = self.glimpse_sensor(self.inputs, loc)

    glimpse_input = tf.reshape(glimpse_input,
                               (self.batch_size,
                                self.total_sensor_bandwidth))

    l_hl = self._weight_variable((2, self.hl_size))
    glimpse_hg = self._weight_variable((self.total_sensor_bandwidth, self.hg_size))

    hg = tf.nn.relu(tf.matmul(glimpse_input, glimpse_hg))
    hl = tf.nn.relu(tf.matmul(loc, l_hl))

    hg_g = self._weight_variable((hg_size, g_size))
    hl_g = self._weight_variable((hl_size, g_size))

    g = tf.nn.relu(tf.matmul(hg, hg_g) + tf.matmul(hl, hl_g))

    return g


def _get_next_input(self, output, i):
    mean_loc = tf.tanh(tf.matmul(output, h_l_out))
    mean_locs.append(mean_loc)

    sample_loc = mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd)

    sampled_locs.append(sample_loc)

    return get_glimpse(sample_loc)
