"""modules for RAM model, from [1]_.
Based on two implementations:
https://github.com/seann999/tensorflow_mnist_ram
https://github.com/kevinzakka/recurrent-visual-attention

.. [1] Mnih, Volodymyr, Nicolas Heess, and Alex Graves.
   "Recurrent models of visual attention."
   Advances in neural information processing systems. 2014.
"""

import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.nn import rnn_cell


class GlimpseSensor(tf.keras.Model):
    """glimpse sensor, returns retina-like representation rho
    of a region of an image x, given a location l to 'fixate'.
    """

    def __init__(self, g_w=8, k=3, s=2):
        """__init__ for GlimpseSensor

        Parameters
        ----------
        g_w : int
            length of one side of square patches in glimpses extracted by glimpse sensor.
            Default is 8.
        k : int
            number of patches that the retina encoding rho(x,l) extracts
            at location l from image x. Default is 3.
        s : int
            scaling factor, controls size of successive patches. Default is 2.
        """
        super(GlimpseSensor, self).__init__()
        self.g_w = g_w
        self.k = k
        self.s = s

    def glimpse(self, images, loc_normd):
        """take a "glimpse" of a batch of images.
        Returns retina-like representation rho(img, loc)
        consisting of patches from each image.

        Parameters
        ----------
        images : tf.Tensor
            with shape (B, H, W, C). Minibatch of images.
        loc_normd : tf.Tensor
            with shape (B, 2). Location of retina "fixation",
            in normalized co-ordinates where center of image is (0,0),
            upper left corner is (-1,-1), and lower right corner is (1,1).

        Returns
        -------
        rho : tf.Tensor
            with shape (B, k, g_w, g_w,
            retina-like representation of k patches of increasing size
            and decreasing resolution, centered around location loc within
            image img
        """
        batch_size, img_H, img_W, C = images.shape
        # convert image co-ordinates from normalized to co-ordinates within
        # the specific size of the images
        loc_0 = ((loc_normd[:, 0] + 1) / 2) * img_H
        loc_0 = tf.cast(tf.round(loc_0), tf.int32)
        loc_1 = ((loc_normd[:, 1] + 1) / 2) * img_W
        loc_1 = tf.cast(tf.round(loc_1), tf.int32)
        loc = tf.stack([loc_0, loc_1], axis=1)

        rho = []
        for ind in range(batch_size):
            img = images[ind, :, :, :]
            patches = []
            for patch_num in range(self.k):
                size = self.g_w * (self.s ** patch_num)

                # pad image with zeros
                # (in case patch at current location extends beyond edges of image)
                img_padded = tf.image.pad_to_bounding_box(img,
                                                          offset_height=size,
                                                          offset_width=size,
                                                          target_height=(size * 2) + img_H,
                                                          target_width=(size * 2) + img_W)

                # compute top left corner of patch
                patch_x = loc[ind, 0] - (size // 2) + size
                patch_y = loc[ind, 1] - (size // 2) + size

                patch = tf.slice(img_padded,
                                 begin=tf.stack([patch_x, patch_y, 0]),
                                 size=tf.stack([size, size, C])
                                 )
                if size == self.g_w:
                    # convert to float32 to be consistent with
                    # tensors output after resizing
                    patch = tf.cast(patch, dtype=tf.float32)
                else:
                    # resize cropped image to (size x size)
                    patch = tf.image.resize_images(patch, size=(self.g_w, self.g_w))
                patches.append(patch)

            rho.append(patches)

        rho = tf.stack(rho)
        return rho


class GlimpseNetwork(tf.keras.Model):
    """Network that maps retina representation rho and
    location loc into a hidden space; defines a trainable
    bandwidth-limited sensor that produces the glimpse
    representation g_t

    Attributes
    ----------
    self.forward : forward pass through network, accepts image and
        location tensors and returns tensor of glimpse representations g_t
    """

    def __init__(self, g_w, k, s, c, h_g_units=128, h_l_units=128, h_gt_units=256):
        """__init__ function for GlimpseNetwork

        Parameters
        ----------
        g_w : int
            size of square patches extracted by glimpse sensor.
        k : int
            number of patches to extract per glimpse.
        s : int
            scaling factor that controls size of successive patches.
        c : int
            number of channels in each image.
        h_g_units : int
            number of units in fully-connected layer for retina-like representation rho.
            Default is 128.
        h_l_units : int
            number of units in fully-connected layer for location l.
            Default is 128.
        h_gt_units : int
            number of units in fully-connected layer for output g_t. This must be equal
            to the number of hidden units in the core network. Default is 256.
        """
        super(GlimpseNetwork, self).__init__()
        self.glimpse_sensor = GlimpseNetwork(g_w=g_w, k=k, s=s)
        self.theta_g_0 = tf.keras.layers.Dense(units=h_g_units, activation='ReLu')
        self.theta_g_1 = tf.keras.layers.Dense(units=h_l_units, activation='ReLu')
        self.theta_g_2 = tf.keras.layers.Dense(units=h_gt_units, activation='ReLu')

    def forward(self, img, loc):
        """

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
        g_t : tf.Tensor
            glimpse representation, output by glimpse network
        """

        rho = self.glimpse_sensor.glimpse(img, loc)
        h_g = self.theta_g_0(rho)
        h_l = self.theta_g_1(loc)
        g_t = self.theta_g_2(h_g + h_l)
        return g_t


def _get_next_input(self, output, i):
    mean_loc = tf.tanh(tf.matmul(output, h_l_out))
    mean_locs.append(mean_loc)

    sample_loc = mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd)

    sampled_locs.append(sample_loc)

    return get_glimpse(sample_loc)
