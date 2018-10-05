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
            length of one side of square patches in glimpes extracted by glimpse sensor.
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

    def glimpse(self, img, loc):
        """take a "glimpse" of a batch of images.
        Returns retina-like representation rho(img, loc)
        consisting of patches from each image.

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
        rho : tf.Tensor
            retina-like representation of k patches of increasing size
            and decreasing resolution, centered around location loc within
            image img
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

        return patches


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
