"""modules for RAM model, from [1]_.
Based on two implementations:
https://github.com/seann999/tensorflow_mnist_ram
https://github.com/kevinzakka/recurrent-visual-attention

.. [1] Mnih, Volodymyr, Nicolas Heess, and Alex Graves.
   "Recurrent models of visual attention."
   Advances in neural information processing systems. 2014.
   https://arxiv.org/abs/1406.6247
"""
from typing import NamedTuple

import numpy as np
import tensorflow as tf


class Glimpse(NamedTuple):
    """represents output of GlimpseSensor"""
    rho: tf.Tensor
    fixations: np.ndarray


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
        Glimpse : NamedTuple
            typed tuple, with following fields
                rho : tf.Tensor
                    with shape (B, k, g_w, g_w,
                    retina-like representation of k patches of increasing size
                    and decreasing resolution, centered around location loc within
                    image img
                fixations: np.ndarray
                    locations where glimpse sensor "fixates" converted from
                    normalized values in loc_normd to pixels in co-ordinate plane
                    of input images.
                top_left_corners: np.ndarray
                    calculated top left corners of extracted glimpse 'patches'.
                    Useful for plotting the glimpses.
        """
        batch_size, img_H, img_W, C = images.shape.as_list()
        # convert image co-ordinates from normalized to co-ordinates within
        # the specific size of the images
        # first convert location to range from 0 to 1
        # and then multiply by number of pixels - 1 (because of zero indexing)
        loc_0 = ((loc_normd[:, 0] + 1) / 2) * (img_H - 1)
        loc_0 = tf.cast(tf.round(loc_0), tf.int32)
        loc_1 = ((loc_normd[:, 1] + 1) / 2) * (img_W - 1)
        loc_1 = tf.cast(tf.round(loc_1), tf.int32)
        fixations = tf.stack([loc_0, loc_1], axis=1)

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

                # compute top left corner of patch.
                # note we add 'size' to compensate for padding
                patch_x = fixations[ind, 0] - (size // 2) + size
                patch_y = fixations[ind, 1] - (size // 2) + size
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
        fixations = fixations.numpy()
        return Glimpse(rho, fixations)


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

    def __init__(self, g_w=8, k=3, s=2, h_g_units=128, h_l_units=128, h_gt_units=256):
        """__init__ function for GlimpseNetwork

        Parameters
        ----------
        g_w : int
            size of square patches extracted by glimpse sensor.
        k : int
            number of patches to extract per glimpse.
        s : int
            scaling factor that controls size of successive patches.
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
        self.g_w = g_w
        self.k = k
        self.s = s
        self.h_g_units = h_g_units
        self.h_l_units = h_l_units
        self.h_gt_units = h_gt_units
        self.glimpse_sensor = GlimpseSensor(g_w=g_w, k=k, s=s)
        self.theta_g_0 = tf.keras.layers.Dense(units=h_g_units, activation='relu')
        self.theta_g_1 = tf.keras.layers.Dense(units=h_l_units, activation='relu')
        self.linear_h_g = tf.keras.layers.Dense(units=h_g_units, activation='linear')
        self.linear_h_l = tf.keras.layers.Dense(units=h_l_units, activation='linear')
        self.theta_g_2 = tf.keras.layers.Dense(units=h_gt_units, activation='relu')

    def forward(self, images, loc):
        """computes forward pass through GlimpseNetwork

        Parameters
        ----------
        images : tf.Tensor
            with shape (batch size, height, width, channels). Minibatch of images.
        loc : tf.Tensor
            with shape (batch size, 2). Location of retina "fixation",
            in normalized co-ordinates where center of image is (0,0),
            upper left corner is (-1,-1), and lower right corner is (1,1).

        Returns
        -------
        Glimpse : NamedTuple
            typed tuple, with following fields
                rho : tf.Tensor
                    with shape
                    (batch size, number of glimpses, glimpse size, glimpse size, channels);
                    Glimpse representation extracted by GlimpseSensor.
            fixations: np.ndarray
                locations where glimpse sensor "fixates" converted from
                normalized values in loc_normd to pixels in co-ordinate plane
                of input images.
            top_left_corners: np.ndarray
                calculated top left corners of extracted glimpse 'patches'.
                Useful for plotting the glimpses.
        g_t : tf.Tensor
            glimpse representation, output by glimpse network
        """
        glimpse = self.glimpse_sensor.glimpse(images, loc)
        batch_size, k, g, _, channels = glimpse.rho.shape.as_list()
        rho_vec = tf.reshape(glimpse.rho, shape=(batch_size, k * g * g * channels))
        h_g = self.theta_g_0(rho_vec)
        h_l = self.theta_g_1(loc)
        g_t = self.theta_g_2(
            self.linear_h_g(h_g) + self.linear_h_l(h_l)
        )
        return glimpse, g_t


class CoreNetwork(tf.keras.Model):
    """RNN that maintains an internal state which summarizes
    information extracted from the history of past observations.
    The external input to the network is the glimpse feature
    vector g_t.

    The output h_t = f_h(h_t_minus_1, g_t; theta_h) where theta_h
    is parameterized by the layers listed below in Attributes.

    Attributes
    ----------
    self.linear_h_t_minus_1 : linear layer of units that accepts
        hidden state from the last time step as an input
    self.linear_g_t : linear layer of units that accepts the
        glimpse feature vector g_t as an input

    So h_t = f_h(h_t_minus_1, g_t) =
        Rect(Linear(h_t_minus_1) + Linear(g_t))
    """
    def __init__(self, hidden_size=256):
        """__init__ function for CoreNetwork.
        Note that in [1]_ the network as implemented here is only
        used for classification; an LSTM was used for dynamic
        environments.

        Parameters
        ----------
        hidden_size : int
            Number of units in hidden layers that maintain internal state.
            Default is 256.
        """
        super(CoreNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.linear_h_t_minus_1 = tf.keras.layers.Dense(units=hidden_size)
        self.linear_g_t = tf.keras.layers.Dense(units=hidden_size)

    def forward(self, g_t, h_t_minus_1):
        """computes forward pass through CoreNetwork

        Parameters
        ----------
        g_t : tf.Tensor
            glimpse feature vector
        h_t_minus_1 : tf.Tensor
            output of CoreNetwork from previous time step

        Returns
        -------
        h_t : tf.Tensor
            = f_h(h_t_minus_1, g_t) = Rect(Linear(h_t_minus_1) + Linear(g_t))
            = tf.relu(self.linear_g_t(g_t) + self.linear_h_t_minus_1(h_t_minus_1))
        """
        h_t = self.linear_g_t(g_t) + self.linear_h_t_minus_1(h_t_minus_1)
        h_t = tf.nn.relu(h_t)
        return h_t


class LocationNetwork(tf.keras.Model):
    """Uses internal state `h_t` of core network
    to produce location coordinates `l_t` for the
    next time step.

    The location network is a fully-connected layer
    that parameterizes a normal distribution with
    mean mu and a constant standard deviation sigma
    (specified by the user). Locations are drawn from
    this distribution on each time step, by passing
    the hidden state h_t through the location network l_t.

    Attributes
    ----------
    self.forward
    """
    def __init__(self, loc_std, output_size=2):
        """__init__ for LocationNetwork

        Parameters
        ----------
        loc_std : float
            standard deviation of normal distribution from which
            location co-ordinates are drawn.
        output_size : int
            dimensionality of output of fully connected layer
            in location network. Default is 2, i.e. x and y co-ordinates
            of a location.
        """
        super(LocationNetwork, self).__init__()
        self.output_size = output_size
        self.loc_std = loc_std
        self.fc = tf.keras.layers.Dense(units=output_size, activation='tanh')

    def forward(self, h_t):
        """forward pass through LocationNetwork.
        Ues location policy to compute l_t, the location to glimpse next,
        given internal state `h_t` of core network.

        Passes h_t through a fully connected layer with tan_h activation
        to clamp the output between [-1, 1]. This results in mu of
        distribution with standard deviation self.loc_std. The location
        l_t is then drawn from this distribution then returned.

        Parameters
        ----------
        h_t : tf.Tensor
            with shape (B, num_hidden). Output of core network.

        Returns
        -------
        mu : tf.Tensor
            with shape (B, 2). mu parameter for normal distributions
            from which l_t will be drawn. Since l_t is a location with
            x and y co-ordinates, there is one value of mu for each
            distribution (the one that represents the x co-ordinate and
            the one that represents the y co-ordinate).
        l_t : tf.Tensor
            with shape (B, 2)
        """
        mu = self.fc(h_t)
        y = tf.random_normal(mu.get_shape(), mean=mu, stddev=self.loc_std)
        # run through tanh again to bound between -1 and 1
        l_t = tf.tanh(y)
        return mu, l_t


class ActionNetwork(tf.keras.Model):
    """Uses internal state `h_t` of CoreNetwork to
    produce final output classification.

    Feeds hidden state `h_t` through a fully-connected
    layer followed by a softmax to yield the vector of
    output probabilities over the possible classes.
    """
    def __init__(self, num_actions):
        super(ActionNetwork, self).__init__()
        self.num_actions = num_actions
        self.fc = tf.keras.layers.Dense(units=num_actions, activation='linear')

    def forward(self, h_t):
        """forward pass through ActionNetwork

        Returns
        -------
        a_t : tf.Tensor
            "actions" to take, currently just classification
        """
        a_t = self.fc(h_t)
        return a_t


class BaselineNetwork(tf.keras.Model):
    """Provides estimate of (state) value that does not depend on
    actions taken. Used during optimization as a baseline,
    subtracted from return on each time step,
    to reduce variance of the policy function gradient.

    Attributes
    ----------
    self.fc : tf.keras.layers.Dense
        fully-connected layer with Rectified Linear activation
    """
    def __init__(self, output_size=1):
        """__init__ for BaselineNetwork

        Parameters
        ----------
        output_size : int
            Number of units in fully-connected layer of BaselineNetwork.
            Should be a scalar, since it is the estimate of total return
            from the current state. Default is 1.
        """
        super(BaselineNetwork, self).__init__()
        self.output_size = output_size
        self.fc = tf.keras.layers.Dense(units=output_size, activation='relu')

    def forward(self, h_t):
        """forward pass through BaselineNetwork.

        Returns
        -------
        b_t : tf.Tensor
            baseline network output with size (batch, self.output_size)
        """
        b_t = self.fc(h_t)
        return b_t
