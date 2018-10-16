"""RAM model, from [1]_.
Based on two implementations:
https://github.com/seann999/tensorflow_mnist_ram
https://github.com/kevinzakka/recurrent-visual-attention

.. [1] Mnih, Volodymyr, Nicolas Heess, and Alex Graves.
   "Recurrent models of visual attention."
   Advances in neural information processing systems. 2014.
   https://arxiv.org/abs/1406.6247
"""
from collections import namedtuple

import tensorflow as tf

from . import modules


class RAM:
    """RAM model, from [1]_.
    Based on this implementation: https://github.com/seann999/tensorflow_mnist_ram

    Attributes
    ----------
    self.glimpse_sensor : provides input to glimpse_network
        input size is square with length = self.sensor_bandwidth
    self.glimpse_network :

    .. [1] Mnih, Volodymyr, Nicolas Heess, and Alex Graves.
       "Recurrent models of visual attention."
       Advances in neural information processing systems. 2014.
       https://arxiv.org/abs/1406.6247
    """

    def __init__(self,
                 g_w=8,
                 k=3,
                 s=2,
                 hg_size=128,
                 hl_size=128,
                 g_size=256,
                 hidden_size=256,
                 glimpses=6,
                 batch_size=10,
                 num_classes=10,
                 loc_std=0.1
                 ):
        """__init__ for RAM class

        Parameters
        ----------
        g_w : int
            length of one side of square patches in glimpses
            extracted by glimpse sensor. Default is 8.
        k : int
            number of patches that the glimpse sensor extracts
            at location l from image x and returns in the
            retina-like encoding rho(x,l). Default is 3.
        s : int
            scaling factor, size to increase each successive
            patch in one glimpse, i.e. size of patch k will
            be g_w * s**k. Default is 2.
        hg_size : int
            Size of hidden layer (number of units) in GlimpseNetwork
            that embeds glimpse
        hl_size : int
            Size of hidden layer (number of units) in GlimpseNetwork
            that embeds location
        g_size : int
            Size of hidden layer (number of units) in GlimpseNetwork
            that produces glimpse feature vector by combining
        hidden_size : int
            Size of hidden layer (number of units) in CoreNetwork
        glimpses : int
            Number of glimpses to take before acting.
        batch_size : int
            Number of samples per batch. Default is 10.
        num_classes : int
            Number of classes, i.e. for MNIST, num_classes = 10
        loc_std : float
            Standard deviation of two-component Gaussian from which
            locations are drawn. The Gaussian distribution is
            parameterized by the LocationNetwork, that takes the
            hidden state h_t from the CoreNetwork as its input.
        """

        # user-specified properties
        self.g_w = g_w
        self.k = k
        self.hg_size = hg_size
        self.hl_size = hl_size
        self.g_size = g_size
        self.hidden_size = hidden_size
        self.glimpses = glimpses
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.loc_std = loc_std

        self.glimpse_network = modules.GlimpseNetwork(g_w, k, s, hg_size,
                                                      hl_size, g_size)
        self.core_network = modules.CoreNetwork(hidden_size)
        self.location_network = modules.LocationNetwork(loc_std)
        self.action_network = modules.ActionNetwork(num_classes)
        self.baseline = modules.BaselineNetwork()

        self.Out = namedtuple('Out', ['h_t', 'b_t', 'l_t', 'mu', 'log_pi', 'log_probas'])

        self.initial_l_t = tf.distributions.Uniform(low=-1.0, high=1.0)

    def reset(self):
        """get initial states for h_t and l_t.
        called at the beginning of an episode / trial.

        Returns
        -------
        h_t : tf.Tensor
            of zeros
        l_t : tf.Tensor
            with size (self.batch_size, 2),
            drawn from uniform distribution between -1 and 1.
        """
        # TODO: test other initializations for hidden state besides zeros
        # see https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
        h_t = tf.zeros(shape=(self.batch_size, self.hidden_size,))
        l_t = self.initial_l_t.sample(sample_shape=(self.batch_size, 2))
        return h_t, l_t

    def step(self, images, l_t_minus_1, h_t_minus_1):
        """executes one time step of the model.

        Parameters
        ----------
        images : tf.Tensor
            with shape (B, H, W, C). Images the network is glimpsing
        l_t_minus_1 : tf.Tensor
            with shape (B, 2). Where to glimpse,
            i.e., output of location network from previous time step.
        h_t_minus_1 : tf.Tensor
            with shape (B, h_size).
            Hidden state of core network from previous time step.

        Returns
        -------
        out : self.Out
            namedtuple with fields h_t, mu, l_t, log_pi, and log_probas
        """
        g_t = self.glimpse_network.forward(images, l_t_minus_1)
        h_t = self.core_network.forward(g_t, h_t_minus_1)
        mu, l_t = self.location_network.forward(h_t)
        b_t = self.baseline(h_t)

        log_pi = tf.distributions.Normal(
            loc=mu, scale=self.loc_std).log_prob(value=l_t)
        log_pi = tf.reduce_sum(log_pi, axis=1)
        log_probas = self.action_network(h_t)
        out = self.Out(h_t, l_t, b_t, mu, log_pi, log_probas)
        return out





