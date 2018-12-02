"""RAM model, from [1]_.
Based on two implementations:
https://github.com/kevinzakka/recurrent-visual-attention
https://github.com/seann999/tensorflow_mnist_ram

.. [1] Mnih, Volodymyr, Nicolas Heess, and Alex Graves.
   "Recurrent models of visual attention."
   Advances in neural information processing systems. 2014.
   https://arxiv.org/abs/1406.6247
"""
from collections import namedtuple

import tensorflow as tf
from tensorflow.python.training.checkpointable import tracking

from . import modules


StateAndMeta = namedtuple('StateAndMeta', ['rho', 'fixations', 'h_t', 'mu', 'l_t', 'a_t', 'b_t'])


class RAM(tracking.Checkpointable):
    """RAM model, from [1]_.
    Based on two implementations:
    https://github.com/kevinzakka/recurrent-visual-attention
    https://github.com/seann999/tensorflow_mnist_ram

    Attributes
    ----------
    self.g_w : int
        length of one side of square patches in glimpses
        extracted by glimpse sensor. Default is 8.
    self.k : int
        number of patches that the glimpse sensor extracts
        at location l from image x and returns in the
        retina-like encoding rho(x,l). Default is 3.
    self.s : int
        scaling factor, size to increase each successive
        patch in one glimpse, i.e. size of patch k will
        be g_w * s**k. Default is 2.
    self.hg_size : int
        Size of hidden layer (number of units) in GlimpseNetwork
        that embeds glimpse
    self.hl_size : int
        Size of hidden layer (number of units) in GlimpseNetwork
        that embeds location
    self.g_size : int
        Size of hidden layer (number of units) in GlimpseNetwork
        that produces glimpse feature vector by combining
    self.hidden_size : int
        Size of hidden layer (number of units) in CoreNetwork
    self.glimpses : int
        Number of glimpses to take before acting.
    self.batch_size : int
        Number of samples per batch. Default is 10.
    self.num_classes : int
        Number of classes, i.e. for MNIST, num_classes = 10
    self.loc_std : float
        Standard deviation of two-component Gaussian from which
        locations are drawn. The Gaussian distribution is
        parameterized by the LocationNetwork, that takes the
        hidden state h_t from the CoreNetwork as its input.
    self.glimpse_sensor : ram.modules.GlimpseSensor
        provides input to glimpse_network
        input size is square with length = self.sensor_bandwidth
    self.glimpse_network : ram.modules.GlimpseNetwork

    self.location_network : ram.modules.LocationNetwork

    self.action_network : ram.modules.ActionNetwork

    self.baseline_network : ram.modules.BaselineNetwork

    self.Out : namedtuple
    with fields rho, h_t, mu, l_t, a_t, and b_t
        rho : tf.Tensor
            glimpse representation extracted by glimpse sensor from image
        h_t : tf.Tensor
            hidden state of core network at time step t
        mu : tf.Tensor
            mu parameter of normal distribution from which l_t is drawn
        l_t : tf.Tensor
            output of location network at time step t,
            will be location to glimpse on time step t plus one
        a_t : tf.Tensor
            output of action network at time step t.
            For images, probability of classes. Only used at final step
            as "action" of deciding class of image.
        b_t : tf.Tensor
            output of baseline network at time step t.
            Provides estimate of q(t) that is used during


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
        self.s = s
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

        self.initial_l_t_distrib = tf.distributions.Uniform(low=-1.0, high=1.0)

    def reset(self):
        """get initial states for h_t and l_t.
        called at the beginning of an episode / trial.

        Returns
        -------
        h_t : tf.Tensor
            of zeros, with size (self.batch_size, self.hidden_size).
        l_t : tf.Tensor
            with size (self.batch_size, 2),
            drawn from uniform distribution between -1 and 1.
        """
        # TODO: test other initializations for hidden state besides zeros
        # see https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
        h_t = tf.zeros(shape=(self.batch_size, self.hidden_size,))
        l_t = self.initial_l_t_distrib.sample(sample_shape=(self.batch_size, 2))
        out = StateAndMeta(None, None, h_t, None, l_t, None, None)
        return out

    def step(self, images, l_t_minus_1, h_t_minus_1):
        """executes one time step of the model.

        Parameters
        ----------
        images : tf.Tensor
            with shape (B, H, W, C). Images the network is glimpsing.
        l_t_minus_1 : tf.Tensor
            with shape (B, 2). Where to glimpse,
            i.e., output of location network from previous time step.
        h_t_minus_1 : tf.Tensor
            with shape (B, h_size).
            Hidden state of core network from previous time step.

        Returns
        -------
        State : self.Out
            namedtuple with fields h_t, mu, l_t, a_t, and b_t
                rho : tf.Tensor
                    glimpse representation extracted by glimpse sensor from image
                h_t : tf.Tensor
                    hidden state of core network at time step t.
                mu : tf.Tensor
                    mu parameter of normal distribution from which l_t is drawn.
                l_t : tf.Tensor
                    output of location network at time step t,
                    will be location to glimpse on time step t plus one.
                a_t : tf.Tensor
                    output of action network at time step t.
                    For images, probability of classes. Only used at final step
                    as "action" of deciding class of image.
                b_t : tf.Tensor
                    output of baseline network at time step t.
                    Provides estimate of q(t) that is used during
                    training to reduce variance.
        """
        glimpse, g_t = self.glimpse_network.forward(images, l_t_minus_1)
        h_t = self.core_network.forward(g_t, h_t_minus_1)
        mu, l_t = self.location_network.forward(h_t)
        b_t = self.baseline.forward(h_t)
        a_t = self.action_network.forward(h_t)
        return StateAndMeta(glimpse.rho, glimpse.fixations, h_t, mu, l_t, a_t, b_t)





