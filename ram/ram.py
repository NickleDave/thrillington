"""RAM model, from [1]_.
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
        self.num_classes = num_classes
        self.loc_std = loc_std

        self.glimpse_network = modules.GlimpseNetwork(g_w, k, s, hg_size,
                                                      hl_size, g_size)
        self.core_network = modules.CoreNetwork(hidden_size)
        self.location_network = modules.LocationNetwork(loc_std)

    def step(self, images, l_t_minus_1, h_t_minus_1):
        """executes one time step of the model.

        Parameters
        ----------
        images : tf.Tensor
            with shape (B, H, W, C). Images the network is glimpsing
        l_t_minus_1 : tf.Tensor
            with shape (B, 2). Locations of glimpses from previous time step.
        h_t_minus_1 : tf.Tensor
            with shape (B, h_size). Hidden states of core network from previous time step.

        Returns
        -------

        """

        lstm_cell = rnn_cell.LSTMCell(cell_size, g_size, num_proj=cell_out_size)
        initial_state = lstm_cell.zero_state(batch_size, tf.float32)
        inputs = [initial_glimpse]
        inputs.extend([0] * (glimpses - 1))
        outputs, _ = seq2seq.rnn_decoder(inputs, initial_state, lstm_cell,
                                         loop_function=self._get_next_input)
        self._get_next_input(outputs[-1], 0)
        return outputs




