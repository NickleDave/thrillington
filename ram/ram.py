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

from .decorators import define_scope
from . import graphs

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
                 hg_size=128,
                 hl_size=128,
                 g_size=256,
                 cell_size=256,
                 glimpses=6,
                 num_classes=10,
                 learning_rate=1e-3,
                 max_iters=1000000,
                 loc_sd=0.1
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
        hg_size : int

        hl_size : int

        g_size : int

        cell_size : int

        glimpses : int

        num_classes : int

        learning_rate : float

        max_iters : int

        loc_sd : float
        """

        # user-specified properties
        self.g_w = g_w
        self.k = k
        self.hg_size = hg_size
        self.hl_size = hl_size
        self.g_size = g_size
        self.cell_size = cell_size
        self.glimpses = glimpses
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.loc_sd = loc_sd

        self.graph = tf.Graph()
        with self.graph:
            self.labels = tf.placeholder("float32", shape=[self.batch_size, self.num_classes],
                                         name="labels")
            self.inputs = tf.placeholder(tf.float32,
                                         shape=(self.batch_size,
                                                self.input_image_size), name="images")
            self.labels = tf.placeholder(tf.float32,
                                         shape=(self.batch_size), name="labels")
            self.onehot = tf.placeholder(tf.float32, shape=(batch_size, 10), name="oneHotLabels")

            self.glimpse_network =
            self.forward

    @define_scope
    def forward(self, images, l_t_minus_1, h_t_minus_1):
        """

        Parameters
        ----------
        images
        l_t_minus_1
        h_t_minus_1

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

    @define_scope
    def train(self):
        initial_loc = tf.random_uniform((self.batch_size, 2), minval=-1, maxval=1)
        initial_glimpse = self.forward(self.inputs,
                                       initial_loc)
        for glimpse in range(self.glimpses):


    def gaussian_pdf(self, mean, sample):
        """used to estimate maximum likelihood for glimpse location"""
        Z = 1.0 / (loc_sd * tf.sqrt(2.0 * math.pi))
        a = -tf.square(sample - mean) / (2.0 * tf.square(loc_sd))
        return Z * tf.exp(a)

    def calc_reward(self, outputs):
        outputs = outputs[-1]  # look at ONLY THE END of the sequence
        outputs = tf.reshape(outputs, (self.batch_size, self.cell_out_size))
        h_a_out = self._weight_variable((cell_out_size, n_classes))

        p_y = tf.nn.softmax(tf.matmul(outputs, h_a_out))
        max_p_y = tf.arg_max(p_y, 1)
        correct_y = tf.cast(self.labels, tf.int64)

        R = tf.cast(tf.equal(max_p_y, correct_y), tf.float32)  # reward per example

        reward = tf.reduce_mean(R)  # overall reward

        p_loc = self.gaussian_pdf(mean_locs, sampled_locs)
        p_loc = tf.reshape(p_loc, (self.batch_size, self.glimpses * 2))

        R = tf.reshape(R, (batch_size, 1))
        J = tf.concat(1, [tf.log(p_y + 1e-5) * self.onehot_labels, tf.log(p_loc + 1e-5) * R])
        J = tf.reduce_sum(J, 1)
        J = tf.reduce_mean(J, 0)
        cost = -J

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(cost)

        return cost, reward, max_p_y, correct_y, train_op
