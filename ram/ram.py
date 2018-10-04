"""RAM model, from [1].
Based on the implementation at https://github.com/seann999/tensorflow_mnist_ram
"""

import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib import rnn_cell

from .decorators import define_scope


class RAM:
    """RAM model, from [1].
    Based on the implementation at https://github.com/seann999/tensorflow_mnist_ram

    Attributes
    ----------
    glimpse_sensor : provides input to glimpse_network
        input size is square with length = self.sensor_bandwidth
    glimpse_network :
    """

    def __init__(self,
                 sensor_bandwidth=8,
                 min_radius=4,  # zooms -> minRadius * 2**<num_patches_level>
                 num_patches=3,  # zooms
                 channels=1,  # grayscale
                 batch_size = 10,
                 input_image_size = (28,28),
                 hg_size = 128,
                 hl_size = 128,
                 g_size = 256,
                 cell_size = 256,
                 glimpses=6,
                 n_classes = 10,
                 learning_rate = 1e-3,
                 max_iters = 1000000,
                 loc_sd = 0.1
                 ):
        """

        Parameters
        ----------
        sensor_bandwidth : int
            size of square input window, in pixels. Default is 8.
        min_radius : int

        num_patches : int
            number of patches k that the retina encoding p(x,l) extracts
            at location l from image x. Default is 3.
        channels : int
            Number of color channels in images. Default is 1 (grayscale).
        batch_size : int
            Default is 10.
        input_image_size : tuple
            Default is (28, 28) (size of MNIST).
        hg_size
        hl_size
        g_size
        cell_size
        """

        self.min_radius = min_radius
        self.num_patches = num_patches
        self.sensor_bandwidth = sensor_bandwidth
        self.sensor_area = sensor_bandwidth ** 2
        self.batch_size = batch_size
        self.input_image_size = input_image_size
        self.total_sensor_bandwidth = (num_patches * sensor_bandwidth * sensor_bandwidth * channels)
        self.cell_size = cell_size
        self.cell_out_size = self.cell_size  # not clear to me why
        self.learning_rate = learning_rate

        self.graph = tf.Graph()
        with self.graph:
            self.labels = tf.placeholder("float32", shape=[self.batch_size, self.n_classes],
                                         name="labels")
            self.inputs = tf.placeholder(tf.float32,
                                         shape=(self.batch_size,
                                                self.input_image_size), name="images")
            self.labels = tf.placeholder(tf.float32,
                                         shape=(batch_size), name="labels")
            self.onehot = tf.placeholder(tf.float32, shape=(batch_size, 10), name="oneHotLabels")

            self.model

    @define_scope
    def _weight_variable(self, shape):
        """convenience function to return variable with weights"""
        initial = tf.truncated_normal(shape, stddev=1.0 / shape[0])
        return tf.Variable(initial)

    @define_scope
    def glimpse_sensor(self, img, loc_normd):
        """glimpse sensor, returns retina-like representation

        Parameters
        ----------
        img : ndarray
            image
        loc_normd : ndarray
            location of retina "fixation", in normalized co-ordinates
            where center of image is (0,0), upper left corner is (-1,-1),
            and lower right corner is (1,1)

        Returns
        -------

        """

        loc = ((loc_normd + 1) / 2) * self.input_image_size
        loc = tf.cast(loc, tf.int32)
        img = tf.reshape(img, (self.batch_size,
                               self.input_image_size[0],
                               self.input_image_size[1],
                               self.channels))

        zooms = []

        # process each image individually
        for k in range(self.batch_size):
            img_zooms = []
            one_img = img[k, :, :, :]
            max_radius = self.min_radius * (2 ** (self.num_patches - 1))
            offset = max_radius

            # pad image with zeros
            one_img = tf.image.pad_to_bounding_box(one_img, offset, offset,
                                                   max_radius * 2 + mnist_size,
                                                   max_radius * 2 + mnist_size)

            for i in range(self.num_patches):
                r = int(self.min_radius * (2 ** (i - 1)))

                d_raw = 2 * r
                d = tf.constant(d_raw, shape=[1])

                d = tf.tile(d, [2])

                loc_k = loc[k, :]
                adjusted_loc = offset + loc_k - r

                one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value,
                                                one_img.get_shape()[1].value))

                # crop image to (d x d)
                zoom = tf.slice(one_img2, adjusted_loc, d)

                # resize cropped image to (sensorBandwidth x sensorBandwidth)
                zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw, d_raw, 1)), (sensorBandwidth, sensorBandwidth))
                zoom = tf.reshape(zoom, (sensorBandwidth, sensorBandwidth))
                img_zooms.append(zoom)

            zooms.append(tf.stack(img_zooms))

        zooms = tf.stack(zooms)

        tf.summary.image(name='patches', tensor=zooms)

        return zooms

    @define_scope
    def glimpse_network(self, loc):
        glimpse_input = self.glimpse_sensor(self.inputs, loc)

        glimpse_input = tf.reshape(glimpse_input,
                                   (self.batch_size,
                                    self.totalSensorBandwidth))

        l_hl = self._weight_variable((2, hl_size))
        glimpse_hg = self._weight_variable((totalSensorBandwidth, hg_size))

        hg = tf.nn.relu(tf.matmul(glimpse_input, glimpse_hg))
        hl = tf.nn.relu(tf.matmul(loc, l_hl))

        hg_g = self._weight_variable((hg_size, g_size))
        hl_g = self._weight_variable((hl_size, g_size))

        g = tf.nn.relu(tf.matmul(hg, hg_g) + tf.matmul(hl, hl_g))

        return g

    @define_scope
    def _get_next_input(self, output, i):
        mean_loc = tf.tanh(tf.matmul(output, h_l_out))
        mean_locs.append(mean_loc)

        sample_loc = mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd)

        sampled_locs.append(sample_loc)

        return get_glimpse(sample_loc)

    @define_scope
    def model(self):
        initial_loc = tf.random_uniform((self.batch_size, 2), minval=-1, maxval=1)
        initial_glimpse = self.glimpse_network(initial_loc)
        lstm_cell = rnn_cell.LSTMCell(cell_size, g_size, num_proj=cell_out_size)
        initial_state = lstm_cell.zero_state(batch_size, tf.float32)
        inputs = [initial_glimpse]
        inputs.extend([0] * (glimpses - 1))
        outputs, _ = seq2seq.rnn_decoder(inputs, initial_state, lstm_cell,
                                         loop_function=self._get_next_input)
        self._get_next_input(outputs[-1], 0)
        return outputs

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
