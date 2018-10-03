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
                 min_radius=4,  # zooms -> minRadius * 2**<depth_level>
                 depth=3,  # zooms
                 channels=1,  # grayscale
                 batch_size = 10,
                 hg_size = 128,
                 hl_size = 128,
                 g_size = 256,
                 cell_size = 256,
                 cell_out_size = cell_size,
                 glimpses=6,
                 n_classes = 10,
                 lr = 1e-3,
                 max_iters = 1000000,
                 mnist_size = 28,
                 loc_sd = 0.1
                 ):
        """

        Parameters
        ----------
        sensor_bandwidth : int
            size of square input window, in pixels. Default is 8.
        min_radius

        sensor_area
        depth
        channels
        batch_size
        hg_size
        hl_size
        g_size
        cell_size
        cell_out_size
        """

        self.min_radius = min_radius
        self.sensor_bandwidth = sensor_bandwidth
        self.sensor_area = sensor_bandwidth ** 2,
        self.total_sensor_bandwidth = depth * sensorBandwidth * sensorBandwidth * channels
        self.mean_locs = []
        self.sampled_locs = []  # ~N(mean_locs[.], loc_sd)
        self.glimpse_images = []  # to show in window


    @define_scope
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=1.0 / shape[0])  # for now
        return tf.Variable(initial)

    @define_scope
    def glimpse_sensor(self, img, normLoc):
        loc = ((normLoc + 1) / 2) * mnist_size  # normLoc coordinates are between -1 and 1
        loc = tf.cast(loc, tf.int32)

        img = tf.reshape(img, (batch_size, mnist_size, mnist_size, channels))

        zooms = []

        # process each image individually
        for k in range(self.batch_size):
            img_zooms = []
            one_img = img[k, :, :, :]
            max_radius = minRadius * (2 ** (depth - 1))
            offset = max_radius

            # pad image with zeros
            one_img = tf.image.pad_to_bounding_box(one_img, offset, offset,
                                                   max_radius * 2 + mnist_size, max_radius * 2 + mnist_size)

            for i in range(self.depth):
                r = int(self.min_radius * (2 ** (i - 1)))

                d_raw = 2 * r
                d = tf.constant(d_raw, shape=[1])

                d = tf.tile(d, [2])

                loc_k = loc[k, :]
                adjusted_loc = offset + loc_k - r

                one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value, \
                                                one_img.get_shape()[1].value))

                # crop image to (d x d)
                zoom = tf.slice(one_img2, adjusted_loc, d)

                # resize cropped image to (sensorBandwidth x sensorBandwidth)
                zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw, d_raw, 1)), (sensorBandwidth, sensorBandwidth))
                zoom = tf.reshape(zoom, (sensorBandwidth, sensorBandwidth))
                img_zooms.append(zoom)

            zooms.append(tf.stack(img_zooms))

        zooms = tf.stack(zooms)

        glimpse_images.append(zooms)

        return zooms

    @define_scope
    def get_glimpse(self, loc):
        glimpse_input = self.glimpse_sensor(inputs_placeholder, loc)

        glimpse_input = tf.reshape(glimpse_input,
                                   (self.batch_size,
                                    self.totalSensorBandwidth))

        l_hl = self.weight_variable((2, hl_size))
        glimpse_hg = self.weight_variable((totalSensorBandwidth, hg_size))

        hg = tf.nn.relu(tf.matmul(glimpse_input, glimpse_hg))
        hl = tf.nn.relu(tf.matmul(loc, l_hl))

        hg_g = self.weight_variable((hg_size, g_size))
        hl_g = self.weight_variable((hl_size, g_size))

        g = tf.nn.relu(tf.matmul(hg, hg_g) + tf.matmul(hl, hl_g))

        return g

    @define_scope
    def get_next_input(self, output, i):
        mean_loc = tf.tanh(tf.matmul(output, h_l_out))
        mean_locs.append(mean_loc)

        sample_loc = mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd)

        sampled_locs.append(sample_loc)

        return get_glimpse(sample_loc)

    @define_scope
    def model(self):
        initial_loc = tf.random_uniform((self.batch_size, 2), minval=-1, maxval=1)

        initial_glimpse = get_glimpse(initial_loc)

        lstm_cell = rnn_cell.LSTMCell(cell_size, g_size, num_proj=cell_out_size)

        initial_state = lstm_cell.zero_state(batch_size, tf.float32)

        inputs = [initial_glimpse]
        inputs.extend([0] * (glimpses - 1))

        outputs, _ = seq2seq.rnn_decoder(inputs, initial_state, lstm_cell,
                                         loop_function=get_next_input)
        get_next_input(outputs[-1], 0)

        return outputs
