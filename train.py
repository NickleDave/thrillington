import tensorflow as tf


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


def train():
    """trains RAM model 
    """
    learning_rate = 1e-3,
    max_iters = 1000000,

    with tf.Graph() as graph:
        labels = tf.placeholder("float32", shape=[self.batch_size, self.num_classes],
                                     name="labels")
        inputs = tf.placeholder(tf.float32,
                                     shape=(self.batch_size,
                                            self.input_image_size), name="images")
        labels = tf.placeholder(tf.float32,
                                     shape=(self.batch_size), name="labels")
        onehot = tf.placeholder(tf.float32, shape=(batch_size, 10), name="oneHotLabels")

        
        initial_loc = tf.random_uniform((self.batch_size, 2), minval=-1, maxval=1)
        initial_glimpse = self.forward(self.inputs,
                                       initial_loc)
        for glimpse in range(self.glimpses):

if __name__ == 'main':
    train()
