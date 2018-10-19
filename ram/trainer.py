import tensorflow as tf

# optimizer = tf.train.AdamOptimizer(self.learning_rate)
# train_op = optimizer.minimize(cost)


class Trainer:
    def __init__(self,
                 data,
                 batch_size,
                 learning_rate=1e-3,
                 max_iters=1000000,
                 ):
        self.data = data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iters = max_iters

    def train(self):
        """trains RAM model
        """
        for batch_num, (img, lbl) in self.data.batch(batch_size):
            self._train_one_epoch(self, img, lbl)

    def _train_one_epoch(self, img, lbl):
        # training loop
        out_t_minus_1 = ram_model.reset()

        locs = []
        mus = []
        log_pis = []
        baselines = []

        for t in range(ram_model.glimpses):
            out = ram_model.step(img, out_t_minus_1.l_t, out_t_minus_1.h_t)

            locs.append(out.l_t)
            mus.append(out.mu)
            baselines.append(out.b_t)

            # determine probability of choosing location l_t, given
            # distribution parameterized by mu (output of location network)
            # and the constant standard deviation specified as a parameter
            log_pi = tf.distributions.Normal(loc=out.mu,
                                             scale=ram_model.loc_std).log_prob(value=out.l_t)
            log_pi = tf.reduce_sum(log_pi, axis=1)
            log_pis.append(log_pi)

            out_t_minus_1 = out

        # convert lists to tensors, reshape to (batch size x number of glimpses)
        # for calculations below
        baselines = tf.stack(baselines)
        baselines = tf.squeeze(baselines)
        baselines = tf.transpose(baselines, perm=[1, 0])

        log_pi = tf.stack(log_pi)
        log_pi = tf.squeeze(log_pi)
        log_pi = tf.transpose(log_pi, perm=[1, 0])

        # calculate reward
        predicted = tf.argmax(out.a_t, output_type=tf.int32)  # a_t = predictions from last time step
        R = tf.equal(predicted, lbl)
        # reshape reward to (batch size x number of glimpses)
        R = tf.reshape(R, shape=(-1, 1))  # add axis
        R = tf.transpose(R, perm=[1, 0])  # flip so it's a column vector
        # repeat column vector n times where n = glimpses
        R = tf.tile(R, tf.constant([ram_model.glimpses, 1]))

        # compute losses for differentiable modules
        loss_action = tf.losses.softmax_cross_entropy(tf.onehot(lbl), out.a_t)
        R = tf.reshape
        baselines = tf.stack(baselines)
        loss_baseline = tf.losses.mean_squared_error(baselines, R)

        # compute loss for REINFORCE algorithm
        # summed over timesteps and averaged across batch
        adjusted_reward = R - baselines
        loss_reinforce = tf.reduce_sum(-log_pi * adjusted
        reward, axis = 1)
        loss_reinforce = tf.mean(loss_reinforce, axis=0)

        # sum up into hybrid loss
        loss = loss_action + loss_baseline + loss_reinforce
