"""trainer for RAM model, from [1]_.
Based on two implementations:
https://github.com/kevinzakka/recurrent-visual-attention
https://github.com/seann999/tensorflow_mnist_ram

.. [1] Mnih, Volodymyr, Nicolas Heess, and Alex Graves.
   "Recurrent models of visual attention."
   Advances in neural information processing systems. 2014.
   https://arxiv.org/abs/1406.6247
"""
import tensorflow as tf

import ram


class Trainer:
    """Trainer object for training the RAM model"""
    def __init__(self,
                 config,
                 data
                 ):
        """__init__ for Trainer

        Parameters
        ----------
        config : namedtuple
            as returned by ram.utils.parse_config.
        data : tf.data.Dataset object
            such as the mnist training set,
            returned by calling ram.dataset.train
        """
        # apply model config
        self.model = ram.RAM(**config.model._asdict())

        # then unpack train config
        self.data = data
        self.batch_size = config.train.batch_size
        self.learning_rate = config.train.learning_rate
        self.max_iters = config.train.max_iters
        if config.train.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(momentum=0.9,
                                                        learning_rate=self.learning_rate)
        elif config.train.optimizer == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

    def train(self):
        """trains RAM model
        """
        for batch_num, (img, lbl) in self.data.batch(self.batch_size):
            self._train_one_epoch(self, img, lbl)

    def _train_one_epoch(self, img, lbl):

        out_t_minus_1 = self.model.reset()

        locs = []
        mus = []
        log_pis = []
        baselines = []

        with tf.GradientTape(persistent=True) as tape:
            for t in range(self.model.glimpses):
                out = self.model.step(img, out_t_minus_1.l_t, out_t_minus_1.h_t)

                locs.append(out.l_t)
                mus.append(out.mu)
                baselines.append(out.b_t)

                # determine probability of choosing location l_t, given
                # distribution parameterized by mu (output of location network)
                # and the constant standard deviation specified as a parameter.
                # Assume both dimensions are independent
                # 1. we get log probability from pdf for each dimension
                # 2. we want the joint distribution which is the product of the pdfs
                # 3. so we sum the log prob, since log(p(x) * p(y)) = log(p(x)) + log(p(y))
                mu_distrib = tf.distributions.Normal(loc=out.mu,
                                                     scale=self.model.loc_std)
                log_pi = mu_distrib.log_prob(value=out.l_t)
                log_pi = tf.reduce_sum(log_pi, axis=1)
                log_pis.append(log_pi)

                out_t_minus_1 = out

            # convert lists to tensors, reshape to (batch size x number of glimpses)
            # for calculations below
            baselines = tf.stack(baselines)
            baselines = tf.squeeze(baselines)
            baselines = tf.transpose(baselines, perm=[1, 0])

            log_pis = tf.stack(log_pis)
            log_pis = tf.squeeze(log_pis)
            log_pis = tf.transpose(log_pis, perm=[1, 0])

            # repeat column vector n times where n = glimpses
            # calculate reward
            predicted = tf.argmax(out.a_t, axis=1, output_type=tf.int32)  # a_t = predictions from last time step
            R = tf.equal(predicted, lbl)
            R = tf.cast(R, dtype=tf.float32)
            # reshape reward to (batch size x number of glimpses)
            R = tf.expand_dims(R, axis=1)  # add axis
            R = tf.tile(R, tf.constant([1, self.model.glimpses]))

            # compute losses for differentiable modules
            loss_action = tf.losses.softmax_cross_entropy(tf.one_hot(lbl, depth=self.model.num_classes), out.a_t)
            loss_baseline = tf.losses.mean_squared_error(baselines, R)

            # compute loss for REINFORCE algorithm
            # summed over timesteps and averaged across batch
            adjusted_reward = R - baselines
            loss_reinforce = tf.reduce_sum((-log_pis * adjusted_reward), axis=1)
            loss_reinforce = tf.reduce_mean(loss_reinforce)

            # sum up into hybrid loss
            hybrid_loss = loss_action + loss_baseline + loss_reinforce

        # apply reinforce loss **only** to location network and baseline network
        lt_bt_params = [var for net in [self.model.location_network,
                                  self.model.baseline]
                  for var in net.variables]
        reinforce_grads = tape.gradient(loss_reinforce, lt_bt_params)
        self.optimizer.apply_gradients(zip(reinforce_grads, lt_bt_params),
                                       global_step=tf.train.get_or_create_global_step())
        # apply hybrid loss to glimpse network, core network, and action network
        params = [var for net in [self.model.glimpse_network,
                                  self.model.action_network,
                                  self.model.core_network]
                  for var in net.variables]
        hybrid_grads = tape.gradient(hybrid_loss, params)
        self.optimizer.apply_gradients(zip(hybrid_grads, params),
                                       global_step=tf.train.get_or_create_global_step())