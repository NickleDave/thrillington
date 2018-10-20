import tensorflow as tf


class Trainer:
    def __init__(self,
                 data,
,
                 ):
        self.data = config.data
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.max_iters = config.max_iters
        self.model =
        if optimizer is None:
            self.optimizer = tf.train.MomentumOptimizer(momentum=0.9,
                                                   learning_rate=self.learning_rate)

    def train(self):
        """trains RAM model
        """
        for batch_num, (img, lbl) in self.data.batch(batch_size):
            self._train_one_epoch(self, img, lbl)

    def _train_one_epoch(self, img, lbl):
        for batch, (img, lbl) in enumerate(train_data.batch(batch_size)):
            # training loop
            out_t_minus_1 = ram_model.reset()

            locs = []
            mus = []
            log_pis = []
            baselines = []

            with tf.GradientTape(persistent=True) as tape:
                for t in range(ram_model.glimpses):
                    out = ram_model.step(img, out_t_minus_1.l_t, out_t_minus_1.h_t)

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
                                                         scale=ram_model.loc_std)
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
                predicted = tf.argmax(out.a_t, output_type=tf.int32)  # a_t = predictions from last time step
                R = tf.equal(predicted, lbl)
                R = tf.cast(R, dtype=tf.float32)
                # reshape reward to (batch size x number of glimpses)
                R = tf.expand_dims(R, axis=1)  # add axis
                R = tf.tile(R, tf.constant([1, ram_model.glimpses]))

                # compute losses for differentiable modules
                loss_action = tf.losses.softmax_cross_entropy(tf.one_hot(lbl, depth=ram_model.num_classes), out.a_t)
                loss_baseline = tf.losses.mean_squared_error(baselines, R)

                # compute loss for REINFORCE algorithm
                # summed over timesteps and averaged across batch
                adjusted_reward = R - baselines
                loss_reinforce = tf.reduce_sum((-log_pis * adjusted_reward), axis=1)
                loss_reinforce = tf.reduce_mean(loss_reinforce)

                # sum up into hybrid loss
                hybrid_loss = loss_action + loss_baseline + loss_reinforce

            reinforce_grads = tape.gradient(loss_reinforce, ram_model.location_network.variables)
            optimizer.apply_gradients(zip(reinforce_grads, ram_model.location_network.variables),
                                      global_step=tf.train.get_or_create_global_step())

            hybrid_grads = tape.gradient(hybrid_loss, [ram_model.glimpse_network.variables,
                                                       ram_model.action_network.variables,
                                                       ram_model.core_network.variables])
            optimizer.apply_gradients(zip(hybrid_grads, [ram_model.glimpse_network.variables,
                                                         ram_model.action_network.variables,
                                                         ram_model.core_network.variables]),
                                      global_step=tf.train.get_or_create_global_step())