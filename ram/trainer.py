"""trainer for RAM model, from [1]_.
Based on two implementations:
https://github.com/kevinzakka/recurrent-visual-attention
https://github.com/seann999/tensorflow_mnist_ram

.. [1] Mnih, Volodymyr, Nicolas Heess, and Alex Graves.
   "Recurrent models of visual attention."
   Advances in neural information processing systems. 2014.
   https://arxiv.org/abs/1406.6247
"""
import time
from collections import namedtuple

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import ram

LossTuple = namedtuple('LossTuple', ['loss_reinforce',
                                     'loss_baseline',
                                     'loss_action',
                                     'loss_hybrid'])


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
        data : ram.mnist.dataset.Data
            subclass of NamedTuple that includes
            a tensorflow dataset object and the
            number of samples as fields.
            E.g., the MNIST training set,
            returned by calling ram.dataset.train.
        """
        # apply model config
        self.model = ram.RAM(**config.model._asdict())

        # then unpack train config
        self.data = data
        self.batch_size = config.train.batch_size
        self.learning_rate = config.train.learning_rate
        self.epochs = config.train.epochs
        if config.train.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(momentum=0.9,
                                                        learning_rate=self.learning_rate)
        elif config.train.optimizer == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

    def train(self):
        """trains RAM model
        """
        for epoch in range(self.epochs):
            print(
                '\nEpoch: {}/{} - learning rate: {:.6f}'.format(
                    epoch+1, self.epochs, self.learning_rate)
            )
            mean_acc, mean_loss = self._train_one_epoch()
            print(f'mean accuracy: {mean_acc}\nmean losses: {mean_loss}')

    def _train_one_epoch(self):
        """helper function that trains for one epoch.
        Called by trainer.train"""
        losses_reinforce = []
        losses_baseline = []
        losses_action = []
        losses_hybrid = []
        accs = []

        tic = time.time()

        with tqdm(total=self.data.num_samples) as progress_bar:
            for img, lbl in self.data.dataset.batch(self.batch_size):

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
                    # calculate reward.
                    # Remember that action network output a_t becomes predictions at last time step
                    predicted = tf.argmax(out.a_t, axis=1, output_type=tf.int32)
                    R = tf.equal(predicted, lbl)
                    acc = np.sum(R.numpy()) / R.numpy().shape[-1] * 100
                    accs.append(acc)
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
                    loss_hybrid = loss_action + loss_baseline + loss_reinforce

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
                hybrid_grads = tape.gradient(loss_hybrid, params)
                self.optimizer.apply_gradients(zip(hybrid_grads, params),
                                               global_step=tf.train.get_or_create_global_step())

                losses_reinforce.append(loss_reinforce.numpy())
                losses_baseline.append(loss_baseline.numpy())
                losses_action.append(loss_action.numpy())
                losses_hybrid.append(loss_hybrid.numpy())

                toc = time.time()

                progress_bar.set_description(
                    (
                        "{:.1f}s - hybrid loss: {:.3f} - acc: {:.3f}".format(
                            (toc-tic), loss_hybrid, acc)
                    )
                )
                progress_bar.update(self.batch_size)

        mn_acc = np.mean(accs)

        mn_loss_reinforce = np.asarray(losses_reinforce).mean()
        mn_loss_baseline = np.asarray(losses_baseline).mean()
        mn_loss_action = np.asarray(losses_action).mean()
        mn_losses_hybrid = np.asarray(losses_hybrid).mean()

        return mn_acc, LossTuple(mn_loss_reinforce,
                                 mn_loss_baseline,
                                 mn_loss_action,
                                 mn_losses_hybrid)
