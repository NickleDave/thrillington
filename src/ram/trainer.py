"""trainer for RAM model, from [1]_.
Based on two implementations:
https://github.com/kevinzakka/recurrent-visual-attention
https://github.com/seann999/tensorflow_mnist_ram

.. [1] Mnih, Volodymyr, Nicolas Heess, and Alex Graves.
   "Recurrent models of visual attention."
   Advances in neural information processing systems. 2014.
   https://arxiv.org/abs/1406.6247
"""
import os
import time
from collections import namedtuple

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from . import ram

LossTuple = namedtuple('LossTuple', ['loss_reinforce',
                                     'loss_baseline',
                                     'loss_action',
                                     'loss_hybrid'])


class Trainer:
    """Trainer object for training the RAM model"""
    def __init__(self,
                 config,
                 train_data,
                 val_data=None
                 ):
        """__init__ for Trainer

        Parameters
        ----------
        config : namedtuple
            as returned by ram.utils.parse_config.
        train_data : ram.dataset.Dataset
            Training data.
            Named tuple with fields:
                dataset : tensorflow Dataset object with images and labels / correct action
                num_samples : number of samples, int
            E.g., the MNIST training set,
            returned by calling ram.dataset.train.
        val_data : ram.dataset.Dataset
            Validation data. Default is None (in which case a validation score is not computed).
        """
        if train_data.num_samples % config.train.batch_size != 0:
            raise ValueError(f'Number of training samples, {train_data.num_samples}, '
                             f'is not evenly divisible by batch size, {config.train.batch_size}.\n'
                             f'This will cause an error when training network;'
                             f'please change either so that data.num_samples % config.train.batch_size == 0:')
        # apply model config
        self.model = ram.RAM(batch_size=config.train.batch_size,
                             **config.model._asdict())

        # then unpack train config
        self.train_data = train_data.dataset
        self.num_train_samples = train_data.num_samples

        if val_data:
            self.val_data = val_data.dataset
            self.num_val_samples = val_data.num_samples

        self.batch_size = config.train.batch_size
        self.learning_rate = config.train.learning_rate
        self.epochs = config.train.epochs
        if config.train.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(momentum=0.9,
                                                        learning_rate=self.learning_rate)
        elif config.train.optimizer == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.restore = config.train.restore
        self.checkpoint_dir = os.path.abspath(config.train.checkpoint_dir)
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_path = os.path.join(config.train.checkpoint_dir,
                                            config.train.checkpoint_prefix)
        self.checkpointer = tf.train.Checkpoint(optimizer=self.optimizer,
                                                model=self.model,
                                                optimizer_step=tf.train.get_or_create_global_step())
        if hasattr(config.train, 'save_examples_every'):
            self.save_examples_every = config.train.save_examples_every
            self.examples_dir = os.path.abspath(config.train.examples_dir)
            if not os.path.isdir(self.examples_dir):
                os.makedirs(self.examples_dir)
            self.num_examples_to_save = config.train.num_examples_to_save

    def load_checkpoint(self):
        """loads model and optimizer from a checkpoint.
        Called when config.train.restore is True"""
        self.checkpointer.restore(
            tf.train.latest_checkpoint(self.checkpoint_path))

    def save_checkpoint(self):
        """save model and optimizer to a checkpoint file"""
        self.checkpointer.save(file_prefix=self.checkpoint_path)

    def train(self):
        """trains RAM model
        """
        if self.restore:
            print('config.train.restore is True,\n'
                  f'loading model and optimizer from checkpoint: {self.checkpoint_path}')
            self.load_checkpoint()
        else:
            print('config.train.resume is False,\n'
                  f'will save new model and optimizer to checkpoint: {self.checkpoint_path}')
        for epoch in range(1, self.epochs+1):
            print(
                f'\nEpoch: {epoch}/{self.epochs} - learning rate: {self.learning_rate:.6f}'
            )

            if epoch % self.save_examples_every == 0:
                save_examples = True
            else:
                save_examples = False

            mean_acc_dict, mean_loss = self._train_one_epoch(current_epoch=epoch, save_examples=save_examples)
            # violating DRY by unpacking dict into vars,
            # because apparently format strings with dict keys blow up the PyCharm parser
            mn_acc = mean_acc_dict['mn_acc']
            if 'mn_val_acc' in mean_acc_dict:
                mn_val_acc = mean_acc_dict['mn_val_acc']
                print(f'mean accuracy: {mn_acc}\n'
                      f'mean validation accuracy: {mn_val_acc}\n'
                      f'mean losses: {mean_loss}')
            else:
                print(f'mean accuracy: {mn_acc}\nmean losses: {mean_loss}')
            self.save_checkpoint()

    def _train_one_epoch(self, current_epoch, save_examples=False):
        """helper function that trains for one epoch.
        Called by trainer.train"""
        losses_reinforce = []
        losses_baseline = []
        losses_action = []
        losses_hybrid = []
        accs = []

        tic = time.time()

        with tqdm(total=self.num_train_samples) as progress_bar:
            for img, lbl in self.train_data.batch(self.batch_size):

                out_t_minus_1 = self.model.reset()

                mus = []
                log_pis = []
                baselines = []
                if save_examples:
                    locs = []
                    fixations = []

                with tf.GradientTape(persistent=True) as tape:
                    for t in range(self.model.glimpses):
                        out = self.model.step(img, out_t_minus_1.l_t, out_t_minus_1.h_t)

                        mus.append(out.mu)
                        baselines.append(out.b_t)
                        if save_examples:
                            locs.append(out.l_t.numpy()[:self.num_examples_to_save, :])
                            fixations.append(out.fixations[:self.num_examples_to_save, :])

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
                    loss_action = tf.losses.softmax_cross_entropy(tf.one_hot(lbl, depth=self.model.num_classes),
                                                                  out.a_t)
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

        if save_examples:
            locs = np.asarray(locs)
            locs.dump(os.path.join(self.examples_dir,
                                   f'locations_epoch_{current_epoch}'))
            fixations = np.asarray(fixations)
            fixations.dump(os.path.join(self.examples_dir,
                                        f'fixations_epoch_{current_epoch}'))
            glimpses = out.rho.numpy()[:self.num_examples_to_save]
            glimpses.dump(os.path.join(self.examples_dir,
                                       f'glimpses_epoch_{current_epoch}'))
            img = img.numpy()[:self.num_examples_to_save]
            img.dump(os.path.join(self.examples_dir,
                                  f'images_epoch_{current_epoch}'))

            pred = predicted.numpy()[:self.num_examples_to_save]
            pred.dump(os.path.join(self.examples_dir,
                                   f'predictions_epoch_{current_epoch}'))

            if self.val_data:
                print('calculating validation accuracy')
                val_accs = []
                with tqdm(total=self.num_val_samples) as progress_bar:
                    for img, lbl in self.val_data.batch(self.batch_size):
                        out_t_minus_1 = self.model.reset()
                        for t in range(self.model.glimpses):
                            out = self.model.step(img, out_t_minus_1.l_t, out_t_minus_1.h_t)
                            out_t_minus_1 = out

                        # Remember that action network output a_t becomes predictions at last time step
                        predicted = tf.argmax(out.a_t, axis=1, output_type=tf.int32)
                        val_acc = tf.equal(predicted, lbl)
                        val_acc = np.sum(val_acc.numpy()) / val_acc.numpy().shape[-1] * 100
                    progress_bar.update(self.batch_size)
                    val_accs.append(val_acc)
                mn_val_acc = np.mean(val_accs)

        mn_acc_dict = {'mn_acc': np.mean(accs)}
        if self.val_data:
            mn_acc_dict['mn_val_acc'] = mn_val_acc

        mn_loss_reinforce = np.asarray(losses_reinforce).mean()
        mn_loss_baseline = np.asarray(losses_baseline).mean()
        mn_loss_action = np.asarray(losses_action).mean()
        mn_losses_hybrid = np.asarray(losses_hybrid).mean()

        return mn_acc_dict, LossTuple(mn_loss_reinforce,
                                      mn_loss_baseline,
                                      mn_loss_action,
                                      mn_losses_hybrid)
