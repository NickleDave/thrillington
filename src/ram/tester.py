""" "tester" for RAM model, from [1]_.
For measuring accuracy on a test set.

.. [1] Mnih, Volodymyr, Nicolas Heess, and Alex Graves.
   "Recurrent models of visual attention."
   Advances in neural information processing systems. 2014.
   https://arxiv.org/abs/1406.6247
"""
import os
import sys
import time
from collections import namedtuple
import logging

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import attr

from . import ram

LossesTuple = namedtuple('LossesTuple', ['reinforce_loss',
                                         'baseline_loss',
                                         'action_loss',
                                         'hybrid_loss',
                                         ])

MeanLossTuple = namedtuple('MeanLossTuple', ['mn_reinforce_loss',
                                             'mn_baseline_loss',
                                             'mn_action_loss',
                                             'mn_hybrid_loss',
                                             ])


class Tester:
    """Trainer object for training the RAM model"""
    def __init__(self,
                 config,
                 train_data,
                 val_data=None
                 ):
        """__init__ for Trainer"""
        self.model = ram.RAM(batch_size=config.train.batch_size,
                        **config.model._asdict())
        self.learning_rate = config.train.learning_rate

        if config.train.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(momentum=config.train.momentum,
                                                        learning_rate=self.learning_rate)
        elif config.train.optimizer == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        checkpointer = tf.train.Checkpoint(optimizer=self.optimizer,
                                           model=self.model,
                                           optimizer_step=tf.train.get_or_create_global_step())
        checkpoint_path = config.data.checkpoint_path
        checkpointer.restore(tf.train.latest_checkpoint(checkpoint_path))
        
        self.batch_size = config.train.batch_size
        self.test_data = test_data

    def get_test_acc(self, save_examples=False,
                     test_examples_dir='./test_examples/', num_examples_to_save=9):
        """compute accuracy of trained RAM model on test data set"""
        accs = []

        tic = time.time()

        with tqdm(total=test_data.num_samples) as progress_bar:
            batch = 0
            for img, lbl in test_data.dataset.batch(batch_size):
                batch += 1

                out_t_minus_1 = model.reset()

                if save_examples:
                    locs = []
                    fixations = []

                for t in range(model.glimpses):
                    out = model.step(img, out_t_minus_1.l_t, out_t_minus_1.h_t)

                    if save_examples:
                        locs.append(out.l_t.numpy()[:num_examples_to_save, :])
                        fixations.append(out.fixations[:num_examples_to_save, :])

                    out_t_minus_1 = out

                # repeat column vector n times where n = glimpses
                # calculate reward.
                # Remember that action network output a_t becomes predictions at last time step
                predicted = tf.argmax(out.a_t, axis=1, output_type=tf.int32)
                R = tf.equal(predicted, lbl)
                acc = np.sum(R.numpy()) / R.numpy().shape[-1] * 100
                accs.append(acc)

                toc = time.time()

                progress_bar.set_description(
                    (
                        "{:.1f}s - acc: {:.3f}".format((toc - tic), acc)
                    )
                )
                progress_bar.update(batch_size)

        if save_examples:
            locs = np.asarray(locs)
            locs.dump(os.path.join(test_examples_dir,
                                   f'locations_epoch_test'))
            fixations = np.asarray(fixations)
            fixations.dump(os.path.join(test_examples_dir,
                                        f'fixations_epoch_test'))
            glimpses = out.rho.numpy()[:num_examples_to_save]
            glimpses.dump(os.path.join(test_examples_dir,
                                       f'glimpses_epoch_test'))
            img = img.numpy()[:num_examples_to_save]
            img.dump(os.path.join(test_examples_dir,
                                  f'images_epoch_test'))

            pred = predicted.numpy()[:num_examples_to_save]
            pred.dump(os.path.join(test_examples_dir,
                                   f'predictions_epoch_test'))

        return accs