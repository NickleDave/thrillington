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

from ram import ram

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
    """Class for measuring accuracy of trained RAM model on a test data set"""
    def __init__(self,
                 config,
                 batch_size,
                 learning_rate,
                 optimizer,
                 test_data,
                 replicates=1,
                 logger=None
                 ):
        """__init__ for Trainer"""
        self.config = config
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.save_log = save_log  # if True, will create logfile in train method, using that method's result_dir arg
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel('INFO')
            self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.test_data = test_data.dataset
        self.logger.info(f'Test data: {self.test_data}')
        self.num_test_samples = test_data.num_samples
        self.logger.info(f'Number of samples in test data: {self.num_test_samples}')

        self.replicates = replicates

        # below get replaced with an object when we start training
        self.checkpointer = None
        self.model = None

    @classmethod
    def from_config(cls, config, test_data, logger=None):
        if config.train.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(momentum=config.train.momentum,
                                                   learning_rate=config.train.learning_rate)
        elif config.train.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.train.learning_rate)

        return cls(config=config,
                   batch_size=config.train.batch_size,
                   learning_rate=config.train.learning_rate,
                   optimizer=optimizer,
                   test_data=test_data,
                   replicates=config.train.replicates,
                   logger=logger)

    def load_checkpoint(self, checkpoint_path):
        """loads model and optimizer from a checkpoint.
        Called when config.train.restore is True"""
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint:
            self.logger.info(f'restoring model from latest checkpoint: {latest_checkpoint}')
            self.checkpointer.restore(latest_checkpoint)
        else:
            raise ValueError(f'no checkpoint found in checkpoint path: {checkpoint_path}')

    def test(self, results_dir):
        self.model = ram.RAM(batch_size=self.batch_size,
                             **attr.asdict(self.config.model))
        checkpointer = tf.train.Checkpoint(optimizer=self.optimizer,
                                           model=self.model,
                                           optimizer_step=tf.train.get_or_create_global_step())
        checkpointer.restore(tf.train.latest_checkpoint(checkpoint_path))
        for replicate in range(1, self.replicates + 1):
            self.logger.info(f'replicate {replicate} of {self.replicates}')

            self._test_one_model()

    def _test_one_model(self, test_data, save_examples=False, test_examples_dir='./test_examples/',
             num_examples_to_save=9):
        """compute accuracy of trained RAM model on test data set"""
        accs = []
        sample_inds = []
        preds = []

        tic = time.time()

        with tqdm(total=test_data.num_samples) as progress_bar:
            batch = 0
            for img, lbl, batch_sample_inds in test_data.dataset.batch(self.batch_size):
                sample_inds.append(batch_sample_inds)
                batch += 1

                out_t_minus_1 = self.model.reset()

                if save_examples:
                    locs = []
                    fixations = []

                for t in range(self.model.glimpses):
                    out = self.model.step(img, out_t_minus_1.l_t, out_t_minus_1.h_t)

                    if save_examples:
                        locs.append(out.l_t.numpy()[:num_examples_to_save, :])
                        fixations.append(out.fixations[:num_examples_to_save, :])

                    out_t_minus_1 = out

                # repeat column vector n times where n = glimpses
                # calculate reward.
                # Remember that action network output a_t becomes predictions at last time step
                predicted = tf.argmax(out.a_t, axis=1, output_type=tf.int32)
                preds.append(predicted)
                R = tf.equal(predicted, lbl)
                acc = np.sum(R.numpy()) / R.numpy().shape[-1] * 100
                accs.append(acc)

                toc = time.time()

                progress_bar.set_description(
                    (
                        "{:.1f}s - acc: {:.3f}".format((toc - tic), acc)
                    )
                )
                progress_bar.update(self.batch_size)

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

        return accs, preds, sample_inds
