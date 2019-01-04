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
from datetime import datetime

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
                 save_log,
                 replicates=1,
                 logger=None,
                 ):
        """__init__ for Tester"""
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
                   save_log=config.misc.save_log,
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

    def test(self, results_dir, save_examples=False, num_examples_to_save=None, test_examples_dir=None):
        """compute accuracy of trained RAM models on test data set"""
        if not os.path.isdir(results_dir):
            raise NotADirectoryError(f"Couldn't find directory with results: {results_dir}")
        timenow = datetime.now().strftime('%y%m%d_%H%M%S')
        test_results_dir = os.path.join(results_dir, 'test_results_' + timenow)
        os.makedirs(test_results_dir)
        for replicate in range(1, self.replicates + 1):
            replicate_path = os.path.join(results_dir, f'replicate_{replicate}')
            if not os.path.isdir(replicate_path):
                raise NotADirectoryError(f"Couldn't find directory for replicate: {replicate_path}")
            self.logger.info(f'replicate {replicate} of {self.replicates}')
            self.model = ram.RAM(batch_size=self.batch_size,
                                 **attr.asdict(self.config.model))
            self.checkpointer = tf.train.Checkpoint(optimizer=self.optimizer,
                                                    model=self.model,
                                                    optimizer_step=tf.train.get_or_create_global_step())
            checkpoint_path = os.path.join(replicate_path, 'checkpoint')
            self.load_checkpoint(checkpoint_path)

            test_results_dir_this_replicate = os.path.join(test_results_dir, f'replicate_{replicate}')
            os.makedirs(test_results_dir_this_replicate)
            if save_examples and test_examples_dir is None:
                # if save examples is True but test_examples_dir not specified
                # (e.g. because called by cli), then make test_examples_dir
                test_examples_dir = os.path.join(test_results_dir_this_replicate, 'examples')
                os.makedirs(test_examples_dir)

            (accs,
             preds,
             true_lbl,
             sample_inds) = self._test_one_model(save_examples=save_examples,
                                                 num_examples_to_save=num_examples_to_save,
                                                 test_examples_dir=test_examples_dir)

            for name, arr in zip(['accs', 'preds', 'true_lbl', 'sample_inds'],
                                 [accs, preds, true_lbl, sample_inds]):
                fname = os.path.join(test_results_dir_this_replicate, name + '.py')
                np.save(fname, arr)

            self.logger.info(f'mean accuracy on test data set: {np.mean(accs)}')

    def _test_one_model(self, save_examples, num_examples_to_save, test_examples_dir):

        accs = []
        sample_inds = []
        preds = []
        true_lbl = []

        if save_examples:
            locs = []
            fixations = []
            glimpses = []
            img_to_save = []
            pred = []
            num_examples_saved = 0

        tic = time.time()

        with tqdm(total=self.num_test_samples) as progress_bar:
            batch = 0
            for img, lbl, batch_sample_inds in self.test_data.batch(self.batch_size):
                sample_inds.append(batch_sample_inds)
                batch += 1

                out_t_minus_1 = self.model.reset()

                if save_examples:
                    if num_examples_saved < num_examples_to_save:
                        locs_t = []
                        fixations_t = []
                        glimpses_t = []

                for t in range(self.model.glimpses):
                    out = self.model.step(img, out_t_minus_1.l_t, out_t_minus_1.h_t)

                    if save_examples:
                        if num_examples_saved < num_examples_to_save:
                            locs_t.append(out.l_t.numpy())
                            fixations_t.append(out.fixations)
                            glimpses_t.append(out.rho.numpy())

                    out_t_minus_1 = out

                # repeat column vector n times where n = glimpses
                # calculate reward.
                # Remember that action network output a_t becomes predictions at last time step
                predicted = tf.argmax(out.a_t, axis=1, output_type=tf.int32)
                preds.append(predicted)
                R = tf.equal(predicted, lbl)
                acc = np.sum(R.numpy()) / R.numpy().shape[-1] * 100
                accs.append(acc)
                true_lbl.append(lbl)

                # deal with examples if we are saving them
                if save_examples:
                    # note we save the **first** n samples
                    if num_examples_saved < num_examples_to_save:
                        # stack so axis 0 is sample index, axis 1 is number of glimpses
                        locs_t = np.stack(locs_t, axis=1)
                        fixations_t = np.stack(fixations_t, axis=1)
                        glimpses_t = np.stack(glimpses_t, axis=1)

                        num_samples = locs_t.shape[0]

                        if num_examples_saved + num_samples <= num_examples_to_save:
                            locs.append(locs_t)
                            fixations.append(fixations_t)
                            glimpses.append(glimpses_t)
                            pred.append(predicted)
                            img_to_save.append(img)

                            num_examples_saved = num_examples_saved + num_samples

                        elif num_examples_saved + num_samples > num_examples_to_save:
                            num_needed = num_examples_to_save - num_examples_saved

                            locs.append(locs_t[:num_needed])
                            fixations.append(fixations_t[:num_needed])
                            glimpses.append(glimpses_t[:num_needed])
                            pred.append(predicted[:num_needed])
                            img_to_save.append(img[:num_needed])

                            num_examples_saved = num_examples_saved + num_needed

                toc = time.time()

                progress_bar.set_description(
                    (
                        "{:.1f}s - acc: {:.3f}".format((toc - tic), acc)
                    )
                )
                progress_bar.update(self.batch_size)

        if save_examples:
            for arr, stem in zip(
                    (locs, fixations, glimpses, img_to_save, pred),
                    ('locations', 'fixations', 'glimpses', 'images', 'predictions')
            ):
                arr = np.concatenate(arr)
                file = os.path.join(test_examples_dir,
                                    f'{stem}_epoch_test')
                np.save(file=file, arr=arr)

        accs = np.asarray(accs)
        preds = np.concatenate(preds)
        true_lbl = np.concatenate(true_lbl)
        sample_inds = np.concatenate(sample_inds)

        return accs, preds, true_lbl, sample_inds
