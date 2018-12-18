import os
import tempfile
import shutil

import numpy as np
import tensorflow as tf

import ram.dataset.mnist

tfe = tf.contrib.eager


class TestDatasetMNIST(tf.test.TestCase):
    def setUp(self):
        self.tmp_output_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_output_dir)

    @tfe.run_test_in_graph_and_eager_modes
    def test_normalize(self):
        fake_mnist_image = np.random.choice(a=np.arange(255), size=(28*28))
        normed = ram.dataset.mnist.normalize(fake_mnist_image)
        if tf.executing_eagerly():
            self.assertTrue(tf.greater_equal(tf.reduce_min(normed), 0).numpy())
            self.assertTrue(tf.less_equal(tf.reduce_max(normed), 1).numpy())
        else:
            self.assertTrue(tf.greater_equal(tf.reduce_min(normed), 0).eval())
            self.assertTrue(tf.less_equal(tf.reduce_max(normed), 1).eval())

    @tfe.run_test_in_graph_and_eager_modes
    def test_prep_default(self):
        paths_dict = ram.dataset.mnist.prep(download_dir=self.tmp_output_dir, val_size=None,
                                            random_seed=None, output_dir=self.tmp_output_dir)
        self.assertTrue(list(paths_dict.keys()) == ['train', 'test'])
        for split, paths_mapping in paths_dict.items():
            for file_type, file_path in paths_mapping.items():
                self.assertTrue(os.path.isfile(file_path))

    @tfe.run_test_in_graph_and_eager_modes
    def test_prep_with_val(self):
        # make sure prep function works when validation set is created by specifying val_size
        paths_dict = ram.dataset.mnist.prep(download_dir=self.tmp_output_dir, val_size=0.25,
                                            random_seed=None, output_dir=self.tmp_output_dir)
        self.assertTrue(list(paths_dict.keys()) == ['train', 'test', 'val'])
        for split, paths_mapping in paths_dict.items():
            for file_type, file_path in paths_mapping.items():
                self.assertTrue(os.path.isfile(file_path))
        train_inds = np.load(paths_dict['train']['sample_inds'])
        val_inds = np.load(paths_dict['val']['sample_inds'])
        self.assertTrue(not any([val_ind in train_inds for val_ind in val_inds]))


if __name__ == '__main__':
    tf.test.main()
