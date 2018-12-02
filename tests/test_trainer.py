import shutil
import tempfile
import unittest

import ram


class TestInit(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_num_samples_batch_size_mismatch(self):
        # __init__ should raise an error if
        # num samples % batch size != 0
        config = src.ram.utils.parse_config(config_file=None)  # default batch size is 10
        data = src.ram.mnist.dataset.train(directory=self.test_dir, num_samples=23)
        with self.assertRaises(ValueError):
            ram.Trainer(config, data)


if __name__ == '__main__':
    unittest.main()
