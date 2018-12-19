import os
import unittest

import ram.config

HERE = os.path.dirname(__file__)


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.test_configs_dir = os.path.join(HERE, 'test_configs')

    def test_good_config(self):
        config_file = os.path.join(self.test_configs_dir, 'good_config.ini')
        good_config = ram.config.parse_config(config_file=config_file)
        self.assertTrue(type(good_config) == ram.config.Config)

    def test_missing_data_dir_raises(self):
        config_file = os.path.join(self.test_configs_dir, 'missing_data_dir_config.ini')
        with self.assertRaises(ValueError):
            ram.config.parse_config(config_file=config_file)

    def test_missing_root_results_dir_raises(self):
        config_file = os.path.join(self.test_configs_dir, 'missing_root_results_dir_config.ini')
        with self.assertRaises(ValueError):
            ram.config.parse_config(config_file=config_file)

    def test_invalid_model_keys_raises(self):
        config_file = os.path.join(self.test_configs_dir, 'invalid_model_keys_config.ini')
        with self.assertRaises(ValueError):
            ram.config.parse_config(config_file=config_file)

    def test_invalid_train_keys_raises(self):
        config_file = os.path.join(self.test_configs_dir, 'invalid_train_keys_config.ini')
        with self.assertRaises(ValueError):
            ram.config.parse_config(config_file=config_file)

    def test_invalid_data_keys_raises(self):
        config_file = os.path.join(self.test_configs_dir, 'invalid_data_keys_config.ini')
        with self.assertRaises(ValueError):
            ram.config.parse_config(config_file=config_file)


if __name__ == '__main__':
    unittest.main()
