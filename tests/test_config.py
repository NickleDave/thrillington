import os
import unittest
from configparser import ConfigParser

import ram.config

HERE = os.path.dirname(__file__)


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.test_configs_dir = os.path.join(HERE, 'test_configs')

    def test_TrainConfig_bool(self):
        config = ConfigParser()
        config.add_section('train')
        config['train']['save_log'] = 'True'
        a_train_config = ram.config.TrainConfig(**config['train'])
        self.assertTrue(a_train_config.save_log)
        config['train']['save_log'] = 'False'
        a_train_config = ram.config.TrainConfig(**config['train'])
        self.assertTrue(not a_train_config.save_log)

    def test_DataConfig_bool(self):
        config = ConfigParser()
        config.add_section('data')
        config['data']['root_results_dir'] = '.'
        config['data']['data_dir'] = '.'

        config['data']['save_loss'] = 'True'
        a_data_config = ram.config.DataConfig(**config['data'])
        self.assertTrue(a_data_config.save_loss)
        config['data']['save_loss'] = 'False'
        a_data_config = ram.config.DataConfig(**config['data'])
        self.assertTrue(not a_data_config.save_loss)
        config['data']['save_train_inds'] = 'True'
        a_data_config = ram.config.DataConfig(**config['data'])
        self.assertTrue(a_data_config.save_train_inds)
        config['data']['save_train_inds'] = 'False'
        a_data_config = ram.config.DataConfig(**config['data'])
        self.assertTrue(not a_data_config.save_train_inds)

    def test_DataConfig_float_or_None(self):
        config = ConfigParser()
        config.add_section('data')
        config['data']['root_results_dir'] = '.'
        config['data']['data_dir'] = '.'

        a_data_config = ram.config.DataConfig(**config['data'])
        self.assertTrue(a_data_config.train_size is None)
        self.assertTrue(a_data_config.val_size is None)
        config['data']['train_size'] = '0.7'
        config['data']['val_size'] = '0.3'
        a_data_config = ram.config.DataConfig(**config['data'])
        self.assertTrue(type(a_data_config.train_size) is float)
        self.assertTrue(type(a_data_config.val_size) is float)

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
