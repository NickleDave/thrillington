import shutil
import tempfile
import unittest
from configparser import ConfigParser
from pathlib import Path
import importlib

import ram
import ram.__main__


HERE = Path(__file__).resolve().parent


class TestTrainer(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_data_dir = HERE.joinpath('test_data')
        self.test_config_dir = HERE.joinpath('test_configs')
        self.tmp_config_dir = Path(tempfile.mkdtemp())
        self.tmp_results_dir = Path(tempfile.mkdtemp())
        self.tmp_data_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        # Remove directories after the test
        shutil.rmtree(self.tmp_config_dir)
        shutil.rmtree(self.tmp_results_dir)
        shutil.rmtree(self.tmp_data_dir)

    def test_num_samples_batch_size_mismatch_raises(self):
        # __init__ should raise an error if
        # num samples % batch size != 0
        config_parser = ConfigParser()
        config_parser.read(self.test_config_dir.joinpath('bad_batch_size_config.ini'))
        config_parser['data']['data_dir'] = str(self.tmp_data_dir)
        config_parser['data']['root_results_dir'] = str(self.tmp_results_dir)
        tmp_config = str(self.tmp_config_dir.joinpath('tmp_config.ini'))
        with open(tmp_config, 'w') as tmp_config_fp:
            config_parser.write(tmp_config_fp)
        config = ram.parse_config(tmp_config)
        dataset_module = importlib.import_module(name=config.data.module)
        paths_dict = dataset_module.prep(download_dir=config.data.data_dir,
                                         train_size=config.data.train_size,
                                         val_size=config.data.val_size,
                                         output_dir=config.data.data_dir)
        train_data = dataset_module.get_split(paths_dict, setname='train')
        with self.assertRaises(ValueError):
            ram.Trainer.from_config(config=config, train_data=train_data)

    def test_with_quick_run_config(self):
        # this should run without failing
        config_parser = ConfigParser()
        config_parser.read(self.test_config_dir.joinpath('quick_run_config.ini'))
        config_parser['data']['data_dir'] = str(self.tmp_data_dir)
        config_parser['data']['root_results_dir'] = str(self.tmp_results_dir)
        tmp_config = str(self.tmp_config_dir.joinpath('tmp_config.ini'))
        with open(tmp_config, 'w') as tmp_config_fp:
            config_parser.write(tmp_config_fp)
        config = ram.parse_config(tmp_config)
        dataset_module = importlib.import_module(name=config.data.module)
        paths_dict = dataset_module.prep(download_dir=config.data.data_dir,
                                         train_size=config.data.train_size,
                                         val_size=config.data.val_size,
                                         output_dir=config.data.data_dir)
        train_data, val_data = dataset_module.get_split(paths_dict, setname=['train', 'val'])
        trainer = ram.Trainer.from_config(config=config, train_data=train_data, val_data=val_data)
        trainer.train(config.data.root_results_dir)


if __name__ == '__main__':
    unittest.main()
