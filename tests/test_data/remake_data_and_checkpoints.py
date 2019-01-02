"""makes reduced size MNIST dataset we can use just to test everything runs without crashing.
Also make some checkpoints from training on reduced MNIST dataset,
so we can test that Tester runs with a checkpoint without crashing"""
from configparser import ConfigParser
from pathlib import Path

import ram

HERE = Path(__file__).resolve().parent


def main():
    config_parser = ConfigParser()
    config_parser.read(HERE.joinpath('../test_configs/quick_run_config.ini'))
    config_parser['data']['data_dir'] = str(HERE.joinpath('mnist'))  # will save data here
    config_parser['data']['root_results_dir'] = str(HERE)  # and also save results w/checkpoints here
    tmp_config = HERE.joinpath('tmp_config.ini')
    with open(tmp_config, 'w') as tmp_config_fp:
        config_parser.write(tmp_config_fp)
    ram.cli(command='train', configfile=tmp_config)


if __name__ == '__main__':
    main()
