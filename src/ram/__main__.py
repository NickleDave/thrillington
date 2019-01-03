"""
Invokes __main__ when the module is run as a script.
Example: python -m ram --help
The same function is run by the script `ram-cli` which is installed on the
path by pip, so `$ ram-cli --help` would have the same effect (i.e., no need
to type the python -m)
"""
import os
import sys
import argparse
from datetime import datetime
import logging
import importlib
import importlib.util
from configparser import ConfigParser
import json

import tensorflow as tf

import ram

tf.enable_eager_execution()


def add_option_to_config_file(config_file, section, option, value):
    config_parser = ConfigParser()
    config_parser.read(config_file)
    config_parser[section][option] = value
    with open(config_file, 'w') as config_file_obj:
        config_parser.write(config_file_obj)


def add_FileHandlerto_logger(logger, results_dir, command, timenow):
    logfile_name = os.path.join(results_dir,
                                f'logfile_from_ram_{command}_{timenow}.log')
    logger.addHandler(logging.FileHandler(logfile_name))
    logger.info('Logging results to {}'.format(results_dir))


def cli(command, configfile):
    """command-line interface
    Called by main() when user runs ram from the command-line by typing 'ram'

    Parameters
    ----------
    command : str
        Command to follow. One of {'train', 'test'}
            Train : train models using configuration defined in config file.
            Test : test accuracy of trained models using configuration defined in configfile.

    configfile : str
        Path to a `config.ini` file that defines the configuration.

    Returns
    -------
    None

    Examples
    --------
    >>> cli(command='train', config='./configs/quick_run_config.ini')

    Notes
    -----
    This function is not really meant to be run by the user, but has its own arguments
    to make it easier to test (instead of throwing everything into one 'main' function)
    """
    # get config first so we can know if we should save log, where to make results directory, etc.
    config = ram.parse_config(configfile)

    tf.random.set_random_seed(config.misc.random_seed)

    # start logging; instantiate logger through getLogger function
    logger = logging.getLogger('ram-cli')
    logger.setLevel('INFO')
    logger.addHandler(logging.StreamHandler(sys.stdout))

    try:
        dataset_module = importlib.import_module(name=config.data.module)
    except ModuleNotFoundError:
        if os.path.isfile(config.data.module):
            module_name = os.path.basename(config.data.module)
            if module_name.endswith('.py'):
                module_name = module_name.replace('.py', '')
        else:
            raise FileNotFoundError(f'{config.data.module} could not be imported, and not recognized as a file')
        spec = importlib.util.spec_from_file_location(name=module_name, location=config.data.module)
        dataset_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset_module)

    if command == 'train':
        timenow = datetime.now().strftime('%y%m%d_%H%M%S')
        results_dirname = 'RAM_results_' + timenow
        results_dir = os.path.join(config.data.root_results_dir, results_dirname)
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        add_option_to_config_file(configfile, 'data', 'results_dir_made_by_main', results_dir)

        if config.misc.save_log:
            add_FileHandlerto_logger(logger=logger, results_dir=results_dir, command=command, timenow=timenow)

        logger.info(f'Used config file: {configfile}')
        logger.info(f'Used random seed: {config.misc.random_seed}')

        logger.info("\nRunning main in 'train' mode, will train new models.")

        logger.info(f'\nUsing {config.data.module} module to prepare and load datasets')
        paths_dict = dataset_module.prep(download_dir=config.data.data_dir,
                                         train_size=config.data.train_size,
                                         val_size=config.data.val_size,
                                         output_dir=config.data.data_dir)
        logger.info(f'Prepared dataset from {config.data.data_dir}')
        paths_dict_fname = os.path.join(results_dir, 'paths_dict.json')
        with open(paths_dict_fname, 'w') as paths_dict_json:
            json.dump(paths_dict, paths_dict_json)
        logger.info(f'Saved paths to files prepared for dataset in {paths_dict_fname}')

        logger.info(f'train size (None = use all training data): {config.data.train_size}')
        logger.info(f'val size (None = no validation set): {config.data.val_size}')
        logger.info(f'saved .npy files in: {config.data.data_dir}')
        logger.info(f"Full paths to files returned by dataset.mnist.prep:\n{paths_dict}")
        if config.data.val_size:
            logger.info(f'Will use validation data set')
            train_data, val_data = dataset_module.get_split(paths_dict, setname=['train', 'val'])
        else:
            train_data = dataset_module.get_split(paths_dict, setname=['train'])
            val_data = None

            logger.info("\nStarting training.")

        trainer = ram.Trainer.from_config(config=config,
                                          train_data=train_data,
                                          val_data=val_data,
                                          logger=logger)
        trainer.train(results_dir=results_dir)

    elif command == 'test':
        results_dir = config.data.results_dir_made_by_main
        timenow = datetime.now().strftime('%y%m%d_%H%M%S')
        if config.misc.save_log:
            add_FileHandlerto_logger(logger=logger, results_dir=results_dir, command=command, timenow=timenow)

        logger.info(f'Used config file: {configfile}')
        logger.info(f'Used random seed: {config.misc.random_seed}')

        logger.info("\nRunning main in 'test' mode, will test accuracy of previously trained models\n"
                    "on the test data set.")
        paths_dict_fname = os.path.join(results_dir, 'paths_dict.json')
        with open(paths_dict_fname) as paths_dict_json:
            paths_dict = json.load(paths_dict_json)
        logger.info(f'\nLoading test data from path in {paths_dict_fname}, ')
        logger.info(f'\nUsing {config.data.module} module to load dataset')
        test_data = dataset_module.get_split(paths_dict, setname=['test'])
        tester = ram.Tester.from_config(config=config, test_data=test_data, logger=logger)
        tester.test(results_dir=results_dir, save_examples=config.test.save_examples,
                    num_examples_to_save=config.test.num_examples_to_save)

    logger.info("\nFinished running.")


def get_parser():
    parser = argparse.ArgumentParser(description='main script',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('command', type=str, choices=['train', 'test'],
                        help="Command to run, either 'train' or 'test' \n"
                             "$ ram train scripts/ram_configs/config_2018-12-17.ini")
    parser.add_argument('configfile', type=str,
                        help='name of config.ini file to use \n'
                             '$ ram train scripts/ram_configs/config_2018-12-17.ini')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    cli(command=args.command,
        configfile=args.configfile)


if __name__ == '__main__':
    main()
