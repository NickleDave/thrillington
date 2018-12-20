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

import tensorflow as tf

import ram

tf.enable_eager_execution()


def main():
    parser = argparse.ArgumentParser(description='main script',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str,
                        help='run experiment with configuration defined in config.ini file\n'
                             '$ ram-cli --config scripts/ram_configs/config_2018-12-17.ini')

    # get config first so we can know if we should save log, where to make results directory, etc.
    args = parser.parse_args()
    config = ram.parse_config(args.config)

    # start logging; instantiate logger through getLogger function
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')
    logger.addHandler(logging.StreamHandler(sys.stdout))

    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    results_dirname = 'RAM_results_' + timenow
    results_dir = os.path.join(config.data.root_results_dir, results_dirname)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    if config.train.save_log:
        logfile_name = os.path.join(results_dir,
                                    'logfile_from_ram_' + timenow + '.log')
        logger.addHandler(logging.FileHandler(logfile_name))
        logger.info('Logging results to {}'.format(results_dir))
        config.train.logfile_name = logfile_name

    logger.info(f'Using config file: {args.config}')

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

    logger.info(f'\nUsing {config.data.module} module to prepare and load datasets')
    paths_dict = dataset_module.prep(download_dir=config.data.data_dir,
                                     train_size=config.data.train_size,
                                     val_size=config.data.val_size,
                                     output_dir=config.data.data_dir)
    logger.info(f'Prepared dataset from {config.data.data_dir}')
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

    for replicate in range(1, config.train.replicates + 1):
        logger.info(f"Starting replicate {replicate}\n")
        config.train.current_replicate = replicate

        replicate_results_dir = os.path.join(results_dir, f'replicate_{replicate}')
        logger.info(f"Saving results in {replicate_results_dir}")
        if not os.path.isdir(replicate_results_dir):
            os.makedirs(replicate_results_dir)

        checkpoint_dir = os.path.join(replicate_results_dir, 'checkpoint')
        logger.info(f"Saving checkpoints in {checkpoint_dir}")
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        config.data.checkpoint_dir = checkpoint_dir

        if hasattr(config.data, 'save_examples_every'):
            logger.info(f"Will save examples every {config.data.save_examples_every} epochs")
            examples_dir = os.path.join(replicate_results_dir, 'examples')
            logger.info(f"Saving examples in {examples_dir}")
            if not os.path.isdir(examples_dir):
                os.makedirs(examples_dir)
            config.data.examples_dir = examples_dir
        else:
            logger.info("Will not save examples")

        if config.data.save_loss:
            loss_dir = os.path.join(replicate_results_dir, 'loss')
            logger.info(f"Saving loss in {loss_dir}")
            if not os.path.isdir(loss_dir):
                os.makedirs(loss_dir)
            config.data.loss_dir = loss_dir
        else:
            logger.info("Will not save record of loss")

        if config.data.save_train_inds:
            logger.info("Will save indices of samples from original training set")
            train_inds_dir = os.path.join(replicate_results_dir, 'train_inds')
            logger.info(f"Saving train_indices in {train_inds_dir}")
            if not os.path.isdir(train_inds_dir):
                os.makedirs(train_inds_dir)
            config.data.train_inds_dir = train_inds_dir
        else:
            logger.info("Will not save indices of samples from original training set")

        logger.info("\nStarting training.")
        trainer = ram.Trainer(config=config,
                              train_data=train_data,
                              val_data=val_data)
        trainer.train()

    logger.info("\nFinished run.")


if __name__ == '__main__':
    main()
