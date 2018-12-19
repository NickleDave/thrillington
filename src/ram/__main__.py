"""
Invokes __main__ when the module is run as a script.
Example: python -m ram --help
The same function is run by the script `ram-cli` which is installed on the
path by pip, so `$ ram-cli --help` would have the same effect (i.e., no need
to type the python -m)
"""
import os
import argparse
from datetime import datetime

import tensorflow as tf

import ram

tf.enable_eager_execution()


def main():
    parser = argparse.ArgumentParser(description='main script',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str,
                        help='run experiment with configuration defined in config.ini file\n'
                             '$ ram-cli --config scripts/ram_configs/config_2018-12-17.ini')
    args = parser.parse_args()
    config = ram.parse_config(args.config)

    if config.data.type == 'mnist':
        paths_dict = ram.dataset.mnist.prep(download_dir=config.data.data_dir,
                                            train_size=config.data.train_size,
                                            val_size=config.data.val_size,
                                            output_dir=config.data.data_dir)
        if config.data.val_size:
            train_data, val_data = ram.dataset.mnist.get_split(paths_dict, setname=['train', 'val'])
        else:
            train_data = ram.dataset.mnist.get_split(paths_dict, setname=['train'])
            val_data = None

    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    results_dirname = 'RAM_results_' + timenow
    results_dir = os.path.join(config.data.root_results_dir, results_dirname)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    for replicate in range(1, config.train.replicates + 1):
        config.train.current_replicate = replicate

        replicate_results_dir = os.path.join(results_dir, f'replicate_{replicate}')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        checkpoint_dir = os.path.join(replicate_results_dir, 'checkpoint')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        config.data.checkpoint_dir = checkpoint_dir

        if hasattr(config.data, 'save_examples_every'):
            examples_dir = os.path.join(replicate_results_dir, 'examples')
            if not os.path.isdir(examples_dir):
                os.makedirs(examples_dir)
            config.data.examples_dir = examples_dir

        if config.data.save_loss:
            loss_dir = os.path.join(replicate_results_dir, 'loss')
            if not os.path.isdir(loss_dir):
                os.makedirs(loss_dir)
            config.data.loss_dir = loss_dir

        if config.data.save_train_inds:
            train_inds_dir = os.path.join(replicate_results_dir, 'train_inds')
            if not os.path.isdir(train_inds_dir):
                os.makedirs(train_inds_dir)
            config.data.train_inds_dir = train_inds_dir

        trainer = ram.Trainer(config=config,
                              train_data=train_data,
                              val_data=val_data)
        trainer.train()


if __name__ == '__main__':
    main()
