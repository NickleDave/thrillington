"""
Invokes __main__ when the module is run as a script.
Example: python -m ram --help
The same function is run by the script `ram-cli` which is installed on the
path by pip, so `$ ram-cli --help` would have the same effect (i.e., no need
to type the python -m)
"""
import argparse

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

    for replicate in range(config.train.replicates):
        config.train.current_replicate = replicate
        trainer = ram.Trainer(config=config,
                              train_data=train_data,
                              val_data=val_data)
        trainer.train()


if __name__ == '__main__':
    main()
