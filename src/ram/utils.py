import os
from configparser import ConfigParser

import attr

this_file_dir = os.path.dirname(__file__)


@attr.s
class ModelConfig(object):
    """class that represents configuration for RAM model"""
    g_w = attr.ib(converter=int, default=8)
    k = attr.ib(converter=int, default=3)
    s = attr.ib(converter=int, default=2)
    hg_size = attr.ib(converter=int, default=128)
    hl_size = attr.ib(converter=int, default=128)
    g_size = attr.ib(converter=int, default=256)
    hidden_size = attr.ib(converter=int, default=256)
    glimpses = attr.ib(converter=int, default=6)
    num_classes = attr.ib(converter=int, default=10)
    loc_std = attr.ib(converter=float, default=0.1)

@attr.s
class TrainConfig(object):
    """class that represents configuration for training a RAM model"""
    batch_size = attr.ib(converter=int, default=10)
    learning_rate = attr.ib(converter=float, default=1e-3)
    epochs = attr.ib(converter=int, default=200)
    optimizer = attr.ib(type=str, default='momentum')
    momentum = attr.ib(converter=float, default=0.9)
    root_results_dir = attr.ib(type=str, default='.')
    replicates = attr.ib(converter=int, default=5)
    checkpoint_prefix = attr.ib(type=str, default='ckpt')
    restore = attr.ib(converter=bool, default=False)
    save_examples_every = attr.ib(converter=int, default=25)
    num_examples_to_save = attr.ib(converter=int, default=9)
    save_loss = attr.ib(converter=bool, default=False)
    shuffle_each_epoch = attr.ib(converter=bool, default=True)
    save_train_inds = attr.ib(converter=bool, default=False)

@attr.s
class DataConfig(object):
    """class that represents data associated with model:
    training data, testing data, checkpoints of trained model,
    outputs during training, etc."""
    output_dir = attr.ib(type=str, default='.')


@attr.s
class Config(object):
    model = attr.ib(type=ModelConfig, default=ModelConfig())
    train = attr.ib(type=TrainConfig, default=TrainConfig())


def parse_config(config_file=None):
    """read config.ini file with config parser,
    returns options from file loaded into an
    instance of Config class.

    Parameters
    ----------
    config_file : str
        Path to a config.ini file. If None, the
        default configuration is returned.
        Default is None.

    Returns
    -------
    config : instance of Config class
        with attributes that represent configuration for model, training, and data
    """
    if config_file is not None:
        if os.path.isfile(config_file):
            config = ConfigParser()
            config.read(config_file)
        else:
            raise FileNotFoundError(f'did not find config file: {config_file}')

        model_config = ModelConfig(**config['model'])
        train_config = TrainConfig(**config['train'])
        config = Config(model=model_config, train=train_config)
    else:
        # return default config
        config = Config()

    return config
