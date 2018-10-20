import configparser
from collections import namedtuple

config_keys = ['batch_size',
               'learning_rate',
               'max_iters',
               'optimizer',
               'momentum'
               ]

ConfigTuple = namedtuple('config', config_keys)

defaults = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'max_iters': 1000000,
    'optimizer': 'momentum',
    'momentum': 0.9,
}


def parse_config(config_file):
    """read config.ini file with config parser,
    returns namedtuple ConfigTuple with following
    fields:

    batch_size : int
        size of batch to feed network. Default is 32.
    learning_rate : float
        learning rate for training network. Default is 1e-3.
    max_iters: int
        maximum number of iterations, i.e. training epochs.
        Default is 1000000.
    optimizer : str
        {'sgd', 'momentum'} where 'sgd' is Stochastic Gradient
        Descent (SGD) and 'momentum' is SGD with a momentum
        term. Default is 'momentum' (as in Mnih et al. 2014).
    momentum : float
        Value for momentum term. Only used if optimizer is
        'momentum'. Default is 0.9 (as in Mnih et al. 2014).
    """
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)

    config_vals = []
    for config_key in config_keys:
        if config_parser.has_option(option=config_key):
            config_vals.append(config_parser[config_key])
        else:
            config_vals.append(defaults[config_key])
    config_kwargs = dict(zip(config_keys, config_vals))
    config = ConfigTuple(**config_kwargs)
    return config
