import os
from configparser import ConfigParser

import attr

VALID_OPTIONS = {
    'model': [
        'g_w',
        'k',
        's',
        'hg_size',
        'hl_size',
        'g_size',
        'hidden_size',
        'glimpses',
        'num_classes',
        'loc_std'
    ],
    'train': [
        'batch_size',
        'learning_rate',
        'decay_rate',
        'epochs',
        'optimizer',
        'beta1',
        'beta2',
        'epsilon',
        'momentum',
        'replicates',
        'checkpoint_prefix',
        'restore',
        'shuffle_each_epoch',
        'save_examples_every',
        'num_examples_to_save',
        'save_loss',
        'save_train_inds',

    ],
    'data': [
        'root_results_dir',
        'data_dir',
        'module',
        'train_size',
        'val_size',
        'results_dir_made_by_main'
    ],
    'test': [
        'save_examples',
        'num_examples_to_save',
    ],
    'misc': [
        'save_log',
        'random_seed'
    ]
}

VALID_SECTIONS = set(VALID_OPTIONS.keys())


# adapted from cPython/Lib/distutils/util.py
def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


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
class DataConfig(object):
    """class that represents data associated with model:
    training data, testing data, checkpoints of trained model,
    outputs during training, etc."""
    root_results_dir = attr.ib(type=str)
    data_dir = attr.ib(type=str)
    module = attr.ib(type=str, default='mnist')
    train_size = attr.ib(converter=attr.converters.optional(float), default=None)
    val_size = attr.ib(converter=attr.converters.optional(float), default=None)
    # below gets added by main script to config file during training so it can be used
    # when measuring accuracy on test set
    results_dir_made_by_main = attr.ib(type=attr.converters.optional(str), default=None)


@attr.s
class TrainConfig(object):
    """class that represents configuration for training a RAM model"""
    batch_size = attr.ib(converter=int, default=10)
    learning_rate = attr.ib(converter=float, default=1e-3)
    decay_rate = attr.ib(converter=attr.converters.optional(float), default=None)
    epochs = attr.ib(converter=int, default=200)
    optimizer = attr.ib(type=str, default='momentum')
    beta1 = attr.ib(converter=float, default=0.9)
    beta2 = attr.ib(converter=float, default=0.999)
    epsilon = attr.ib(converter=float, default=1e-08)
    momentum = attr.ib(converter=float, default=0.9)
    replicates = attr.ib(converter=int, default=5)
    checkpoint_prefix = attr.ib(type=str, default='ckpt')
    restore = attr.ib(converter=strtobool, default='False')
    shuffle_each_epoch = attr.ib(converter=strtobool, default='True')
    save_examples_every = attr.ib(converter=int, default=25)
    num_examples_to_save = attr.ib(converter=int, default=9)
    save_loss = attr.ib(converter=strtobool, default='False')
    save_train_inds = attr.ib(converter=strtobool, default='False')
    # user does not specify current replicate, gets changed by main()
    current_replicate = attr.ib(type=int, default=None)


@attr.s
class TestConfig(object):
    """class that represents configuration for testing a RAM model"""
    save_examples = attr.ib(converter=strtobool, default='True')
    num_examples_to_save = attr.ib(converter=attr.converters.optional(int), default=None)


@attr.s
class MiscConfig(object):
    """class that represents configuration parameters that do not fit in sections above"""
    save_log = attr.ib(converter=strtobool, default='True')
    random_seed = attr.ib(converter=attr.converters.optional(int), default=None)


@attr.s
class Config(object):
    """class that represents configuration loaded from config.ini file

    Attributes
    ----------
    model : ModelConfig
        instance of ModelConfig class, represents configuration of model
    train : TrainConfig
        instance of TrainConfig class, represents configuration for training model
    data : DataConfig
        instance of DataConfig class, represent configuration for data associated
        with model (training, testing, outputs)
    test : TestConfig
        instance of TestConfig class, represent configuration for testing model
    """
    data = attr.ib(type=DataConfig)
    model = attr.ib(type=ModelConfig, default=ModelConfig())
    train = attr.ib(type=TrainConfig, default=TrainConfig())
    test = attr.ib(type=TestConfig, default=TestConfig())
    misc = attr.ib(type=MiscConfig, default=MiscConfig())


def parse_config(config_file):
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
    if os.path.isfile(config_file):
        config = ConfigParser()
        config.read(config_file)
        sections = set(config.sections())
    else:
        raise FileNotFoundError(f'did not find config file: {config_file}')

    if sections != VALID_SECTIONS:
        if sections < VALID_SECTIONS:
            missing_sections = VALID_SECTIONS - sections
            raise ValueError(f'config missing section(s): {missing_sections}')
        elif not VALID_SECTIONS >= sections:
            extra_sections = sections - VALID_SECTIONS
            raise ValueError(f'sections in config not valid: {extra_sections}')

    for section in sections:
        options = set(config.options(section))
        valid_options = set(VALID_OPTIONS[section])
        # any options not defined go to default, so don't check if options < valid_options
        # but do check if valid_options is a superset of options,
        # i.e. test whether every element in options is in valid_options
        if not valid_options >= options:
            extra_options = options - valid_options
            raise ValueError(f'options in in config section {section} not valid: {extra_options}')
        if section == 'data':
            if 'root_results_dir' not in options:
                raise ValueError("config must specify 'root_results_dir' option")
            if 'data_dir' not in options:
                raise ValueError("config must specify 'data_dir' option")

    model_config = ModelConfig(**config['model'])
    train_config = TrainConfig(**config['train'])
    data_config = DataConfig(**config['data'])
    test_config = TestConfig(**config['test'])
    misc_config = MiscConfig(**config['misc'])
    config = Config(model=model_config, train=train_config, data=data_config, test=test_config, misc=misc_config)
    return config
