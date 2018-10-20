import os
import configparser
from collections import namedtuple

this_file_dir = os.path.dirname(__file__)

config_types = configparser.ConfigParser()
config_types.read(os.path.join(this_file_dir, 'types.ini'))


def parse_config(config_file=None):
    """read config.ini file with config parser,
    returns namedtuple ConfigTuple with
    sections and options as attributes.

    Parameters
    ----------
    config_file : str
        Path to a config.ini file. If None, the
        default configuration is returned.
        Default is None.

    Returns
    -------
    config : namedtuple
        where fields are sections of the config, and
        values for those fields are also namedtuples,
        with fields being options and values being the
        values for those options from the config.ini file.
    """
    config = configparser.ConfigParser()
    # first read defaults
    config.read(os.path.join(this_file_dir, 'default.ini'))
    # then replace values with any specified by user
    if config_file is not None:
        config.read(config_file)

    sections = [key for key in list(config.keys()) if key != 'DEFAULT']
    ConfigTuple = namedtuple('ConfigTuple', sections)
    config_dict = {}
    for section in sections:
        section_keys = list(config[section].keys())
        section_values = list(config[section].values())
        SubTup = namedtuple(section, section_keys)
        subtup_dict = {}
        for key, val in zip(section_keys, section_values):
            val_type = config_types[section][key]
            if val_type == 'int':
                typed_val = int(val)
            elif val_type == 'float':
                typed_val = float(val)
            elif val_type == 'str':
                typed_val = val
            subtup_dict[key] = typed_val
        config_dict[section] = SubTup(**subtup_dict)
    config = ConfigTuple(**config_dict)
    return config
