import os

import ram.config

this_file_dir = os.path.dirname(__file__)
default_config_file = os.path.join(this_file_dir,
                                   '../ram/default.ini')


class TestParseConfig:
    def test_default_config(self):
        default_config = src.ram.utils.parse_config(config_file=None)
        default_config_from_file = src.ram.utils.parse_config(config_file=default_config_file)
        assert default_config == default_config_from_file
