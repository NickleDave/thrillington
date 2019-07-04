"""
Invokes __main__ when the module is run as a script.
Example: python -m ram --help
The same function is run by the script `ram-cli` which is installed on the
path by pip, so `$ ram-cli --help` would have the same effect (i.e., no need
to type the python -m)
"""
import argparse

from .cli import cli


def get_parser():
    parser = argparse.ArgumentParser(description='main script',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('command', type=str, choices=['prep', 'train', 'test'],
                        help="Command to run, either 'prep, 'train' or 'test' \n"
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
