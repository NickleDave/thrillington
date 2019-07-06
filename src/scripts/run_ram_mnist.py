import importlib.util
import logging

import tensorflow as tf

import ram


def main():
    for configfile in CONFIG_FILES:
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

        results_dir = config.data.results_dir_made_by_main
        timenow = datetime.now().strftime('%y%m%d_%H%M%S')

        logger.info(f'Used config file: {configfile}')
        logger.info(f'Used random seed: {config.misc.random_seed}')

        logger.info(f'\nLoading data from path in {config.data.paths_dict_fname}, ')
        with open(config.data.paths_dict_fname) as fp:
            paths_dict = json.load(fp)

        logger.info(f'\nUsing {config.data.module} module to load dataset')
        data = dataset_module.get_split(paths_dict, setname=['test'])
        runnner = ram.Runner.from_config(config=config,
                                         data=data,
                                         logger=logger)
        tester.test(results_dir=results_dir, save_examples=config.test.save_examples,
                    num_examples_to_save=config.test.num_examples_to_save)

        runner = ram.Runner()


if __name__ == '__main__':
    main()
