from pathlib import Path
import tempfile
import shutil

import tensorflow as tf

import ram

TESTS_DIR = Path(__file__).resolve().parent
TEST_CONFIGS_DIR = TESTS_DIR.joinpath('test_configs')

tfe = tf.contrib.eager


class TestRAM(tf.test.TestCase):
    def setUp(self):
        self.tmp_config_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_config_dir)

    @tfe.run_test_in_graph_and_eager_modes
    def test_init(self):
        config_file = TEST_CONFIGS_DIR.joinpath('good_config.ini')
        config = ram.config.parse_config(config_file=config_file)
        tester = ram.Tester(config=config)

    @tfe.run_test_in_graph_and_eager_modes
    def test_test(self):
        config_file = bla
        config = ram.config.parse_config(config_file=config_file)
        tester = ram.Tester(config=config)
        test_data = ram.dataset.mnist.get_split(setname='test')
        tester.test(test_data=test_data)


if __name__ == '__main__':
    tf.test.main()
