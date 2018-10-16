import tensorflow as tf

import ram.modules

tfe = tf.contrib.eager


class TestLocationNetwork(tf.test.TestCase):
    def test_init(self):
        loc_std = 0.01
        loc_net = ram.modules.LocationNetwork(loc_std=loc_std)
        assert hasattr(loc_net, 'output_size')
        assert hasattr(loc_net, 'loc_std')
        assert hasattr(loc_net, 'fc')
        assert hasattr(loc_net, 'forward')

    @tfe.run_test_in_graph_and_eager_modes()
    def test_output(self):
        batch_size = 10
        hidden_size = 256
        loc_std = 0.01
        output_size = 2
        loc_net = ram.modules.LocationNetwork(loc_std=loc_std, output_size=output_size)
        h_t = tf.zeros(shape=(batch_size, hidden_size))
        mu, l_t = loc_net.forward(h_t)
        assert mu.shape == (batch_size, output_size)
        assert l_t.shape == (batch_size, output_size)


if __name__ == "__main__":
    tf.test.main()
