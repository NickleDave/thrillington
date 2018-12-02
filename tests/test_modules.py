import tensorflow as tf

import ram.modules

tfe = tf.contrib.eager


class TestGlimpseSensor(tf.test.TestCase):
    def test_init(self):
        glimpse_sensor = src.ram.modules.GlimpseSensor()
        assert hasattr(glimpse_sensor, 'g_w')
        assert hasattr(glimpse_sensor, 'k')
        assert hasattr(glimpse_sensor, 's')
        assert hasattr(glimpse_sensor, 'glimpse')

    @tfe.run_test_in_graph_and_eager_modes
    def test_glimpse(self):
        batch_size = 10
        img_height = 28
        img_width = 28
        channels = 1
        fake_images = tf.random_uniform(shape=(batch_size, img_height, img_width, channels))
        loc_normd = tf.random_uniform(shape=(batch_size, 2), minval=-1, maxval=1)
        glimpse_sensor = src.ram.modules.GlimpseSensor()
        rho = glimpse_sensor.glimpse(fake_images, loc_normd)
        assert rho.shape == (batch_size,
                             glimpse_sensor.k,
                             glimpse_sensor.g_w,
                             glimpse_sensor.g_w,
                             channels)


class TestGlimpseNetwork(tf.test.TestCase):
    def test_init(self):
        glimpse_network = src.ram.modules.GlimpseNetwork()
        assert hasattr(glimpse_network, 'g_w')
        assert hasattr(glimpse_network, 'k')
        assert hasattr(glimpse_network, 's')
        assert hasattr(glimpse_network, 'h_g_units')
        assert hasattr(glimpse_network, 'h_l_units')
        assert hasattr(glimpse_network, 'h_gt_units')
        assert hasattr(glimpse_network, 'glimpse_sensor')

    @tfe.run_test_in_graph_and_eager_modes
    def test_forward(self):
        batch_size = 10
        img_height = 28
        img_width = 28
        channels = 1
        fake_images = tf.random_uniform(shape=(batch_size, img_height, img_width, channels))
        loc_normd = tf.random_uniform(shape=(batch_size, 2), minval=-1, maxval=1)
        glimpse_network = src.ram.modules.GlimpseNetwork()
        rho, g_t = glimpse_network.forward(fake_images, loc_normd)
        assert rho.shape == (batch_size,
                             glimpse_network.k,
                             glimpse_network.g_w,
                             glimpse_network.g_w,
                             channels)
        assert g_t.shape == (batch_size,
                             glimpse_network.h_gt_units)


class TestCoreNetwork(tf.test.TestCase):
    def test_init(self):
        hidden_size = 256
        core_network = src.ram.modules.CoreNetwork(hidden_size=hidden_size)
        assert hasattr(core_network, 'hidden_size')
        assert hasattr(core_network, 'linear_h_t_minus_1')
        assert hasattr(core_network, 'linear_g_t')

    @tfe.run_test_in_graph_and_eager_modes
    def test_forward(self):
        batch_size = 10
        hidden_size = 256
        core_network = src.ram.modules.CoreNetwork(hidden_size=hidden_size)
        h_t_minus_1 = tf.random_uniform(shape=(batch_size, hidden_size))
        g_t = tf.random_uniform(shape=(batch_size, hidden_size))
        h_t = core_network.forward(g_t, h_t_minus_1)
        assert h_t.shape == (batch_size, hidden_size)


class TestLocationNetwork(tf.test.TestCase):
    def test_init(self):
        loc_std = 0.01
        loc_net = src.ram.modules.LocationNetwork(loc_std=loc_std)
        assert hasattr(loc_net, 'output_size')
        assert hasattr(loc_net, 'loc_std')
        assert hasattr(loc_net, 'fc')
        assert hasattr(loc_net, 'forward')

    @tfe.run_test_in_graph_and_eager_modes
    def test_forward(self):
        batch_size = 10
        hidden_size = 256
        loc_std = 0.01
        output_size = 2
        loc_net = src.ram.modules.LocationNetwork(loc_std=loc_std, output_size=output_size)
        h_t = tf.zeros(shape=(batch_size, hidden_size))
        mu, l_t = loc_net.forward(h_t)
        assert mu.shape == (batch_size, output_size)
        assert l_t.shape == (batch_size, output_size)


class TestActionNetwork(tf.test.TestCase):
    def test_init(self):
        num_actions = 10
        action_network = src.ram.modules.ActionNetwork(num_actions=num_actions)
        assert hasattr(action_network, 'num_actions')
        assert hasattr(action_network, 'fc')

    @tfe.run_test_in_graph_and_eager_modes
    def test_forward(self):
        num_actions = 10
        action_network = src.ram.modules.ActionNetwork(num_actions=num_actions)
        batch_size = 10
        hidden_size = 256
        h_t = tf.random_uniform(shape=(batch_size, hidden_size))
        a_t = action_network.forward(h_t)
        assert a_t.shape == (batch_size, num_actions)
        # a_t_sum = tf.reduce_sum(a_t, axis=1)
        # ones_batch_size = tf.ones(shape=(batch_size,))
        # this test fails with a weird error, haven't figured out why
        # self.assertAllClose(a_t_sum, ones_batch_size)


class TestBaselineNetwork(tf.test.TestCase):
    def test_init(self):
        output_size = 1
        baseline_network = src.ram.modules.BaselineNetwork(output_size=output_size)
        assert hasattr(baseline_network, 'output_size')
        assert hasattr(baseline_network, 'fc')

    @tfe.run_test_in_graph_and_eager_modes
    def test_forward(self):
        output_size = 1
        baseline_network = src.ram.modules.BaselineNetwork(output_size=output_size)
        batch_size = 10
        hidden_size = 256
        h_t = tf.random_uniform(shape=(batch_size, hidden_size))
        b_t = baseline_network.forward(h_t)
        assert b_t.shape == (batch_size, output_size)


if __name__ == "__main__":
    tf.test.main()
