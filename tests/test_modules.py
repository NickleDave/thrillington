import tensorflow as tf

import ram.modules

tfe = tf.contrib.eager


class TestGlimpseSensor(tf.test.TestCase):
    def test_init(self):
        glimpse_sensor = ram.modules.GlimpseSensor()
        assert hasattr(glimpse_sensor, 'g_w')
        assert hasattr(glimpse_sensor, 'k')
        assert hasattr(glimpse_sensor, 's')
        assert hasattr(glimpse_sensor, 'glimpse')

    def test_glimpse(self):
        batch_size = 10
        img_height = 28
        img_width = 28
        channels = 1
        fake_images = tf.random_uniform(shape=(batch_size, img_height, img_width, channels))
        loc_normd = tf.random_uniform(shape=(batch_size, 2), minval=-1, maxval=1)
        glimpse_sensor = ram.modules.GlimpseSensor()
        rho = glimpse_sensor.glimpse(fake_images, loc_normd)
        assert rho.shape == (batch_size,
                             glimpse_sensor.k,
                             glimpse_sensor.g_w,
                             glimpse_sensor.g_w,
                             channels)


class TestGlimpseNetwork(tf.test.TestCase):
    def test_init(self):
        glimpse_network = ram.modules.GlimpseNetwork()
        assert hasattr(glimpse_network, 'g_w')
        assert hasattr(glimpse_network, 'k')
        assert hasattr(glimpse_network, 's')
        assert hasattr(glimpse_network, 'h_g_units')
        assert hasattr(glimpse_network, 'h_l_units')
        assert hasattr(glimpse_network, 'h_gt_units')
        assert hasattr(glimpse_network, 'glimpse_sensor')

    def test_forward(self):
        batch_size = 10
        img_height = 28
        img_width = 28
        channels = 1
        fake_images = tf.random_uniform(shape=(batch_size, img_height, img_width, channels))
        loc_normd = tf.random_uniform(shape=(batch_size, 2), minval=-1, maxval=1)
        glimpse_network = ram.modules.GlimpseNetwork()
        rho, g_t = glimpse_network.forward(fake_images, loc_normd)
        assert rho.shape == (batch_size,
                             glimpse_network.k,
                             glimpse_network.g_w,
                             glimpse_network.g_w,
                             channels)
        assert g_t.shape == (batch_size,
                             glimpse_network.h_gt_units)


class TestLocationNetwork(tf.test.TestCase):
    def test_init(self):
        loc_std = 0.01
        loc_net = ram.modules.LocationNetwork(loc_std=loc_std)
        assert hasattr(loc_net, 'output_size')
        assert hasattr(loc_net, 'loc_std')
        assert hasattr(loc_net, 'fc')
        assert hasattr(loc_net, 'forward')

    @tfe.run_test_in_graph_and_eager_modes()
    def test_forward(self):
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
