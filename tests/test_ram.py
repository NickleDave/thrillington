import tensorflow as tf

import ram

tfe = tf.contrib.eager


class TestRAM(tf.test.TestCase):
    def test_init(self):
        ram_model = ram.RAM()
        assert hasattr(ram_model, 'g_w')
        assert hasattr(ram_model, 'k')
        assert hasattr(ram_model, 's')
        assert hasattr(ram_model, 'hg_size')
        assert hasattr(ram_model, 'hl_size')
        assert hasattr(ram_model, 'g_size')
        assert hasattr(ram_model, 'hidden_size')
        assert hasattr(ram_model, 'glimpses')
        assert hasattr(ram_model, 'batch_size')
        assert hasattr(ram_model, 'num_classes')
        assert hasattr(ram_model, 'loc_std')
        assert hasattr(ram_model, 'glimpse_network')
        assert hasattr(ram_model, 'core_network')
        assert hasattr(ram_model, 'location_network')
        assert hasattr(ram_model, 'action_network')
        assert hasattr(ram_model, 'baseline')
        assert hasattr(ram_model, 'Out')
        assert hasattr(ram_model, 'initial_l_t_distrib')

    @tfe.run_test_in_graph_and_eager_modes()
    def test_reset(self):
        ram_model = ram.RAM()
        out_t_minus_1 = ram_model.reset()
        assert out_t_minus_1.h_t.shape == (ram_model.batch_size, ram_model.hidden_size)
        assert out_t_minus_1.l_t.shape == (ram_model.batch_size, 2)
        assert not any([out_t_minus_1.mu, out_t_minus_1.a_t, out_t_minus_1.b_t])

    @tfe.run_test_in_graph_and_eager_modes()
    def test_step(self):
        batch_size = 10
        img_height = 28
        img_width = 28
        channels = 1
        fake_images = tf.random_uniform(shape=(batch_size, img_height, img_width, channels),
                                        maxval=256,
                                        dtype=tf.int32)
        ram_model = ram.RAM(batch_size=batch_size)
        out_t_minus_1 = ram_model.reset()
        out = ram_model.step(fake_images, out_t_minus_1.h_t, out_t_minus_1.l_t)


if __name__ == "__main__":
    tf.test.main()