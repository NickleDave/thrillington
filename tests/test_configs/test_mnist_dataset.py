import numpy as np

import ram.mnist.dataset

class TestDataset:
    def test_normalize(self):
        fake_mnist_image = np.random.choice(a=np.arange(255), size=(28*28))
        normed = src.ram.mnist.dataset.normalize(fake_mnist_image)
        assert np.min(normed) >= 0.
        assert np.max(normed) <= 1.
