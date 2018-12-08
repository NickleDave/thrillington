import os

import tensorflow as tf

import ram

tf.enable_eager_execution()

RAM_CONFIGS = '/home/art/Documents/repos/coding/L2M/RAM_L2M/src/scripts/ram_configs/'
config = ram.parse_config(os.path.join(RAM_CONFIGS, 'RAM_config_2018-12-07-MNIST.ini'))

data_root = '/home/art/Documents/data/mnist/raw'
train_data = ram.dataset.mnist.train(data_root, num_samples=20000)

trainer = ram.Trainer(config=config,
                      train_data=train_data)
trainer.train()
