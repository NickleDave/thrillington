import os

import tensorflow as tf

import ram

tf.enable_eager_execution()

RAM_CONFIGS = '/home/art/Documents/repos/coding/L2M/RAM/src/scripts/ram_configs/'
config = ram.parse_config(os.path.join(RAM_CONFIGS, 'RAM_config_2018-12-17-MNIST.ini'))

data_root = '/home/art/Documents/data/mnist/raw'
paths_dict = ram.dataset.mnist.prep(download_dir=data_root, train_size=0.1, val_size=0.1, output_dir=data_root)
train_data, val_data = ram.dataset.mnist.get_split(paths_dict, setname=['train', 'val'])

trainer = ram.Trainer(config=config,
                      train_data=train_data,
                      val_data=val_data)
trainer.train()
