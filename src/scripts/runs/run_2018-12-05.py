import os

import tensorflow as tf

import ram

tf.enable_eager_execution()

config = ram.parse_config('/home/art/Documents/repos/coding/L2M/RAM_config_2018-10-21.ini')

data_root = '/home/art/Documents/data/searchstim/2018-12-04/'
train_data = ram.dataset.searchstim.get_split(split_json=os.path.join(data_root, 'split_filenames.json'),
                                              setname='train')

val_data = ram.dataset.searchstim.get_split(split_json=os.path.join(data_root, 'split_filenames.json'),
                                            setname='val')

trainer = ram.Trainer(config=config,
                      train_data=train_data,
                      val_data=val_data)
trainer.train()
