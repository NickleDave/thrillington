# Recurrent Models of Visual Attention
Replication in Tensorflow of the following paper:
Mnih, Volodymyr, Nicolas Heess, and Alex Graves.  
"Recurrent models of visual attention."  
Advances in neural information processing systems. 2014.  
<https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention>

Based in part on the following implementations:  
https://github.com/kevinzakka/recurrent-visual-attention  
https://github.com/seann999/tensorflow_mnist_ram  
(MIT license: https://github.com/seann999/tensorflow_mnist_ram/blob/master/LICENSE)

## installation
`$ pip install ram`

## usage

First load training data, such as the MNIST dataset
```
>>> data = ram.mnist.dataset.train(directory='/home/art/Documents/data/mnist/raw', num_samples=10000)
```

Then load a configuration, using the `config.ini` parser.
```
>>> config = ram.parse_config('./RAM_config_2018-10-21.ini')
```

Lastly instantiate a `Trainer` class, passing the `config` and `data` to it upon initiation, 
and then execute the `train` method.
 
```
>>> trainer = ram.Trainer(config=config, data=data)
>>> trainer.train()
  0%|          | 0/10000 [00:00<?, ?it/s]

config.train.resume is False,
will save new model and optimizer to checkpoint: /home/art/Documents/repos/coding/L2M/ram_output/checkpoints/ckpt

Epoch: 1/200 - learning rate: 0.001000

282.5s - hybrid loss: 1.690 - acc: 6.000: 100%|██████████| 10000/10000 [04:42<00:00, 35.65it/s]
  0%|          | 0/10000 [00:00<?, ?it/s]

mean accuracy: 9.97
mean losses: LossTuple(loss_reinforce=-1.1296023, loss_baseline=0.09972435, loss_action=2.3005059, loss_hybrid=1.2706277)

Epoch: 2/200 - learning rate: 0.001000

282.8s - hybrid loss: 1.223 - acc: 10.000: 100%|██████████| 10000/10000 [04:42<00:00, 35.50it/s]
  0%|          | 0/10000 [00:00<?, ?it/s]
...
```

