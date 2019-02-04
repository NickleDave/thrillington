# Recurrent Models of Visual Attention
Replication in Tensorflow of the following paper:  
Mnih, Volodymyr, Nicolas Heess, and Alex Graves.  
"Recurrent models of visual attention."  
Advances in neural information processing systems. 2014.  
<https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention>

Based in part on the following implementations:  
- <https://github.com/torch/rnn/blob/master/examples/recurrent-visual-attention.lua>
  + (license: <https://github.com/torch/rnn/blob/master/LICENSE.txt>) 
- <https://github.com/seann999/tensorflow_mnist_ram>  
  + (MIT license: <https://github.com/seann999/tensorflow_mnist_ram/blob/master/LICENSE>)
- <https://github.com/kevinzakka/recurrent-visual-attention>  

## installation
`$ pip install thrillington`  
(`thrillington` because there is already a `ram` on PyPI, 
and because <https://en.wikipedia.org/wiki/Thrillington>)

## usage
The library can be run from the command line with a config file.
```
$ ram train ./RAM_config-2018-10-21.ini

...

  0%|          | 0/10000 [00:00<?, ?it/s]

config.train.resume is False,
will save new model and optimizer to checkpoint: /home/you/data/ram_output/results_20181021/checkpoints/ckpt

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

For a detailed explanation of the config file format, please see [here](./doc/config.md)

## CHANGELOG
To see past changes and work in progress, please check out the [CHANGELOG](./doc/CHANGELOG.md).
