# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Trainer has `patience` attribute, number of epochs it will for accuracy to improve before 
  stopping training early.
- add `prep` command to command-line interface, to separate preparing datasets from 
  training and testing models.
- add ability to specify fixed `l0`, initial location on time step zero, that can be used when 
  measuring accuracy on validation and test sets
- add Monte Carlo sampling of policy when measuring accuracy on validation and test sets
  + number of episodes to run for each is specified with `num_mc_episodes` option in 
    [misc] section of config.ini file
- add `dataset.searchstims` module for working with images from `searchstims` package

### Changed
- use `tensorflow_probability` for distributions

### Fixed
- specifically use Normal distribution from `tensorflow_probability` in `LocationNetwork` module so that
  backpropagation works properly through this node
  + not clear if it fails when just using `tf.random.normal` because it's just not that easy yet to introspect 
    gradients in Tensorflow ()although this could be a problem with the programmer)

## [0.0.2a1] 2019-02-04
### Added
- ability to use only a subset of MNIST training data and get a validation set from it
- ability to shuffle dataset on each epoch
- ability to train replicates (experiment repeated n times with same training data,
 only random initialization / shuffling different)
- ability to specify other modules to use to load other datasets
  + an example is provided to load the [kanji MNIST dataset](https://github.com/rois-codh/kmnist)
- logging of runs/experiments, with option to dump to a text file
- tests for MNIST module in datasets
- a CHANGELOG (this one)

### Changed
- change argparser to use positional arguments `command` and `config`
  + before all arguments were "optional" (although the program would crash without them)
- many changes to training, in attempt to reproduce original paper + reconcile different versions 
  + currently: use pdf of Gaussian for policy gradient of location network, and 
  normalize both baseline, target of baseline, and advantage to decrease variance and 
  to keep gradient from exploding

### Fixed
- fix action network and glimpse network, did not have correct number of layers

## [0.0.1] - 2018-12-01
### Added
- original version, probably closest to the [one from Kevin Zakka](https://github.com/kevinzakka/recurrent-visual-attention)
- but in Tensorflow (Eager) instead of PyTorch
