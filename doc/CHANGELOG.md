# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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


## [0.1.0a1] - 2018-06-20
### Added
- original version, probably closest to the [one from Kevin Zakka](https://github.com/kevinzakka/recurrent-visual-attention)
- but in Tensorflow (Eager) instead of PyTorch
