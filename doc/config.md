# Config

The config.ini files are parsed by the Python `ConfigParser` and so follow the .ini file format.

## Model section
```ini
[model]
g_w = 8
k = 3
s = 2
hg_size = 128
hl_size = 128
g_size = 256
hidden_size = 256
glimpses = 6
num_classes = 10
loc_std = 0.1
```
- `g_w` : int
  + size of glimpse window in pixels, default is 8 x 8
- `k` : int
  + number of regions
- `s` : int
  + scaling factor
- `hg_size` : int
  + number of units in hidden layer that takes glimpse as input
- `hl_size` : int
  + number of units in hidden layer that takes glimpse location as input
- `g_size` : int
  + number of units in hidden layer that combines glimpse + location hidden layers
- `hidden_size` : int
  + number of units in core network
- `glimpses` : int
  + number of glimpses to take of each image
- `num_classes` : int
  + number of classes in data
- `loc_std` : float
  + standard deviation of distribution from which each glimpse location is drawn

## Data section
```ini
[data]
root_results_dir = /home/you/Documents/data/RAM_output
data_dir = /home/you/Documents/data/mnist/raw
module = ram.dataset.mnist
train_size = 0.9
val_size = 0.1
```
- `root_results_dir` : str
  + name of directory in which a new sub-directory will be created to hold results from run
- `data_dir` : str
  + name of directory where data is saved
- `module` : str
  + name of module that prepares and returns data sets, in absolute terms, i.e. 
   `package.module`, `package.subpackage.module`, etc. Python will use `importlib` to 
   import this module. This is a quick and dirty way to make it possible to use your 
   own data with the library. The module must implement two functions, `prep` and `get_split`.
   The function should accept arguments and return variables like so:
   ```Python
   paths_dict = dataset_module.prep(download_dir=config.data.data_dir,
                                    train_size=config.data.train_size,
                                    val_size=config.data.val_size,
                                    output_dir=config.data.data_dir)
   train_data, val_data = dataset_module.get_split(paths_dict, setname=['train', 'val'])
   ```
   For more detail and examples, see the `ram.__main__` function and the `ram.dataset.mnist` module.
- `val_size` : float
  + Amount of training data to use for a validation set
- `train_size` : float
  + Amount of entire training data set to use as a subset for training


## Train section
```ini
[train]
batch_size = 200
learning_rate = 1e-3
epochs = 200
optimizer = momentum
momentum = 0.9
checkpoint_prefix = ckpt
restore = False
save_log = Truesave_examples_every = 25
num_examples_to_save = 9
save_loss = True
save_train_inds = True
```
- `batch_size` : int
  + number of samples in each batch fed to the network
- `learning_rate` : float
  + learning rate used by optimizer during training
- `epochs` : int
  + number of training epochs, i.e. number of times network sees entire training set
- `optimizer` : str
  + one of {'adam', 'momentum', 'sgd'}; type of optimizer used
- `momentum`: float
  + if optimizer is 'momentum', the value for 'momentum'
- `checkpoint_prefix` : str
  + prefix for checkpoint files
- `restore` : bool
  + if True, restore from previous checkpoint. Not used by command-line interface but
  useful if you want to train your own models with the Trainer class.
- `save_examples_every` : int
  + if defined, save examples at an interval of this many epochs. Examples are
  glimpses seen by network, locations of gazes, fixations (locations converted from normalized to
  pixel co-ordinates), and images (from which glimpses were taken)
- 'num_examples_to_save' : int
  + number of examples to save, if `save_examples` is `True`
- 'save_loss' : bool
  if `True`, save records of loss from every epoch.
- 'save_train_inds' : bool
  if `True`, save the indices of the training samples used for each epoch.

## Test section
```ini
[test]
save_examples = True
num_examples_to_save = 15
```
- `save_examples` : bool
  + if True, save examples. Examples are
  glimpses seen by network, locations of gazes, fixations (locations converted from normalized to
  pixel co-ordinates), and images (from which glimpses were taken)
- 'num_examples_to_save' : int
  + number of examples to save, if `save_examples` is `True`

## Misc section
```ini
[misc]
save_log = True
```
- `save_log` : bool
  + if True, save logging to a text file.
- `random_seed` : int
  + number used to seed random number generator, to make results reproducible across runs
