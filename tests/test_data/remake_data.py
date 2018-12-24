"""makes reduced size MNIST dataset we can use just to test everything runs without crashing"""
from pathlib import Path

import ram

HERE = Path(__file__).parent


def main():

    paths_dict = ram.dataset.mnist.prep(download_dir=download_dir,
                                        train_size=0.2,
                                        val_size=0.1)


if __name__ == '__main__':
    main()
