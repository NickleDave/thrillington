"""adapted from https://github.com/rois-codh/kmnist under CC BY-SA 4.0 license:
"KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341
"""
import ram


download_dict = {
    'kmnist': {
        'train_images_file': 'train-images-idx3-ubyte',
        'train_labels_file': 'train-labels-idx1-ubyte',
        'test_images_file': 't10k-images-idx3-ubyte',
        'test_labels_file': 't10k-labels-idx1-ubyte',
        'url_file_extension': '.gz'
    },
    'k49': {
        'train_images_file': 'k49-train-imgs',
        'train_labels_file': 'k49-train-labels',
        'test_images_file': 'k49-test-imgs',
        'test_labels_file': 'k49-test-labels',
        'url_file_extension': '.npz'
    },
    'kkanji': [
        'kkanji.tar'],
}


def prep(download_dir, dataset='kmnist', train_size=None, val_size=None, random_seed=None, output_dir=None):

    if dataset not in download_dict.keys():
        raise KeyError(f'dataset name {dataset} not recognized')
    url_root = 'http://codh.rois.ac.jp/kmnist/dataset/' + dataset + '/'
    paths_dict = ram.dataset.mnist.prep(download_dir, train_size=train_size, val_size=val_size, random_seed=random_seed,
                                        train_images_file=download_dict[dataset]['train_images_file'],
                                        train_labels_file=download_dict[dataset]['train_labels_file'],
                                        test_images_file=download_dict[dataset]['test_images_file'],
                                        test_labels_file=download_dict[dataset]['test_labels_file'],
                                        url_root=url_root,
                                        url_file_extension=download_dict[dataset]['url_file_extension'],
                                        output_dir=output_dir)
    return paths_dict


def get_split(paths_dict, setname='train'):
    return ram.dataset.mnist.get_split(paths_dict, setname)
