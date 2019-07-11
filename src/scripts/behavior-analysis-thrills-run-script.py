#!/usr/bin/env python
# coding: utf-8
import os

import matplotlib as mpl
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np

import ram

def main():
    test_results = '/home/bart/Documents/data/RAM_output/RAM_results_190422_190653/test_results_190423_091259/replicate_1/examples'
    suptitle = "learning rate = 0.001, standard devation = 0.2, test acc = 93%"
    save_as = "lr0.01_std0.2_testacc93.png"
    ram.plot.behavior(test_results, suffix='_epoch_test', suptitle=suptitle, save_as=save_as)

    test_results = '/home/bart/Documents/data/RAM_output/RAM_results_190420_231619/test_results_190422_212943/replicate_1/examples'
    suptitle = "learning rate = 0.01, standard devation = 0.2, test acc = 73.8%"
    save_as = "lr0.001_std1.0_testacc73.8.png"
    ram.plot.behavior(test_results, suffix='_epoch_test', suptitle=suptitle, save_as=save_as)

    run_results = '/home/bart/Documents/data/RAM_output/RAM_results_190422_190653/run_results_190706_184546/replicate_1/examples'
    suptitle = "learning rate = 0.001, standard devation = 0.2, test acc = 93%"
    save_as = "run_results_lr0.01_std0.2_testacc93_sample2.png"
    ram.plot.behavior(run_results, suffix='_from_run',
             plot_same_sample_diff_eps=True,
             which_sample=2,
             suptitle=suptitle, save_as=save_as)

    run_results = '/home/bart/Documents/data/RAM_output/RAM_results_190420_231619/run_results_190706_185402/replicate_1/examples'
    suptitle = "learning rate = 0.01, standard devation = 0.2, test acc = 73.8%"
    save_as = "run_results_lr0.001_std1.0_testacc73.8_sample2.png"
    ram.plot.behavior(run_results, suffix='_from_run',
		      plot_same_sample_diff_eps=True,
		      which_sample=2,
		      suptitle=suptitle, save_as=save_as)


if __name__ == '__main__':
    main()

