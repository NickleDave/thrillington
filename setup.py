#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='RAM',
    version='0.1a',
    description="""Tensorflow implementation of Recurrent Models of Visual Attention: 
    Mnih, Volodymyr, Nicolas Heess, and Alex Graves.
    "Recurrent models of visual attention."
    Advances in neural information processing systems. 2014.https://arxiv.org/abs/1406.6247""",
    author='David Nicholson',
    author_email='https://github.com/NickleDave/RAM/issues',
    url='https://github.com/NickleDave/RAM',
    packages=find_packages(),
    package_data={
        'ram': ['*.ini',],
    },
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
    ]
    )
