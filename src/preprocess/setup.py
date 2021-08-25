"""Packager for cloud environment."""
from setuptools import setup, find_packages

setup(
    name='preprocess',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
    ],
)
