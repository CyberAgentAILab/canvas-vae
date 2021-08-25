"""Packager for cloud environment."""
from setuptools import setup, find_packages

setup(
    name='pixelvae',
    version='1.0.0',
    packages=find_packages(),
    package_data={'pixelvae.data': ['*.json']},
)
