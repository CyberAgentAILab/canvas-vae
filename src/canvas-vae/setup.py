"""Packager for cloud environment."""
from setuptools import setup, find_packages

setup(
    name='canvasvae',
    version='1.0.0',
    packages=find_packages(),
    package_data={'canvasvae.data': ['*.json', '*.yml']},
    install_requires=['faiss-cpu'],
)
