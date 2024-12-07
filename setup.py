from setuptools import find_packages
from distutils.core import setup

setup(
    name='dribblebot',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
                      'params-proto==2.10.9',
                      'gym==0.18.0',
                      'tqdm',
                      'matplotlib',
                      'numpy==1.23.5',
                      'wandb==0.15.0',
                      'wandb_osh',
                      #'moviepy',
                      'imageio'
                      ]
)
