from setuptools import setup

with open("README.rst") as f:
    readme = f.read()

kwargs = {
    "name": "welford_torch",
    "version": "0.1",
    "description": "Python (Pytorch) implementation of Welford's algorithm.",
    "author": "Nicky Pochinkov",
    "author_email": "work@nicky.pro",
    "url": "https://github.com/pesvut/welford-torch",
    "license": "MIT",
    "keywords": ["statistics", "online", "welford", "torch"],
    "install_requires": ["torch"],
    "packages": ["welford_torch"],
    "long_description": readme,
}

setup(**kwargs)
