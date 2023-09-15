from setuptools import setup

with open("README.md") as f:
    readme = f.read()

kwargs = {
    "name": "welford-torch",
    "version": "0.2.0",
    "description": "Python (Pytorch) implementation of Online Algorithms for Standard Deviation, Covariance and Correlation Coefficients.",
    "author": "Nicky Pochinkov",
    "author_email": "work@nicky.pro",
    "url": "https://github.com/pesvut/welford-torch",
    "license": "MIT",
    "keywords": ["statistics", "online", "welford", "torch", "covariance", "correlation"],
    "install_requires": ["torch"],
    "packages": ["welford_torch"],
    "long_description": readme,
    "long_description_content_type": "text/markdown",
}

setup(**kwargs)
