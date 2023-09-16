from setuptools import setup

with open("README.md") as f:
    readme = f.read()

kwargs = {
    "name": "welford-torch",
    "version": "0.2.1",
    "description": "Online Pytorch implementation to get Standard Deviation, Covariance, Correlation and Whitening.",
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
