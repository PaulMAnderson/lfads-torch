from setuptools import find_packages, setup

setup(
    name="lfads_torch",
    author="Andrew Sedler",
    author_email="arsedler9@gmail.com",
    description="A PyTorch implementation of "
    "Latent Factor Analysis via Dynamical Systems (LFADS)",
    url="https://github.com/arsedler9/lfads-torch",
    install_requires=[],  # Empty list - no dependencies will be automatically installed
    packages=find_packages(),
)
