# Copyright (c) Facebook, Inc. and its affiliates.
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fronni",                     # This is the name of the package
    packages = ['fronni'],   # Chose the same as "name"
    version="0.0.5",                        # The initial release version
    author="Kaushik Mitra",                     # Full name of the author
    author_email='kaushik.umcp@gmail.com',
    description="Machine Learning model performance metrics & charts with confidence intervals, optimized with numba to be fast",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    url="https://github.com/facebookexperimental",
    license='MIT',
    download_url = 'https://github.com/facebookexperimental/fronni/archive/refs/tags/v_005.tar.gz',    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
)
