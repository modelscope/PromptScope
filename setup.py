# -*- coding: utf-8 -*-
""" Setup for installation."""
from __future__ import absolute_import, division, print_function
import setuptools

# released requires
requires = [
    "dashscope",
    "scipy",
    "numpy",
    "bm25s"
]

setuptools.setup(
    name="meta_icl",
    version="v0.0.1",
    author="",
    author_email="",
    description="A simple framework for prompt optimization",
    packages=setuptools.find_packages(),
    install_requires=requires,
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
