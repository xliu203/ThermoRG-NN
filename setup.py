# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages

setup(
    name="thermorg",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
)
