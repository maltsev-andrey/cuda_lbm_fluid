"""
Setup script for cuda_lbm_fluid package.
"""

from setuptools import setup, find_packages

setup(
    name="cuda_lbm_fluid",
    version="0.1.0",
    description="GPU-accelerated Lattice Boltzmann fluid dynamics simulation",
    author="Andrey",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "numba>=0.56",
        "matplotlib>=3.5",
        "scipy>=1.7",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "flake8"],
    },
)
