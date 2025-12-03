"""
Setup script for FPGA SNN Accelerator Python package

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""

from setuptools import setup, find_packages

setup(
    name="snn_fpga_accelerator",
    version="0.1.0",
    author="Jiwoon Lee",
    author_email="metr0jw@example.com",
    description="PyTorch integration for FPGA-based Spiking Neural Network Accelerator",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/metr0jw/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.8.0",
        "tqdm>=4.64.0",
        "h5py>=3.7.0",
        "jupyter>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "examples": [
            "torchvision>=0.13.0",
            "datasets>=2.0.0",
        ],
        "pynq": [
            "pynq>=2.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "snn-flash=snn_fpga_accelerator.cli:flash_bitstream",
            "snn-test=snn_fpga_accelerator.cli:run_tests",
        ],
    },
)
