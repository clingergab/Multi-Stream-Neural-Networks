from setuptools import setup, find_packages

setup(
    name="multi-stream-neural-networks",
    version="0.1.0",
    author="Your Name",
    description="Multi-Stream Neural Networks for enhanced visual processing",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "matplotlib>=3.4.0",
    ],
)