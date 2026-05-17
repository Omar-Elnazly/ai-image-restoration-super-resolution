"""
Makes the project importable as a Python package.
Run: pip install -e . (from project root)
"""

from setuptools import setup, find_packages

setup(
    name="sr_project",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
)