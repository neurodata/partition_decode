from setuptools import setup, find_packages
import os

# Find mgc version.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "functions", "__init__.py")):
    if line.startswith("__version__ = "):
        VERSION = line.strip().split()[2][1:-1]

with open("README.md", mode="r", encoding = "utf8") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", mode="r", encoding = "utf8") as f:
    REQUIREMENTS = f.read()

setup(
    name="double_descent",
    version=VERSION,
    author="Kaleab A Kinfu",
    author_email="kinfu@jhu.edu",
    maintainer="Kaleab A Kinfu",
    maintainer_email="kinfu@jhu.edu",
    description="A package for doing the double descent experiment in RF and DN.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neurodata/double_descent.git",
    license="MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ],
    install_requires=REQUIREMENTS,
    packages=find_packages(exclude=["tests", "tests.*", "tests/*"]),
    include_package_data=True
)