#Reads requirements from requirements.txt
#Includes metadata about the package
#Sets up entry points for command-line usage
#Configures package classifiers for PyPI
#Specifies Python version requirements

#!/usr/bin/env python
from setuptools import setup, find_packages
import os

# Read the contents of requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read the contents of README.md if it exists
readme = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        readme = f.read()

setup(
    name="aiagmar",
    version="0.1.0",
    description="AI Agent Multi-Agent Research System",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Kishoraditya",
    author_email="kishoraditya@example.com",  # Replace with actual email
    url="https://github.com/Kishoraditya/aiagmar",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "aiagmar=apps.main:main",
        ],
    },
)
