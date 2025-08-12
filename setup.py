import setuptools
from setuptools import find_packages
from setuptools.config import read_configuration
from pathlib import Path

import emergent_morphogenesis as em

# Read the configuration file
config = read_configuration(Path(__file__).parent / "setup.cfg")

# Get metadata from configuration
metadata = config["metadata"]
name = metadata["name"]
version = metadata["version"]
description = metadata["description"]
author = metadata["author"]
author_email = metadata["author_email"]
url = metadata["url"]
download_url = metadata["download_url"]
keywords = metadata["keywords"]
classifiers = metadata["classifiers"]

# Get options from configuration
options = config["options"]
install_requires = options["install_requires"]
packages = find_packages(exclude=["tests", "docs"])

# Include any data files or assets required by the package
data_files = []

# Add any entry points for console scripts or other extensions
entry_points = {
    "console_scripts": [
        "emergent_morphogenesis = emergent_morphogenesis.__main__:main",
    ],
}

# Define the setup function
setuptools.setup(
    name=name,
    version=version,
    description=description,
    author=author,
    author_email=author_email,
    url=url,
    download_url=download_url,
    keywords=keywords,
    classifiers=classifiers,
    install_requires=install_requires,
    packages=packages,
    data_files=data_files,
    entry_points=entry_points,
)

# Example usage:
# python setup.py install
# python -m emergent_morphogenesis