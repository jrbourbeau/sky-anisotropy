#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree
from setuptools import setup, find_packages, Command

NAME = 'sky-anisotropy'
DESCRIPTION = 'Python package for calculating anisotropies vs. sky position'
MAINTAINER = 'James Bourbeau'
URL = 'https://github.com/jrbourbeau/sky-anisotropy'
LICENSE = 'MIT'


here = os.path.abspath(os.path.dirname(__file__))

def read(path, encoding='utf-8'):
    with io.open(path, encoding=encoding) as f:
        content = f.read()
    return content

def get_install_requirements(path):
    content = read(path)
    requirements = [req for req in content.split("\n")
                    if req != '' and not req.startswith('#')]
    return requirements

LONG_DESCRIPTION = read(os.path.join(here,'README.md'))

# Want to read in package version number from __version__.py
about = {}
with io.open(os.path.join(here, 'sky_anisotropy', '__version__.py'), encoding='utf-8') as f:
    exec(f.read(), about)
    VERSION = about['__version__']

INSTALL_REQUIRES = get_install_requirements(os.path.join(here, 'requirements.txt'))

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url=URL,
    author=MAINTAINER,
    license=LICENSE,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    setup_requires=['setuptools>=38.6.0'],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
