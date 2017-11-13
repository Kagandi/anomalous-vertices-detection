#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is based on: https://github.com/kennethreitz/setup.py
# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine
import os
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command

try:
    from pypandoc import convert

    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

# Package meta-data.
NAME = 'anomalous_vertices_detection'
DESCRIPTION = 'The Anomalous-Vertices-Detection project is a Python package for performing graph analysis. The package supports extracting graphs\' topological features, performing link prediction, and identifying anomalous vertices.'
URL = 'https://github.com/Kagandi/anomalous-vertices-detection'
EMAIL = 'kagandi@post.bgu.ac.il'
AUTHOR = 'Dima Kagan'

# What packages are required for this module to be executed?
REQUIRED = [
    'networkx<=2.0', 'scikit-learn<=0.19.0', 'pandas<=0.20.3', 'tqdm<=4.15.0', 'python-dotenv<=0.6.5', 'numpy<=1.13.1', 'requests<=2.9.1', 'scipy<=0.19.1', 'python-louvain<=0.9'
]

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.rst' is present in your MANIFEST.in file!
# with io.open(read_md(os.path.join(here, 'README.md')), encoding='utf-8') as f:
long_description = '\n' + read_md(os.path.join(here, 'README.md'))

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec (f.read(), about)


class PublishCommand(Command):
    """Support setup.py publish."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """
        Prints things in bold.
        """
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass
        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    setup_requires=['numpy'],
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: Implementation :: CPython'
    ],
    # $ setup.py publish support.
    cmdclass={
        'publish': PublishCommand,
    },
)
