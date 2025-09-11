# ezphot

[![PyPI](https://img.shields.io/pypi/v/ezphot.svg)](https://pypi.org/project/ezphot/)
[![Documentation Status](https://readthedocs.org/projects/ezphot/badge/?version=latest)](https://ezphot.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/hhchoi1022/ezphot)](https://github.com/hhchoi1022/ezphot/blob/master/LICENSE)

---

**ezphot** is a high-level Python toolkit for **astronomical image processing and photometry**.  
It provides users with an integrated framework to handle the entire imaging workflow from raw calibration to photometric analysis and transient detection using a clean, modular, and scalable architecture.

## Features

- 🔭 **Complete imaging pipeline**: From raw data to photometric analysis
- ⚡ **Multiprocessing support**: Efficient parallel processing for large datasets
- 📊 **Advanced photometry**: Aperture and PSF photometry with automatic calibration
- 🎯 **Precise astrometry**: Built-in plate solving and coordinate transformations
- 📈 **Quality control**: Image quality assessment and selection tools
- 🔧 **Modular design**: Clean, extensible architecture for custom workflows

## Installation

The latest stable release of **ezphot** is available on [PyPI](https://pypi.org/project/ezphot/).  
Documentation: [ReadTheDocs](https://ezphot.readthedocs.io/en/latest/)  
Source: [GitHub](https://github.com/hhchoi1022/ezphot)

Install with pip::

    pip install ezphot

Upgrade to the newest version::

    pip install --upgrade ezphot

For development builds from TestPyPI::

    pip install -i https://test.pypi.org/simple/ ezphot

From source::

    git clone https://github.com/hhchoi1022/ezphot.git
    cd ezphot
    pip install .
