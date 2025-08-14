# PanStitch
A Python package for downloading and stitching images from Pan-STARRS

<!-- ![Build Status](https://img.shields.io/github/workflow/status/SilverRon/PanStitch/CI) -->
<!-- ![License](https://img.shields.io/github/license/SilverRon/PanStitch) -->

## Installation

Clone the repository and install it locally:

```bash
git clone https://github.com/SilverRon/PanStitch.git
cd PanStitch
pip install -e .
```

## Features

0. Set pointings for query
1. Download images from the Pan-STARRS survey using coordinates.
2. Stitch multiple images together using `SWarp`.

## Requirements

- Python 3.11 or higher
- Required dependencies are automatically installed from `setup.py`.

## Usage

You can use `PanStitch` to download and stitch Pan-STARRS images. Please see an example (`./example.ipynb`).

## Author

- **Gregory S.H. Paek ([GitHub](https://github.com/SilverRon))**
- Postdoctoral Researcher
- Affiliation
	- **Seoul National University (SNU) (~2024.12)**
	- Institute of Astronomy (IfA), University of Hawai'i (2025.01~)