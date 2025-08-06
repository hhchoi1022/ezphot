#%%
from pathlib import Path
import logging
import logging.handlers
import os
from typing import Union

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.io.fits import Header

from tippy.configuration import TIPConfig
from tippy.helper import Helper
#%%
class LazyFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=True):
        # `delay=True` avoids creating the file immediately
        super().__init__(filename, mode, encoding, delay=delay)
    
class Logger:
    def __init__(self, logger_name):
        self.path = logger_name
        self._log = self.createlogger(logger_name)

    def log(self):
        return self._log

    def createlogger(self, logger_name, logger_level='INFO'):
        logger = logging.getLogger(logger_name)
        if len(logger.handlers) > 0:
            return logger  # Logger already exists

        logger.setLevel(logger_level)
        formatter = logging.Formatter(
            datefmt='%Y-%m-%d %H:%M:%S',
            fmt='[%(levelname)s] %(asctime)-15s | %(message)s'
        )

        # Stream Handler
        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(logger_level)
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)

        # Lazy File Handler
        fileHandler = LazyFileHandler(filename=logger_name, delay=True)
        fileHandler.setLevel(logger_level)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        return logger
    
#%%
class DummyImage(TIPConfig):
    """ Handles FITS image processing and tracks its status """

    def __init__(self, path: Union[Path, str]):
        super().__init__()
        self.helper = Helper()
        self.path =  Path(path)
        
        self._data = None
        self._header = Header()

    @property
    def data(self):
        """Lazy-load FITS image data"""
        if not self.is_data_loaded and self.is_exists:
            try:
                self._data = fits.getdata(self.path)
            except Exception as e:
                print(f"Failed to load data from {self.path}: {e}")
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def header(self):
        """Lazy-load FITS header"""
        if not self.is_header_loaded and self.is_exists:
            try:
                self._header = fits.getheader(self.path)
            except Exception as e:
                print(f"Failed to load header from {self.path}: {e}")
        return self._header

    @header.setter
    def header(self, value):
        self._header = value
        
    @property
    def is_data_loaded(self):
        return self._data is not None

    @property
    def is_header_loaded(self):
        return isinstance(self._header, Header) and len(self._header) > 0

    @property
    def is_exists(self):
        return self.path.exists()
    
    def rename(self, new_name: str):
        """Rename the image file (with overwrite support)."""
        old_path = self.path
        new_path = self.path.parent / new_name

        # If the target exists, remove it (overwrite)
        if new_path.exists():
            new_path.unlink()  # remove the existing file

        old_path.rename(new_path)
        self.path = new_path
        print(f"Renamed {old_path} to {new_path}")
    
    def clear(self):
        """Clear the image data and header"""
        self._data = None
        self._header = Header()
        print(f"Cleared data and header for {self.path}")

    def update_header(self, **kwargs):
        if self._header is None:
            print("WARNING: Header is not loaded. Cannot update.")
            return
        else:
            for key, value in kwargs.items():
                key_upper = key.upper()
                if key_upper in self.key_variants.keys():
                    key_variants = self.key_variants[key_upper]
                    for key_variant in key_variants:
                        if key_variant in self._header:
                            self._header[key_variant] = value
                else:
                    print(f'WARNING: Key {key} not found in key_variants.')
                    
    def add_header(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self._header[key] = value
            else:
                self._header[key] = str(value)
            
    def load(self):
        if not self.is_exists:
            raise FileNotFoundError(f'File not found: {self.path}')
        self.load_data_from_path()
        self.load_header_from_path()
        
    def load_data_from_path(self):
        if self.is_exists:
            self._data = fits.getdata(self.path)
        else:
            pass

    def load_header_from_path(self):
        if self.is_exists:
            self._header = fits.getheader(self.path)
        else:
            pass

    def show(self, cmap='gray', scale='zscale', downsample=4, figsize=(8, 6), title=None):
        """
        Visualize the FITS image using slicing-based downsampling and scaling.

        Parameters:
        - cmap: str, matplotlib colormap
        - scale: str, 'zscale' or 'minmax'
        - downsample: int, step size for downsampling via slicing (default = 1, i.e. no downsample)
        - figsize: tuple, matplotlib figure size
        - title: str, plot title
        """
        import matplotlib.pyplot as plt
        from astropy.visualization import ZScaleInterval, MinMaxInterval
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        data = self.data
        if data is None:
            print("WARNING: Image data is not loaded. Please load the image first.")
            return
        
        # Handle NaN and inf
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Store original shape before slicing
        ny, nx = data.shape

        # Downsampling using slicing
        if downsample > 1:
            data = data[::downsample, ::downsample]

        # Scaling
        if scale == 'zscale':
            interval = ZScaleInterval()
        elif scale == 'minmax':
            interval = MinMaxInterval()
        else:
            print(f"Invalid scale option: {scale}. Use 'zscale' or 'minmax'.")
            return

        vmin, vmax = interval.get_limits(data)

        # Plot image
        fig, ax = plt.subplots(figsize=figsize)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)  # colorbar axis

        img = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

        # Set original pixel ticks
        yticks = np.linspace(0, data.shape[0]-1, num=6, dtype=int)
        xticks = np.linspace(0, data.shape[1]-1, num=6, dtype=int)
        ax.set_yticks(yticks)
        ax.set_yticklabels((yticks * downsample).astype(int))
        ax.set_xticks(xticks)
        ax.set_xticklabels((xticks * downsample).astype(int))

        ax.set_title(title or self.path.name)
        fig.colorbar(img, cax=cax, label='Pixel value')  # colorbar on the matched axis

        plt.tight_layout()
        plt.show()
        
    def run_ds9(self):
        self.helper.run_ds9(self.path)    

    @property
    def naxis1(self):
        for key in self.key_variants['NAXIS1']:
            if key in self._header:
                return self._header[key]
        return None

    @property
    def naxis2(self):
        for key in self.key_variants['NAXIS2']:
            if key in self._header:
                return self._header[key]
        return None
    
    @property
    def exptime(self):
        for key in self.key_variants['EXPTIME']:
            if key in self._header:
                return self._header[key]
        return None

    @property
    def egain(self):
        for key in self.key_variants['EGAIN']:
            if key in self._header:
                return self._header[key]
        return None
            
    @property
    def wcs(self):
        """
        Returns the WCS information of the image.
        """
        try:
            from astropy.wcs import WCS
            wcs = WCS(self._header)
            return wcs
        except:
            return None
        
    @property
    def center(self):
        """
        Returns the center coordinates of the image in RA and DEC.
        """
        center_info = dict()
        center_info['x'] = self.naxis1 / 2
        center_info['y'] = self.naxis2 / 2
        center_info['ra'] = None
        center_info['dec'] = None
        if self.wcs is not None:
            center_world = self.wcs.pixel_to_world(center_pixel[0], center_pixel[1])
            center_info['ra'] = center_world.ra.deg
            center_info['dec'] = center_world.dec.deg
        return center_info
        
    @property
    def key_variants(self):
        # Define key variants, if a word is duplicated in the same variant, posit the word with the highest priority first
        key_variants_upper = {
            # Default key in rawimages 
            'NAXIS1': ['NAXIS1'],
            'NAXIS2': ['NAXIS2'],
            'EGAIN': ['EGAIN'],
            'EXPTIME': ['EXPTIME', 'EXPOSURE'],
            'BINNING': ['BINNING', 'XBINNING'],
            'RDNOISE': ['RDNOISE'],
            'IMGTYPE': ['IMGTYPE'],
            'TGTPATH': ['TGTPATH'],
            'BIASPATH': ['BIASPATH'],
            'DARKPATH': ['DARKPATH'],
            'FLATPATH': ['FLATPATH'],
            # Mask
            'MASKPATH': ['MASKPATH'],
            'MASKTYPE': ['MASKTYPE'],
            'MASKATMP': ['MASKATMP'],
            # Background
            'BKGPATH': ['BKGPATH'],
            'BKGTYPE': ['BKGTYPE'],
            'BKGIS2D': ['BKGIS2D'],
            'BKGVALU': ['BKGVALU'],
            'BKGSIG': ['BKGSIG'],
            'BKGITER': ['BKGITER'],
            'BKGBOX': ['BKGBOX'],
            'BKGFILT': ['BKGFILT'],
            # Errormap
            'EMAPPATH': ['EMAPPATH'],
            'EMAPTYPE': ['EMAPTYPE'],
        }

        # Sort each list in the dictionary by string length (descending order)
        sorted_key_variants_upper = {
            key: sorted(variants, key=len, reverse=True)
            for key, variants in key_variants_upper.items()
        }
        return sorted_key_variants_upper
