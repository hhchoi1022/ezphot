#%%
import os
import logging
import logging.handlers
from pathlib import Path
from typing import Union
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
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
class BaseImage(TIPConfig):
    """ Handles FITS image processing and tracks its status """

    def __init__(self, path: Union[Path, str], telinfo : dict = None):
        path = Path(path)

        self.helper = Helper()
        self.path = path
        if telinfo is None:
            telinfo = self.helper.estimate_telinfo(self.path)
        self.telinfo = telinfo
        self.telkey = self._get_telkey()
        # Initialize or load status
        super().__init__(telkey = self.telkey)

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

    def clear(self, clear_data: bool = True, clear_header: bool = False):
        """Clear the image data and header"""
        if clear_data:
            self._data = None
        if clear_header:
            self._header = Header()
    
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

    def show(self, 
             cmap='gray', 
             scale='zscale', 
             downsample=4, 
             figsize=(8, 6), 
             title=None, 
             save_path: str = False,
             close_fig: bool = False):
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
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved: {save_path}")
        
        if close_fig:
            plt.close(fig)
        
        return fig, ax
        
    def show_position(self,
                    x: float,
                    y: float,
                    coord_type: str = 'pixel',
                    downsample: int = 4,
                    zoom_radius_pixel: int = 100,
                    cmap: str = 'gray',
                    scale: str = 'zscale',
                    figsize=(6, 6),
                    ax=None,
                    save_path: str = None,
                    title: bool = True):  # Added flag to control title
        """
        Show a zoomed-in region around a given position in the image.
        """
        import matplotlib.pyplot as plt
        from astropy.visualization import ZScaleInterval, MinMaxInterval
        from matplotlib.patches import Circle

        data = self.data
        wcs = self.wcs
        if data is None:
            print("No image data loaded.")
            return

        # Convert (RA, Dec) to pixel if needed
        if coord_type == 'coord':
            if wcs is None:
                print("No valid WCS for sky-to-pixel conversion.")
                return
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            coord = SkyCoord(x * u.deg, y * u.deg, frame='icrs')
            x_pix, y_pix = wcs.world_to_pixel(coord)
        elif coord_type == 'pixel':
            x_pix, y_pix = x, y
        else:
            raise ValueError("coord_type must be 'pixel' or 'coord'.")

        x_pix, y_pix = int(x_pix), int(y_pix)

        # Extract zoom window
        x_min = max(0, x_pix - zoom_radius_pixel)
        x_max = min(data.shape[1], x_pix + zoom_radius_pixel)
        y_min = max(0, y_pix - zoom_radius_pixel)
        y_max = min(data.shape[0], y_pix + zoom_radius_pixel)
        size = x_max - x_min
        cutout = data[y_min:y_max:downsample, x_min:x_max:downsample]

        # Scaling
        interval = ZScaleInterval() if scale == 'zscale' else MinMaxInterval()
        vmin, vmax = interval.get_limits(cutout)

        # Draw
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        ax.imshow(cutout, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax.add_patch(Circle(
            ((x_pix - x_min) // downsample, (y_pix - y_min) // downsample),
            radius=size * (0.08/downsample), edgecolor='red', facecolor='none', linewidth=0.5
        ))
        ax.axis('off')
        ax.set_aspect('auto')  # ? Avoid square enforcement

        if title is not None:
            ax.set_title(f"{title}", fontsize=8, pad=1)  # Less padding

        if save_path:
            fig.savefig(save_path, dpi=300)
            print(f"Saved: {save_path}")

        return fig, ax

    
    def run_ds9(self):
        self.helper.run_ds9(self.path)        
        
    @property
    def observatory(self):
        return str(self.telinfo['obs'])
    
    @property
    def telname(self):
        header = self._header
        for key in self.key_variants['TELESCOP']:
            if key in header:
                return str(header[key])
        return None
    @property
    def imgtype(self):
        header = self._header
        for key in self.key_variants['IMGTYPE']:
            if key in header:
                imgtype = header[key]
                imgtype_variants = dict(BIAS= ['BIAS', 'Bias', 'bias', 'ZERO', 'Zero', 'zero'],
                                        DARK= ['DARK', 'Dark', 'dark'],
                                        FLAT= ['FLAT', 'Flat', 'flat'],
                                        LIGHT= ['LIGHT', 'Light', 'light', 'OBJECT', 'Object', 'object'])
                for key, variants in imgtype_variants.items():
                    if imgtype in variants:
                        return key
        print('WARNING: IMGTYPE not found in header')
        return 'UNKNOWN'
    
    @property
    def altitude(self):
        header = self._header
        for key in self.key_variants['ALTITUDE']:
            if key in header:
                return float(header[key])
        return None

    
    @property
    def azimuth(self):
        header = self._header
        for key in self.key_variants['AZIMUTH']:
            if key in header:
                return float(header[key])
        return None

    @property 
    def ra(self):
        header = self._header
        for key in self.key_variants['RA']:
            if key in header:
                return float(header[key])
        return None
        
    @property
    def dec(self):
        header = self._header
        for key in self.key_variants['DEC']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def objname(self):
        header = self._header
        for key in self.key_variants['OBJECT']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def obsmode(self):
        header = self._header
        for key in self.key_variants['OBSMODE']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def specmode(self):
        header = self._header
        for key in self.key_variants['SPECMODE']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def filter(self):
        header = self._header
        for key in self.key_variants['FILTER']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def ccdtemp(self):
        header = self._header
        for key in self.key_variants['CCD-TEMP']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def obsdate(self):
        header = self._header
        # First, search for utcdate
        for key in self.key_variants['DATE-OBS']:
            if key in header:
                return Time(header[key], format = 'isot', scale = 'utc').isot
        # If not found, search for jd
        for key in self.key_variants['JD']:
            if key in header:
                return Time(header[key], format = 'jd').isot
        # If not found, search for mjd
        for key in self.key_variants['MJD']:
            if key in header:
                return Time(header[key], format = 'mjd').isot
        return None
    
    @property
    def mjd(self):
        header = self._header
        for key in self.key_variants['MJD']:
            if key in header:
                return float(header[key])
        for key in self.key_variants['JD']:
            if key in header:
                return Time(header[key], format = 'jd').mjd
        for key in self.key_variants['DATE-OBS']:
            if key in header:
                return Time(header[key], format = 'isot', scale = 'utc').mjd
        return None
    
    @property
    def jd(self):
        header = self._header
        for key in self.key_variants['JD']:
            if key in header:
                return float(header[key])
        for key in self.key_variants['MJD']:
            if key in header:
                return Time(header[key], format = 'mjd').jd
        for key in self.key_variants['DATE-OBS']:
            if key in header:
                return Time(header[key], format = 'isot', scale = 'utc').jd
        return None

    @property
    def exptime(self):
        header = self._header
        for key in self.key_variants['EXPTIME']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def binning(self):
        header = self._header
        for key in self.key_variants['BINNING']:
            if key in header:
                return int(header[key])
        return None
        
    @property
    def gain(self):
        header = self._header
        for key in self.key_variants['GAIN']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def egain(self):
        header = self._header
        for key in self.key_variants['EGAIN']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def naxis1(self):
        header = self._header
        for key in self.key_variants['NAXIS1']:
            if key in header:
                return int(header[key])
        return None

    @property
    def naxis2(self):
        header = self._header
        for key in self.key_variants['NAXIS2']:
            if key in header:
                return int(header[key])
        return None
    
    @property
    def ncombine(self):
        header = self._header
        for key in self.key_variants['NCOMBINE']:
            if key in header:
                return int(header[key])
        return None
    
    @property
    def biaspath(self):
        header = self._header
        for key in self.key_variants['BIASPATH']:
            if key in header:
                return str(header[key])
        return None
            
    @property
    def darkpath(self):
        header = self._header
        for key in self.key_variants['DARKPATH']:
            if key in header:
                return str(header[key])
        return None

    @property
    def flatpath(self):
        header = self._header
        for key in self.key_variants['FLATPATH']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def maskpath(self):
        header = self._header
        for key in self.key_variants['MASKPATH']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def masktype(self):
        header = self._header
        for key in self.key_variants['MASKTYPE']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def bkgpath(self):
        header = self._header
        for key in self.key_variants['BKGPATH']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def bkgtype(self):
        header = self._header
        for key in self.key_variants['BKGTYPE']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def emappath(self):
        header = self._header
        for key in self.key_variants['EMAPPATH']:
            if key in header:
                return str(header[key])
        return None

    @property
    def emaptype(self):
        header = self._header
        for key in self.key_variants['EMAPTYPE']:
            if key in header:
                return str(header[key])
        return None
            
    @property
    def fovx(self):
        pixscale = self.pixelscale
        if pixscale is not None:
            fovx = pixscale[0] * self.naxis1 / 3600  # Convert to degrees
            return float('%.3f' % fovx)
        else:
            return float('%.3f' %(self.telinfo['pixelscale'] * self.telinfo['x'] / 3600))
    
    @property
    def fovy(self):
        pixscale = self.pixelscale
        if pixscale is not None:
            fovy = pixscale[1] * self.naxis2 / 3600  # Convert to degrees
            return float('%.3f' % fovy)
        else:
            return float('%.3f' %(self.telinfo['pixelscale'] * self.telinfo['y'] / 3600))
    
    @property
    def wcs(self):
        """
        Returns the WCS information of the image.
        """
        try:
            wcs = WCS(self._header)
            return wcs
        except:
            return None
    
    @property
    def center(self):
        """
        Returns the center pixel (0-based) and its world coordinates (RA, Dec).
        """
        x_center = (self.naxis1 - 1) / 2
        y_center = (self.naxis2 - 1) / 2
        ra = dec = None

        if self.wcs is not None:
            try:
                skycoord = self.wcs.pixel_to_world(x_center, y_center)
                ra = skycoord.ra.deg
                dec = skycoord.dec.deg
            except Exception as e:
                print(f"WCS conversion failed: {e}")

        return {'x': x_center, 'y': y_center, 'ra': ra, 'dec': dec}
        
    @property
    def pixelscale(self):
        """
        Returns the pixel scale of the image.
        """
        try:
            return proj_plane_pixel_scales(self.wcs) * 3600  # Convert to arcseconds
        except:
            return None
    
    @property
    def zp(self):
        """
        Returns the zero point of the image.
        """
        header = self._header
        for key in self.key_variants['ZP']:
            if key in header:
                return float(header[key])
        return None

    @property
    def zperr(self):
        """
        Returns the zero point error of the image.
        """
        header = self._header
        for key in self.key_variants['ZPERR']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def depth(self):
        """
        Returns the depth of the image.
        """
        header = self._header
        for key in self.key_variants['DEPTH']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def seeing(self):
        """
        Returns the seeing of the image.
        """
        header = self._header
        for key in self.key_variants['SEEING']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def ncombine(self):
        """
        Returns the number of combined images.
        """
        header = self._header
        for key in self.key_variants['NCOMBINE']:
            if key in header:
                return int(header[key])
        return None
        
    @property
    def key_variants(self):
        # Define key variants, if a word is duplicated in the same variant, posit the word with the highest priority first
        key_variants_upper = {
            # Observation information
            'ALTITUDE': ['ALT', 'ALTITUDE', 'CENTALT'],
            'AZIMUTH': ['AZ', 'AZIMUTH', 'CENTAZ'],
            'GAIN': ['GAIN'],
            'EGAIN': ['EGAIN'],
            'CCD-TEMP': ['CCDTEMP', 'CCD-TEMP'],
            'FILTER': ['FILTER', 'FILTNAME', 'BAND'],
            'IMGTYPE': ['IMGTYPE', 'IMAGETYP', 'IMGTYP'],
            'EXPTIME': ['EXPTIME', 'EXPOSURE'],
            'DATE-OBS': ['DATE-OBS', 'OBSDATE', 'UTCDATE'],
            'DATE-LOC': ['DATE-LOC', 'DATE-LTC', 'LOCDATE', 'LTCDATE'],
            'JD' : ['JD', 'JD-HELIO', 'JD-UTC', 'JD-OBS'],
            'MJD' : ['MJD', 'MJD-HELIO', 'MJD-UTC', 'MJD-OBS'],
            'RA': ['CRVAL1', 'RA', 'OBJCTRA', 'OBSRA'],
            'DEC': ['CRVAL2', 'DEC', 'DECL', 'DEC.', 'DECL.', 'CRVAL2', 'OBJCTDEC', 'OBSDEC'],   
            'TELESCOP' : ['TELESCOP', 'TELNAME'],
            'BINNING': ['BINNING', 'XBINNING'],
            'OBJECT': ['OBJECT', 'OBJNAME', 'TARGET', 'TARNAME'],
            'OBJCTID': ['OBJCTID', 'OBJID', 'ID'],
            'OBSMODE': ['OBSMODE', 'MODE'],
            'SPECMODE': ['SPECMODE'],
            'NTELESCOP': ['NTELESCOP', 'NTEL'],
            'NCOMBINE': ['NCOMBINE', 'NCOMB'],
            'NOTE': ['NOTE'],
            'NAXIS1': ['NAXIS1'],
            'NAXIS2': ['NAXIS2'],
            # Additional key after processing
            'CTYPE1': ['CTYPE1'],
            'CTYPE2': ['CTYPE2'],
            'CRVAL1': ['CRVAL1'],
            'CRVAL2': ['CRVAL2'],
            'SEEING': ['SEEING'],
            'ELONGATION': ['ELONGATION', 'ELONG'],
            'SKYSIG': ['SKYSIG', 'SKY_SIG'],
            'SKYVAL': ['SKYVAL', 'SKY_VAL'],
            'ZP': ['ZP_AUTO', 'ZP_2'],
            'ZPERR': ['ZPERR_AUTO', 'EZP_2'],
            'DEPTH': ['UL5_APER_2', 'UL5_2'],
            # Path information
            'SAVEPATH': ['SAVEPATH'],
            'BIASPATH': ['BIASPATH'],
            'DARKPATH': ['DARKPATH'],
            'FLATPATH': ['FLATPATH'],
            'BKGPATH': ['BKGPATH'],
            'BKGTYPE': ['BKGTYPE'],
            'EMAPPATH': ['EMAPPATH'],
            'EMAPTYPE': ['EMAPTYPE'],
            'MASKPATH': ['MASKPATH'],
            'MASKTYPE': ['MASKTYPE'],
        }

        # Sort each list in the dictionary by string length (descending order)
        # sorted_key_variants_upper = {
        #     key: sorted(variants, key=len, reverse=True)
        #     for key, variants in key_variants_upper.items()
        # }
        return key_variants_upper

    def _get_telkey(self):
        """ Get the telescope name from the FITS header """
        telinfo = self.telinfo
        if telinfo['mode']:
            telkey = f"{telinfo['obs']}_{telinfo['ccd']}_{telinfo['mode']}_{telinfo['binning']}x{telinfo['binning']}"
        else:
            telkey = f"{telinfo['obs']}_{telinfo['ccd']}_{telinfo['binning']}x{telinfo['binning']}"
        return telkey

# %%