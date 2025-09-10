#%%
import logging
from pathlib import Path
from typing import Union

import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u

from ezphot.configuration import Configuration
from ezphot.helper import Helper

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
class BaseImage(Configuration):
    """Base class for loading, managing, and inspecting FITS images.

    Provides lazy-loading of FITS data and header, WCS access, metadata extraction,
    and visualization support. Used as the parent class for all telescope-acquired images
    such as science, reference, and calibration frames.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the FITS image file.
    telinfo : dict, optional
        Telescope information dictionary. If not provided, it will be estimated automatically.
    """

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

    def rename(self, new_name: str):
        """Rename the image file and update the internal path.
        
        Parameters
        ----------
        new_name : str
            New name of the image file.
        
        Returns
        -------
        None
        """
        old_path = self.path
        new_path = self.path.parent / new_name

        # If the target exists, remove it (overwrite)
        if new_path.exists():
            new_path.unlink()  # remove the existing file

        old_path.rename(new_path)
        self.path = new_path
        print(f"Renamed {old_path} to {new_path}")

    def clear(self, clear_data: bool = True, clear_header: bool = False, verbose: bool = True):
        """Clear the image data and/or header from memory.
        
        Parameters
        ----------
        clear_data : bool, optional
            If True, clear the image data. Default is True.
        clear_header : bool, optional
            If True, clear the image header. Default is False.
        """
        if clear_data:
            self._data = None
        if clear_header:
            self._header = Header()
        self.helper.print("Cleared data and/or header from memory.", verbose)

    def update_header(self, **kwargs):
        """Update FITS header values using known key variants.
        
        Parameters
        ----------
        kwargs : dict
            Keyword arguments to update the FITS header.
        
        Returns
        -------
        None
        """
        if self._header is None:
            print("WARNING: Header is not loaded. Cannot update.")
            return
        else:
            for key, value in kwargs.items():
                key_upper = key.upper()
                if key_upper in self._key_variants.keys():
                    key_variants = self._key_variants[key_upper]
                    for key_variant in key_variants:
                        if key_variant in self._header:
                            self._header[key_variant] = value
                else:
                    print(f'WARNING: Key {key} not found in key_variants.')

    def load(self):
        """Load both image data and header from disk.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        if not self.is_exists:
            raise FileNotFoundError(f'File not found: {self.path}')
        self.data
        self.header
        
    def show(self, 
            coord_type: str = "pixel",
            cmap='gray', 
            scale='zscale', 
            downsample=4, 
            figsize=(8, 6), 
            title=None, 
            save_path: str = False,
            close_fig: bool = False):
        """
        Visualize the FITS image with optional WCS overlay and downsampling.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.visualization import ZScaleInterval, MinMaxInterval
        from astropy import units as u

        # ----------------------------
        # 1. Prepare data and WCS
        # ----------------------------
        if self.data is None:
            print("WARNING: Image data is not loaded. Please load the image first.")
            return

        data = np.nan_to_num(self.data, nan=0.0, posinf=0.0, neginf=0.0)
        w = getattr(self, "wcs", None)

        if downsample > 1:
            data = data[::downsample, ::downsample]
            if coord_type == "coord" and w is not None:
                w = w.deepcopy()
                if w.wcs.has_cd():
                    w.wcs.cd *= downsample
                elif w.wcs.has_pc():
                    w.wcs.cdelt *= downsample
                else:
                    w.wcs.cdelt *= downsample
                w.wcs.crpix = (w.wcs.crpix - 1) / downsample + 1

        # ----------------------------
        # 2. Prepare visualization
        # ----------------------------
        interval = ZScaleInterval() if scale == 'zscale' else MinMaxInterval()
        vmin, vmax = interval.get_limits(data)

        use_wcs = (coord_type == "coord" and w is not None)

        # ----------------------------
        # 3. Plotting
        # ----------------------------
        if use_wcs:
            try:
                fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': w})
                ax.grid(color='white', ls='dotted')

                # Force decimal degree ticks with 0.1 degree spacing
                lon, lat = ax.coords[0], ax.coords[1]
                lon.set_format_unit(u.deg)
                lat.set_format_unit(u.deg)
                
                lon.set_major_formatter('d.dd')  # RA with 4 decimals
                lat.set_major_formatter('d.d')    # Dec with 3 decimals
                
                # Set grid spacing to get exactly 5 grid lines for each axis
                # Calculate spacing based on field of view
                fov_ra = self.fovx  # Field of view in RA (degrees)
                fov_dec = self.fovy  # Field of view in Dec (degrees)
                
                # Correct RA spacing for declination (RA lines converge toward poles)
                center = self.center
                if center['dec'] is not None:
                    dec_rad = np.radians(center['dec'])
                    ra_spacing = (fov_ra / 4) / np.cos(dec_rad)
                else:
                    ra_spacing = fov_ra / 4
                
                dec_spacing = fov_dec / 4
                
                lon.set_ticks(spacing=ra_spacing * u.deg)
                lat.set_ticks(spacing=dec_spacing * u.deg)
                
                # Rotate x-axis tick labels by 45 degrees
                ax.tick_params(axis='x', rotation=45)

                ax.set_xlabel("RA [deg]")
                ax.set_ylabel("Dec [deg]")
            except Exception as e:
                print(f"WARNING: WCS plotting failed ({e}). Falling back to pixel coords.")
                fig, ax = plt.subplots(figsize=figsize)
                use_wcs = False
        else:
            fig, ax = plt.subplots(figsize=figsize)

        # Show image
        img = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

        # Colorbar
        cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label("Pixel value")

        # Pixel coordinate ticks
        if not use_wcs:
            yticks = np.linspace(0, data.shape[0]-1, num=6, dtype=int)
            xticks = np.linspace(0, data.shape[1]-1, num=6, dtype=int)
            ax.set_yticks(yticks)
            ax.set_yticklabels((yticks * downsample).astype(int))
            ax.set_xticks(xticks)
            ax.set_xticklabels((xticks * downsample).astype(int))
            ax.set_xlabel("X pixel")
            ax.set_ylabel("Y pixel")

        # Title
        ax.set_title(title or self.path.name)

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
                    radius_arcsec: float = None,
                    a_arcsec: float = None,
                    b_arcsec: float = None,
                    theta_deg: float = None,
                    coord_type: str = 'pixel',
                    downsample: int = 1,
                    zoom_radius_pixel: int = 100,
                    cmap: str = 'gray',
                    scale: str = 'zscale',
                    figsize=(6, 6),
                    ax=None,
                    save_path: str = None,
                    title: bool = True):
        """
        Display a zoomed-in view around a given (x, y) or (RA, Dec) position.

        Parameters
        ----------
        x : float
            X-coordinate (pixel or RA)
        y : float
            Y-coordinate (pixel or Dec)
        radius_arcsec : float, optional
            Radius of the circle in arcseconds. Default is None.
        a_arcsec : float, optional
            Semi-major axis of the ellipse in arcseconds. Default is None.
        b_arcsec : float, optional
            Semi-minor axis of the ellipse in arcseconds. Default is None.
        theta_deg : float, optional
            Position angle of the ellipse in degrees. Default is None.
        coord_type : str, optional
            'pixel' or 'coord'. Default is 'pixel'.
        downsample : int, optional
            Step size for downsampling via slicing. Default is 4.
        zoom_radius_pixel : int, optional
            Radius of the zoom-in region in pixels. Default is 100.
        cmap : str, optional
            Matplotlib colormap. Default is 'gray'.
        scale : str, optional
            Scaling method. Default is 'zscale'.
        figsize : tuple, optional
            Matplotlib figure size. Default is (6, 6).
        ax : matplotlib Axes object, optional
            If provided, the image will be plotted on this axis. Default is None.
        save_path : str, optional
            Path to save the figure. Default is None.
        title : bool, optional
            Whether to display the title. Default is True.
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
        if radius_arcsec is None:
            radius = size * (0.08/downsample)
        else:
            radius = radius_arcsec / np.mean(self.pixelscale)
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
        if a_arcsec is not None and b_arcsec is not None and theta_deg is not None:
            from matplotlib.patches import Ellipse
            ax.add_patch(Ellipse(
                ((x_pix - x_min) // downsample, (y_pix - y_min) // downsample),
                width=6*a_arcsec/np.mean(self.pixelscale), height=6*b_arcsec/np.mean(self.pixelscale), angle=theta_deg,
                edgecolor='red', facecolor='none', linewidth=0.5
            ))
        else:
            ax.add_patch(Circle(
                ((x_pix - x_min) // downsample, (y_pix - y_min) // downsample),
                radius=radius, edgecolor='red', facecolor='none', linewidth=0.5
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
        """Open the image with SAOImage DS9.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self.helper.run_ds9(self.path)        
        
    @property
    def data(self):
        """Image data array."""
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
        """FITS header object."""
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
        """Whether the image data is loaded."""
        return self._data is not None

    @property
    def is_header_loaded(self):
        """Whether the image header is loaded."""
        return isinstance(self._header, Header) and len(self._header) > 0

    @property
    def is_exists(self):
        """Whether the image file exists."""
        return self.path.exists()
        
    @property
    def observatory(self):
        """ Name of the observatory."""
        return str(self.telinfo['obs'])
    
    @property
    def telname(self):
        """ Name of the telescope."""
        header = self._header
        for key in self._key_variants['TELESCOP']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def imgtype(self):
        """ Type of the image. Among BIAS, DARK, FLAT, LIGHT, OBJECT, UNKNOWN."""
        header = self._header
        for key in self._key_variants['IMGTYPE']:
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
        """ Altitude of the telescope.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        altitude : float
            Altitude of the telescope.
        """
        header = self._header
        for key in self._key_variants['ALTITUDE']:
            if key in header:
                return float(header[key])
        return None

    
    @property
    def azimuth(self):
        """Azimuth of the telescope."""
        header = self._header
        for key in self._key_variants['AZIMUTH']:
            if key in header:
                return float(header[key])
        return None

    @property 
    def ra(self):
        """Right ascension of the target."""
        header = self._header
        for key in self._key_variants['RA']:
            if key in header:
                return float(header[key])
        return None
        
    @property
    def dec(self):
        """Declination of the target."""
        header = self._header
        for key in self._key_variants['DEC']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def objname(self):
        """Object name."""
        header = self._header
        for key in self._key_variants['OBJECT']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def obsmode(self):
        """Observation mode."""
        header = self._header
        for key in self._key_variants['OBSMODE']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def specmode(self):
        """Spectroscopic mode."""
        header = self._header
        for key in self._key_variants['SPECMODE']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def filter(self):
        """Filter name."""
        header = self._header
        for key in self._key_variants['FILTER']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def ccdtemp(self):
        """CCD temperature."""
        header = self._header
        for key in self._key_variants['CCD-TEMP']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def obsdate(self):
        """Observation date in UTC."""
        header = self._header
        # First, search for utcdate
        for key in self._key_variants['DATE-OBS']:
            if key in header:
                return Time(header[key], format = 'isot', scale = 'utc').isot
        # If not found, search for jd
        for key in self._key_variants['JD']:
            if key in header:
                return Time(header[key], format = 'jd').isot
        # If not found, search for mjd
        for key in self._key_variants['MJD']:
            if key in header:
                return Time(header[key], format = 'mjd').isot
        return None
    
    @property
    def mjd(self):
        """Modified Julian date of the observation."""
        header = self._header
        for key in self._key_variants['MJD']:
            if key in header:
                return float(header[key])
        for key in self._key_variants['JD']:
            if key in header:
                return Time(header[key], format = 'jd').mjd
        for key in self._key_variants['DATE-OBS']:
            if key in header:
                return Time(header[key], format = 'isot', scale = 'utc').mjd
        return None
    
    @property
    def jd(self):
        """Julian date of the observation."""
        header = self._header
        for key in self._key_variants['JD']:
            if key in header:
                return float(header[key])
        for key in self._key_variants['MJD']:
            if key in header:
                return Time(header[key], format = 'mjd').jd
        for key in self._key_variants['DATE-OBS']:
            if key in header:
                return Time(header[key], format = 'isot', scale = 'utc').jd
        return None

    @property
    def exptime(self):
        """Exposure time of the image."""
        header = self._header
        for key in self._key_variants['EXPTIME']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def binning(self):
        """Binning of the image."""
        header = self._header
        for key in self._key_variants['BINNING']:
            if key in header:
                return int(header[key])
        return None
        
    @property
    def gain(self):
        """Gain of the image."""
        header = self._header
        for key in self._key_variants['GAIN']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def egain(self):
        """Electron gain of the image."""
        header = self._header
        for key in self._key_variants['EGAIN']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def naxis1(self):
        """Number of pixels along the first axis."""
        header = self._header
        for key in self._key_variants['NAXIS1']:
            if key in header:
                return int(header[key])
        return None

    @property
    def naxis2(self):
        """Number of pixels along the second axis."""
        header = self._header
        for key in self._key_variants['NAXIS2']:
            if key in header:
                return int(header[key])
        return None
    
    @property
    def ncombine(self):
        """Number of combined images."""
        header = self._header
        for key in self._key_variants['NCOMBINE']:
            if key in header:
                return int(header[key])
        return None
    
    @property
    def biaspath(self):
        """Path to the bias image."""
        header = self._header
        for key in self._key_variants['BIASPATH']:
            if key in header:
                return str(header[key])
        return None
            
    @property
    def darkpath(self):
        """Path to the dark image."""
        header = self._header
        for key in self._key_variants['DARKPATH']:
            if key in header:
                return str(header[key])
        return None

    @property
    def flatpath(self):
        """Path to the flat image."""
        header = self._header
        for key in self._key_variants['FLATPATH']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def maskpath(self):
        """Path to the mask image."""
        header = self._header
        for key in self._key_variants['MASKPATH']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def masktype(self):
        """Type of the mask image."""
        header = self._header
        for key in self._key_variants['MASKTYPE']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def bkgpath(self):
        """Path to the background image."""
        header = self._header
        for key in self._key_variants['BKGPATH']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def bkgtype(self):
        """Type of the background image."""
        header = self._header
        for key in self._key_variants['BKGTYPE']:
            if key in header:
                return str(header[key])
        return None
    
    @property
    def emappath(self):
        """Path to the emap image."""
        header = self._header
        for key in self._key_variants['EMAPPATH']:
            if key in header:
                return str(header[key])
        return None

    @property
    def emaptype(self):
        """Type of the emap image."""
        header = self._header
        for key in self._key_variants['EMAPTYPE']:
            if key in header:
                return str(header[key])
        return None
            
    @property
    def fovx(self):
        """Field of view along the first axis."""
        pixscale = self.pixelscale
        if pixscale is not None:
            fovx = pixscale[0] * self.naxis1 / 3600  # Convert to degrees
            return float('%.3f' % fovx)
        else:
            return float('%.3f' %(self.telinfo['pixelscale'] * self.telinfo['x'] / 3600))
    
    @property
    def fovy(self):
        """Field of view along the second axis."""
        pixscale = self.pixelscale
        if pixscale is not None:
            fovy = pixscale[1] * self.naxis2 / 3600  # Convert to degrees
            return float('%.3f' % fovy)
        else:
            return float('%.3f' %(self.telinfo['pixelscale'] * self.telinfo['y'] / 3600))
    
    @property
    def wcs(self):
        """WCS information of the image."""
        try:
            wcs = WCS(self._header)
            if self.imgtype.upper() in ['BIAS', 'DARK', 'FLAT']:
                raise ValueError("WCS is not available for bias, dark, or flat images.")
            return wcs
        except:
            return None
    
    @property
    def center(self):
        """Center pixel (0-based) and its world coordinates (RA, Dec)."""
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
        """Pixel scale of the image."""
        try:
            return proj_plane_pixel_scales(self.wcs) * 3600  # Convert to arcseconds
        except:
            return None
    
    @property
    def zp(self):
        """Zero point of the image."""
        header = self._header
        for key in self._key_variants['ZP']:
            if key in header:
                return float(header[key])
        return None

    @property
    def zperr(self):    
        """Zero point error of the image."""
        header = self._header
        for key in self._key_variants['ZPERR']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def depth(self):
        """Depth of the image."""
        header = self._header
        for key in self._key_variants['DEPTH']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def seeing(self):
        """Seeing of the image."""
        header = self._header
        for key in self._key_variants['SEEING']:
            if key in header:
                return float(header[key])
        return None
    
    @property
    def ncombine(self):
        """Number of combined images."""
        header = self._header
        for key in self._key_variants['NCOMBINE']:
            if key in header:
                return int(header[key])
        return None
        
    @property
    def _key_variants(self):
        '''Key variants of the FITS header.'''
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