#%%
from pathlib import Path
import logging
from typing import Union

import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u


from ezphot.configuration import Configuration
from ezphot.helper import Helper

#%%
class LazyFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=True):
        # `delay=True` avoids creating the file immediately
        super().__init__(filename, mode, encoding, delay=delay)
    
# class Logger:
#     def __init__(self, logger_name):
#         self.path = logger_name
#         self._log = self.createlogger(logger_name)

#     def log(self):
#         return self._log

#     def createlogger(self, logger_name, logger_level='INFO'):
#         logger = logging.getLogger(logger_name)
#         if len(logger.handlers) > 0:
#             return logger  # Logger already exists

#         logger.setLevel(logger_level)
#         formatter = logging.Formatter(
#             datefmt='%Y-%m-%d %H:%M:%S',
#             fmt='[%(levelname)s] %(asctime)-15s | %(message)s'
#         )

#         # Stream Handler
#         streamHandler = logging.StreamHandler()
#         streamHandler.setLevel(logger_level)
#         streamHandler.setFormatter(formatter)
#         logger.addHandler(streamHandler)

#         # Lazy File Handler
#         fileHandler = LazyFileHandler(filename=logger_name, delay=True)
#         fileHandler.setLevel(logger_level)
#         fileHandler.setFormatter(formatter)
#         logger.addHandler(fileHandler)

#         return logger
    
#%%
class DummyImage(Configuration):
    """Base class for loading, managing, and inspecting genFITS images.

    Provides lazy-loading of generated FITS data and header, WCS access, metadata extraction,
    and visualization support. Used as the parent class for all generated images
    such as Mask, Background, and ErrorMap.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the FITS image file.
    """
    def __init__(self, path: Union[Path, str]):
        super().__init__()
        self.helper = Helper()
        self.path =  Path(path).resolve()
        
        self._hdul = None
        self._data = None
        self._header = Header()
    
    def rename(self, new_name: str, verbose: bool = True):
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
        self.helper.print(f"Renamed {old_path} to {new_path}", verbose)
    
    def clear(self,
              clear_data: bool = True,
              clear_header: bool = False,
              verbose: bool = True):
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
            if self._hdul is not None:
                try:
                    self._hdul.close()
                finally:
                    self._hdul = None
        if clear_header:
            self._header = Header()
        self.helper.print("Cleared data and header from memory.", verbose)

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
        self.data()
        self.header()

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

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap. Default is 'gray'.
        scale : str, optional
            Scaling method. Default is 'zscale'.
        downsample : int, optional
            Step size for downsampling via slicing. Default is 4.
        figsize : tuple, optional
            Matplotlib figure size. Default is (8, 6).
        title : str, optional
            Plot title. Default is None.
        save_path : str, optional
            Path to save the figure. Default is None.
        close_fig : bool, optional
            If True, close the figure after saving. Default is False.
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
                      # Position paramters
                      x: float,
                      y: float,
                      coord_type: str = 'pixel',
                      # Aperture parameters
                      radius_arcsec: float = None,
                      a_arcsec: float = None,
                      b_arcsec: float = None,
                      theta_deg: float = None,
                      aperture_linewidth: float = 1.0,
                      aperture_color: str = 'red',
                      aperture_label: str = 'detected',
                      # Visualization paramters
                      downsample: int = 4,
                      zoom_radius_pixel: int = 100,
                      cmap: str = 'gray',
                      scale: str = 'zscale',
                      figsize=(6, 6),
                      title: str = None,
                      title_fontsize: int = 18,
                      # Other parameters
                      ax=None,
                      save_path: str = None,
                      ):
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
        pixelscale = np.mean(self.pixelscale)  # arcsec / original pixel
        x_c, y_c = (x_pix - x_min)//downsample, (y_pix - y_min)//downsample

        if radius_arcsec is None:
            # keep your heuristic, but make units explicit
            radius = (size / downsample) * 0.08
        else:
            # convert arcsec → cutout pixels
            radius = radius_arcsec / (pixelscale * downsample)
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
                (x_c, y_c),
                width=6*a_arcsec/pixelscale, height=6*b_arcsec/pixelscale, angle=theta_deg,
                edgecolor=aperture_color, facecolor='none', linewidth= aperture_linewidth
            ))
            if aperture_label is not None:
                ax.text(
                    x_c,
                    y_c - 1.5 * radius,   # small offset below the aperture
                    aperture_label,
                    color=aperture_color,
                    fontsize=10,
                    ha='center',
                    va='top',
                    zorder=10,
                    bbox=dict(
                        facecolor='white',
                        edgecolor='none',
                        alpha=0.6,
                        pad=1.5
                    )
                )

        else:
            ax.add_patch(Circle(
                (x_c, y_c),
                radius=radius, edgecolor=aperture_color, facecolor='none', linewidth=aperture_linewidth
            ))
            if aperture_label is not None:
                ax.text(
                    x_c,
                    y_c - 1.5 * radius,   # small offset below the aperture
                    aperture_label,
                    color=aperture_color,
                    fontsize=10,
                    ha='center',
                    va='top',
                    zorder=10,
                    bbox=dict(
                        facecolor='white',
                        edgecolor='none',
                        alpha=0.6,
                        pad=1.5
                    )
                )
            
        ax.axis('off')
        ax.set_aspect('auto')  # ? Avoid square enforcement

        if title is not None:
            ax.set_title(f"{title}", fontsize=title_fontsize)

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
    def naxis1(self):
        """Number of pixels along the first axis."""
        for key in self._key_variants['NAXIS1']:
            if key in self._header:
                return self._header[key]
        return None

    @property
    def naxis2(self):
        """Number of pixels along the second axis."""
        for key in self._key_variants['NAXIS2']:
            if key in self._header:
                return self._header[key]
        return None
    
    @property
    def exptime(self):
        """Exposure time of the image."""
        for key in self._key_variants['EXPTIME']:
            if key in self._header:
                return self._header[key]
        return None
    
    @property
    def totexp(self):
        """Exposure time of the image."""
        for key in self._key_variants['TOTEXP']:
            if key in self._header:
                return self._header[key]
        return None

    @property
    def egain(self):
        """Electron gain of the image."""
        for key in self._key_variants['EGAIN']:
            if key in self._header:
                return self._header[key]
        return None
    
    @property
    def fovx(self):
        """Field of view along the first axis."""
        pixelscale = self.pixelscale
        if pixelscale is not None:
            fovx = pixelscale[0] * self.naxis1 / 3600  # Convert to degrees
            return float('%.3f' % fovx)
        else:
            return float('%.3f' %(self.telinfo['pixelscale'] * self.naxis1 / 3600))
    
    @property
    def fovy(self):
        """Field of view along the second axis."""
        pixelscale = self.pixelscale
        if pixelscale is not None:
            fovy = pixelscale[1] * self.naxis2 / 3600  # Convert to degrees
            return float('%.3f' % fovy)
        else:
            return float('%.3f' %(self.telinfo['pixelscale'] * self.naxis2 / 3600))
    
    @property
    def wcs(self):
        """WCS information of the image."""
        try:
            from astropy.wcs import WCS
            wcs = WCS(self._header)
            # Drop stale SIP distortion left over from a previous (astrometry.net)
            # solution when the current CTYPE declares a different distortion
            # (e.g. TPV from SCAMP). Otherwise astropy applies SIP *and* TPV,
            # producing position errors that grow toward the image corners.
            if wcs.sip is not None and not any(str(c).endswith('-SIP') for c in wcs.wcs.ctype):
                wcs.sip = None
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
    def data(self):
        """Image data array."""
        if not self.is_data_loaded and self.is_exists:
            try:
                self._hdul = fits.open(self.path, memmap=True)
                self._data = self._hdul[0].data
            except Exception as e:
                self._hdul = fits.open(self.path, memmap=False)
                self._data = self._hdul[0].data
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
        """Check if the image data is loaded."""
        return self._data is not None

    @property
    def is_header_loaded(self):
        """Check if the image header is loaded."""
        return isinstance(self._header, Header) and len(self._header) > 0

    @property
    def is_exists(self):
        """Check if the image file exists."""
        return self.path.exists()
        
    @property
    def _key_variants(self):
        # Define key variants, if a word is duplicated in the same variant, posit the word with the highest priority first
        key_variants_upper = {
            # Default key in rawimages 
            'NAXIS1': ['NAXIS1'],
            'NAXIS2': ['NAXIS2'],
            'EGAIN': ['EGAIN'],
            'TOTEXP': ['TOTEXP', 'EXPTIME', 'EXPOSURE'],
            'EXPTIME': ['EXPTIME', 'EXPOSURE'],
            'BINNING': ['BINNING', 'XBINNING'],
            'RDNOISE': ['RDNOISE'],
            'IMGTYPE': ['IMGTYPE'],
            'TGTPATH': ['TGTPATH'],
            'BIASPATH': ['BIASPATH'],
            'DARKPATH': ['DARKPATH'],
            'FLATPATH': ['FLATPATH'],
            # Mask
            'MASKTYPE': ['MASKTYPE'],
            'MASKATMP': ['MASKATMP'],
            # Background
            'BKGTYPE': ['BKGTYPE'],
            'BKGIS2D': ['BKGIS2D'],
            'BKGVALU': ['BKGVALU'],
            'BKGSIG': ['BKGSIG'],
            'BKGITER': ['BKGITER'],
            'BKGBOX': ['BKGBOX'],
            'BKGFILT': ['BKGFILT'],
            # Errormap
            'EMAPTYPE': ['EMAPTYPE'],
        }

        # Sort each list in the dictionary by string length (descending order)
        sorted_key_variants_upper = {
            key: sorted(variants, key=len, reverse=True)
            for key, variants in key_variants_upper.items()
        }
        return sorted_key_variants_upper
