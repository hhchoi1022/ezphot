#%%
import os
import json
import logging
from pathlib import Path
from typing import Union, Optional

import numpy as np
from numba import jit
from astropy.io import fits
from astropy.io.fits import Header
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from astropy import units as u
from astropy.wcs.utils import skycoord_to_pixel
import matplotlib.pyplot as plt

from types import SimpleNamespace
from ezphot.imageobjects import ScienceImage, ReferenceImage, Mask
from ezphot.helper import Helper


class Info:
    """Stores metadata of a FITS image with dot-access."""
    
    INFO_FIELDS = ["path", "target_img", "obsdate", "filter", "exptime", "depth", "seeing",
                   "catalog_type", "ra", "dec", "fov_ra", "fov_dec", 'objname',
                   "observatory", "telname", "aperture_diameter_arcsec"]
    DEFAULT_VALUES = [None] * len(INFO_FIELDS)

    def __init__(self, **kwargs):
        # Set defaults, then override with user-provided values
        self._fields = {
            field: kwargs.get(field, default)
            for field, default in zip(self.INFO_FIELDS, self.DEFAULT_VALUES)
        }

    def __getattr__(self, name):
        # Prevent infinite recursion when _fields is not yet initialized
        if '_fields' in self.__dict__ and name in self._fields:
            return self._fields[name]
        raise AttributeError(f"'Info' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_fields":
            super().__setattr__(name, value)
        elif "_fields" in self.__dict__ and name in self._fields:
            self._fields[name] = value
        else:
            raise AttributeError(f"'Info' object has no attribute '{name}'")

    def update(self, key, value):
        if key in self._fields:
            self._fields[key] = value
        else:
            print(f"WARNING: Invalid key: {key}")

    def to_dict(self):
        return dict(self._fields)

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: data.get(key) for key in cls.INFO_FIELDS})

    def __repr__(self):
        lines = [f"{key}: {value}" for key, value in self._fields.items()]
        return "Info ============================================\n  " + "\n  ".join(lines) + "\n==================================================="


class Catalog:
    """
    Catalog class for handling source catalog data.
    
    This class is designed to handle source catalog data from a FITS file. 
    It automatically loads the corresponding target image from the FITS file.

    """
    
    def __init__(self, path: Union[Path, str], catalog_type: str = 'all', info: Info = None, load: bool = True):
        """
        Initialize the Catalog instance.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the catalog file.
        catalog_type : str, optional
            Catalog type. Default is 'all'. ['all', 'reference', 'valid', 'transient', 'forced']
        info : Info, optional
            Info object. Default is None.
        load : bool, optional
            Whether to load the catalog data. Default is True.
        """
        path = Path(path)
        
        if catalog_type not in ['all', 'reference', 'valid', 'transient', 'forced']:
            raise ValueError(f"Invalid catalog type: {catalog_type}")
        self.helper = Helper()
        self.is_loaded = False
        self.path = path
        self.catalog_type = catalog_type
        self.target_img = None
        self._data = None
        self._target_data = None

        self.info = Info(path = str(path), catalog_type = catalog_type)
        if load:
            self.load_info()
            
        if info is not None:
            self.info = info
            self.info.path = str(self.path)
            self.info.catalog_type = catalog_type
            if self.info.target_img is not None:
                if Path(self.info.target_img).exists():
                    self.target_img = ScienceImage(self.info.target_img, telinfo = self.helper.estimate_telinfo(self.info.target_img), load = True)
        
        if self.target_img is None:
            self._load_target_img()

    def __repr__(self):
        return f"Catalog( N_selected/N_sources = {self.nselected}/{self.nsources}, is_exists={self.is_exists}, catalog_type={self.catalog_type}, path={self.path})"
    
    def help(self):
        """Print available public methods and their docstrings."""
        print(f"\n Help for {self.__class__.__name__}\n" + "="*40)
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue  # Skip dunder and private
            attr = getattr(self, attr_name)
            if callable(attr):
                doc = attr.__doc__.strip() if attr.__doc__ else "No documentation."
                print(f"{attr_name}()\n  >>> {doc}\n")
                
    def select_sources(self,
                       x: Union[float, list, np.ndarray],
                       y: Union[float, list, np.ndarray],
                       unit='coord',
                       matching_radius: float = 5.0):
        """
        Select sources from a catalog based on their positions.
        
        Selected sources are stored in self.target_data.
        
        Parameters
        ----------
        x : Union[float, list, np.ndarray]
            X coordinates of the sources.
        y : Union[float, list, np.ndarray]
            Y coordinates of the sources.
        unit : str, optional
            Unit of the coordinates. Default is 'coord'.
        matching_radius : float, optional
            Matching radius in arcseconds. Default is 5.0.

        Returns
        -------
        None
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        target_catalog = self.data

        if unit == 'pixel':
            catalog_coords = np.vstack((target_catalog['X_IMAGE'], target_catalog['Y_IMAGE'])).T
            input_coords = np.vstack((x, y)).T

            tree = cKDTree(catalog_coords)
            sep, idx = tree.query(input_coords, distance_upper_bound=matching_radius)

            valid = sep != np.inf
            sep = sep[valid]
            idx = idx[valid]
            matched_catalog = target_catalog[idx]

            # Sort by separation
            sort_idx = np.argsort(sep)
            self.target_data = matched_catalog[sort_idx]

        elif unit == 'coord':
            cat_sky = SkyCoord(ra=target_catalog['X_WORLD'], dec=target_catalog['Y_WORLD'], unit='deg')
            input_sky = SkyCoord(ra=x, dec=y, unit='deg')

            cat_xyz = np.vstack(cat_sky.cartesian.xyz).T
            input_xyz = np.vstack(input_sky.cartesian.xyz).T

            tree = cKDTree(cat_xyz)
            matching_radius_rad = (matching_radius / 3600.0) * (np.pi / 180)
            sep, idx = tree.query(input_xyz, distance_upper_bound=matching_radius_rad)

            valid = sep != np.inf
            sep = sep[valid]
            idx = idx[valid]
            matched_catalog = target_catalog[idx]
            sep_arcsec = np.rad2deg(sep) * 3600

            # Sort by separation
            sort_idx = np.argsort(sep_arcsec)
            self.target_data = matched_catalog[sort_idx]

        else:
            raise ValueError("unit must be either 'pixel' or 'coord'")
        
    def show_source(self,
                target_ra: float,
                target_dec: float,
                downsample: int = 4,
                zoom_radius_pixel: float = 50,
                matching_radius_arcsec: float = 3.0):
        """
        Show two-panel view of the target image with a single source marked (red).
        The left panel shows the full image, and the right panel shows a zoomed-in view of the target position.
        If a source is matched, it is marked with a blue circle.
        
        Parameters
        ----------
        target_ra : float
            Right ascension of the target source.
        target_dec : float
            Declination of the target source.
        downsample : int, optional
            Downsampling factor for the image. Default is 4.
        zoom_radius_pixel : float, optional
            Radius of the zoomed-in view in pixels. Default is 50.
        matching_radius_arcsec : float, optional
            Matching radius in arcseconds. Default is 3.0.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        """
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        import matplotlib.pyplot as plt
        import numpy as np

        # Load image if not yet loaded
        if self.target_img is None:
            load_result = self._load_target_img(target_img=None)

        # Convert RA/Dec to pixel coordinates
        coord = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
        x, y = self.target_img.wcs.world_to_pixel(coord)

        # Match source in catalog
        self.select_sources(
            x=[target_ra], y=[target_dec], unit='coord',
            matching_radius=matching_radius_arcsec)
        matched_catalog = self.target_data

        # If matched, get pixel coords of catalog source
        matched_xy = None
        if len(matched_catalog) > 0:
            matched_coord = SkyCoord(ra=matched_catalog[0]['X_WORLD'] * u.deg,
                                    dec=matched_catalog[0]['Y_WORLD'] * u.deg)
            matched_xy = self.target_img.wcs.world_to_pixel(matched_coord)

        # Downsampled dimensions for full image view
        image_shape = self.target_img.data.shape
        x_size_ds = image_shape[1] / downsample
        y_size_ds = image_shape[0] / downsample
        x_ds = x / downsample
        y_ds = y / downsample

        # Create figure with two subplots
        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Full image ---
        fig_full, _ = self.target_img.show(downsample=downsample, title=None)
        plt.close(fig_full)
        full_image = fig_full.axes[0].images[0]
        ax_full.imshow(full_image.get_array(), cmap=full_image.get_cmap(), origin='lower',
                    vmin=full_image.get_clim()[0], vmax=full_image.get_clim()[1])
        ax_full.set_title(f"Full Image of {self.target_img.objname}")
        ax_full.plot(x_ds, y_ds, 'ro', markersize=6, label='Requested Position')

        if matched_xy is not None:
            matched_x_ds, matched_y_ds = matched_xy[0] / downsample, matched_xy[1] / downsample
            ax_full.plot(matched_x_ds, matched_y_ds, 'bo', markersize=6, label='Matched Source')

        ax_full.legend()

        # --- Zoomed view ---
        fig_zoom, _ = self.target_img.show(downsample=1, title=None)
        plt.close(fig_zoom)
        zoom_image = fig_zoom.axes[0].images[0]
        ax_zoom.imshow(zoom_image.get_array(), cmap=zoom_image.get_cmap(), origin='lower',
                    vmin=zoom_image.get_clim()[0], vmax=zoom_image.get_clim()[1])
        ax_zoom.set_title("Zoom on Target")

        # Draw red circle for requested position
        pixel_scale = np.abs(self.target_img.wcs.pixel_scale_matrix[0, 0]) * 3600  # arcsec/pixel
        radius_pixel = matching_radius_arcsec / pixel_scale
        circ = plt.Circle((x, y), radius_pixel, color='red', fill=False, linestyle='--',
                        linewidth=2.5, alpha=0.7)
        ax_zoom.add_patch(circ)
        ax_zoom.text(x, y + 1.5 * radius_pixel, 'Requested', color='red', fontsize=13,
                    ha='center', va='center')

        # If matched, draw blue circle
        if matched_xy is not None:
            matched_x, matched_y = matched_xy
            circ_match = plt.Circle((matched_x, matched_y), radius_pixel, color='blue', fill=False,
                                    linestyle='-', linewidth=2.0, alpha=0.8)
            ax_zoom.add_patch(circ_match)
            ax_zoom.text(matched_x, matched_y - 1.5 * radius_pixel, 'Matched', color='blue', fontsize=13,
                        ha='center', va='center')

        # Zoom limits
        ax_zoom.set_xlim(x - zoom_radius_pixel, x + zoom_radius_pixel)
        ax_zoom.set_ylim(y - zoom_radius_pixel, y + zoom_radius_pixel)

        fig.tight_layout()
        plt.show()
        return fig
    
    def copy(self) -> "Catalog":
        """
        Return a deep copy of this Catalog instance.
        
        Returns
        -------
        copied_catalog : Catalog
            A deep copy of the Catalog instance.

        Examples
        --------
        >>> catalog = Catalog(path='catalog.fits')
        >>> copied_catalog = catalog.copy()
        >>> copied_catalog.path
        'catalog.fits'
        """

        new_instance = Catalog(
            path=self.path,
            catalog_type=self.catalog_type,
            info=Info.from_dict(self.info.to_dict()),
            load=False
        )

        # Manually copy loaded data and header
        new_instance.data = None if self.data is None else self.data.copy()
        new_instance.target_img = None if self.target_img is None else ScienceImage(self.target_img.path, telinfo=self.target_img.telinfo, load=True)

        return new_instance
    
    def write(self, format = 'ascii'):
        """
        Write catalog data to the savepath (self.savepath.savepath).
        
        Parameters
        ----------
        format : str, optional
            Format of the output file. Default is 'ascii'.

        Returns
        -------
        None

        Examples
        --------
        """
        if self.data is None:
            raise ValueError("Cannot save MaskImage: data is not registered.")
        os.makedirs(self.savepath.savedir, exist_ok=True)

        # Write to disk
        self.data.write(self.savepath.savepath, format=format, overwrite=True)
        self.save_info()
        
    def remove(self, 
               remove_main: bool = True, 
               remove_connected_files: bool = True,
               skip_exts: list = ['.png', '.cat'],
               verbose: bool = False) -> dict:
        """
        Remove the main FITS file and/or associated connected files.

        Parameters
        ----------
        remove_main : bool
            If True, remove the main FITS file (self.path)
        remove_connected_files : bool
            If True, remove associated files (status, mask, coadd, etc.)
        skip_exts : list
            List of file extensions to skip (e.g. ['.png', '.cat'])
        verbose : bool
            If True, print removal results

        Returns
        -------
        dict
            {file_path (str): success (bool)} for each file attempted
        """
        removed = {}

        def try_remove(p: Union[str, Path]):
            p = Path(p)
            if p.exists() and p.is_file():
                try:
                    p.unlink()
                    if verbose:
                        print(f"[REMOVE] {p}")
                    return True
                except Exception as e:
                    if verbose:
                        print(f"[FAILED] {p} - {e}")
                    return False
            return False

        # Remove main FITS file
        if remove_main and self.path and self.path.is_file():
            removed[str(self.path)] = try_remove(self.path)

        # Remove connected files
        if remove_connected_files:
            for f in self.connected_files:
                if f.suffix in skip_exts:
                    if verbose:
                        print(f"[SKIP] {f} (skipped due to extension)")
                    continue
                removed[str(f)] = try_remove(f)

        return removed

    def apply_mask(self, 
                   target_ivpmask: Mask,
                   x_key: str = 'X_IMAGE',
                   y_key: str = 'Y_IMAGE'):
        """
        Apply a mask to the catalog.
        
        Parameters
        ----------
        target_ivpmask : Mask
            Mask to apply to the catalog.
        x_key : str, optional
            Column name for X coordinates. Default is 'X_IMAGE'.
        y_key : str, optional
            Column name for Y coordinates. Default is 'Y_IMAGE'.

        Returns
        -------
        masked_sources : Table
            Catalog with sources that are not masked.
        """
        mask = target_ivpmask.data
        ny, nx = mask.shape
        
        # Round or convert positions to int for indexing
        x = np.round(self.data['X_IMAGE']).astype(int)
        y = np.round(self.data['Y_IMAGE']).astype(int)
        
        # Ensure coordinates are within image bounds
        valid = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
        x_valid = x[valid]
        y_valid = y[valid]
        
        # Check if pixel is masked (== 0)
        mask_values = mask[y_valid, x_valid]
        is_masked = (mask_values == 0)
        
        # Apply final selection
        final_indices = np.where(valid)[0][is_masked]
        masked_sources = self.data[final_indices]

        return masked_sources
        
    def to_stamp(self,
                 target_img: Union[ScienceImage, ReferenceImage],
                 sort_by: str = 'FLUX_AUTO',
                 max_number: int = 50000):
        """
        Convert X_WORLD and Y_WORLD to pixel coordinates and save to a stamp catalog.
        
        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage
            Target image to convert to pixel coordinates.
        sort_by : str, optional
            Column name to sort the catalog by. Default is 'FLUX_AUTO'.
        max_number : int, optional
            Maximum number of sources to save. Default is 50000.

        Returns
        -------
        stamppath : str
            Path to the stamp catalog.
        """
        # Convert X_WORLD and Y_WORLD to pixel coordinates and save to a stamp catalog.
        stamppath = self.savepath.stamppath
        wcs = target_img.wcs
        if sort_by in self.data.colnames:
            self.data.sort(sort_by)
        ra_deg = self.data['X_WORLD']
        dec_deg = self.data['Y_WORLD']
        skycoord = SkyCoord(ra=ra_deg, dec=dec_deg, unit = 'deg')
        x_pix, y_pix = skycoord_to_pixel(skycoord, wcs, origin=0)        
        if len(x_pix) > max_number:
            x_pix = x_pix[:max_number]
            y_pix = y_pix[:max_number]
        # Save stamp catalog to a file.
        with open(stamppath, "w") as f:
            for x, y in zip(x_pix, y_pix):
                f.write(f"{round(x,3)} {round(y,3)} \n")
                
        return stamppath
    
    def to_region(self, reg_size: float = 6.0, shape : str = 'circle'):
        """
        Convert X_IMAGE and Y_IMAGE to a region file.
        
        Parameters
        ----------
        reg_size : float, optional
            Size of the region in pixels. Default is 6.0.
        shape : str, optional
            Shape of the region. Default is 'circle'.
        """

        reg_x = self.data['X_IMAGE']
        reg_y = self.data['Y_IMAGE']
        
        reg_a = None
        reg_b = None
        reg_theta = None
        if shape != 'circle':
            if 'A_IMAGE' not in self.data.colnames or 'B_IMAGE' not in self.data.colnames or 'THETA_IMAGE' not in self.data.colnames:
                raise ValueError("For non-circle shapes, A_IMAGE, B_IMAGE, and THETA_IMAGE must be present in the catalog data.")
            reg_a = self.data['A_IMAGE']
            reg_b = self.data['B_IMAGE']
            reg_theta = self.data['THETA_IMAGE']
        
        region_path =  str(self.savepath.savepath) + '.reg'
        self.to_regions(reg_x = reg_x, 
                        reg_y = reg_y, 
                        reg_a = reg_a,
                        reg_b = reg_b,
                        reg_theta = reg_theta,
                        reg_size = reg_size,
                        output_file_path = region_path)
        return region_path

    def save_info(self, verbose = False):
        """
        Save processing info to a JSON file.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print the path of the info file. Default is False.

        Returns
        -------
        None
        """
        with open(self.savepath.infopath, 'w') as f:
            json.dump(self.info.to_dict(), f, indent=4)
        self.helper.print(f"Saved: {self.savepath.infopath}", verbose)
    
    def load_info(self, verbose = False):
        """
        Load processing info from a JSON file.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print the path of the info file. Default is False.

        Returns
        -------
        info : Info
            Info object loaded from the JSON file.
        """
        if not self.savepath.infopath.exists():
            self.helper.print(f"Info file does not exist: {self.savepath.infopath}", verbose)
            return self.info
        
        with open(self.savepath.infopath, 'r') as f:
            data = json.load(f)
        
        self.info = Info.from_dict(data)
        if self.info.target_img is not None:
            target_path = Path(self.info.target_img)
            if not target_path.exists():
                self.helper.print(f"Target image does not exist: {target_path}", verbose)
                return self.info
            
            self.target_img = ScienceImage(target_path, telinfo=self.helper.estimate_telinfo(target_path), load=True)
            self.is_loaded = True
        self.helper.print(f"Loaded: {self.savepath.infopath}", verbose)
        return self.info

    def _find_corresponding_fits(self) -> Optional[Path]:
        search_dirs = [self.path.parent, self.path.parent.parent]       
        
        # Iteravely strip suffixes from the path to find candidates
        candidates = []
        path = self.path
        while path.suffix:
            path = path.with_suffix('')
            if path.suffix.startswith('.fits'):
                candidates.append(path.name)
            else:
                candidate = Path(str(path) + '.fits')
                if candidate.name not in candidates:
                    candidates.append(candidate.name)

        # Search for candidate names in possible directories
        for directory in search_dirs:
            for name in candidates:
                candidate_path = directory / name
                if candidate_path.exists():
                    return candidate_path

        print(f"[WARNING] No matching .fits found for: {self.path}")
        return None
        
    def _load_target_img(self, target_img: Union[ScienceImage, ReferenceImage] = None):
        if target_img is None:
            target_path = self._find_corresponding_fits()
            if target_path is None:
                print(f"[ERROR] No corresponding FITS file found for {self.path}")
                return False
            target_img = ScienceImage(target_path, telinfo = self.helper.estimate_telinfo(target_path), load = True)
        
        self.target_img = target_img
        self.info.target_img = str(target_img.path)
        self.info.ra = target_img.ra
        self.info.dec = target_img.dec
        self.info.fov_ra = target_img.fovx
        self.info.fov_dec = target_img.fovy
        self.info.objname = target_img.objname
        self.info.obsdate = target_img.obsdate
        self.info.filter = target_img.filter
        self.info.exptime = target_img.exptime
        self.info.depth = target_img.depth
        self.info.seeing = target_img.seeing
        self.info.observatory = target_img.observatory
        self.info.telname = target_img.telname
        self.is_loaded = True
        return True

    @property
    def savedir(self) -> Union[Path, None]:
        """
        Return the directory where this image and associated files will be saved.
        If a custom savedir was set, use it. Otherwise, build from config and metadata.
        Returns None if required fields are not available.
        """
        # Use manually set savedir if provided
        if hasattr(self, '_savedir') and self._savedir is not None:
            return self._savedir

        # Default construction from config
        base_dir = self.path.parent
        return base_dir 
    
    @savedir.setter
    def savedir(self, value: Union[str, Path]):
        """
        Set a custom directory for saving the image and associated products.
        """
        if value is None:
            self._savedir = None
            return
        value = Path(value)
        if value.is_file():
            value = value.parent
        self._savedir = value

    @property
    def savepath(self):
        """Dynamically builds save paths based on the path"""
        savedir = self.savedir
        filename = self.path.name
        return SimpleNamespace(
            savedir = savedir,
            savepath = savedir / filename,
            refcatalogpath = (savedir / filename).with_suffix('.refcat'),
            transientcatalogpath = (savedir / filename).with_suffix('.transient'),
            candidatecatalogpath = (savedir / filename).with_suffix('.candidate'),
            stamppath = savedir / (filename + '.stamp'),
            infopath= savedir / (filename + '.info'))
    
    @property
    def connected_files(self) -> set:
        """
        Return all associated files that would be deleted in `remove()` if remove_connected_files=True,
        excluding the main FITS file (`self.path`).

        Only includes existing files, not directories.

        Returns
        -------
        connected_files : set
            All connected auxiliary files.
        """
        connected = set()

        # Files in same directory that start with the same base name (excluding self.path)
        base_dir = self.path.parent
        base_name = self.path.name
        for f in base_dir.iterdir():
            if f.is_file() and f.name.startswith(base_name) and f != self.path:
                connected.add(f)

        # Files explicitly listed in savepath (excluding self.path)
        for p in vars(self.savepath).values():
            if isinstance(p, Path) and p.exists() and p.is_file() and p != self.path:
                connected.add(p)

        return connected
        
    @property
    def data(self):
        """Lazy-load table data from path by trying multiple formats."""
        if not self.is_data_loaded and self.is_exists:
            tried_formats = [
                'fits',
                'ascii.sextractor',
                'ascii',
                'csv',
                'ascii.basic',
                'ascii.commented_header',
                'ascii.tab',
                'ascii.fast_no_header',
            ]
            for fmt in tried_formats:
                try:
                    self._data = Table.read(self.path, format=fmt)
                    self._target_data = self._data.copy()  # Keep a copy of the original data
                    return self._data  # Success
                except Exception:
                    continue  # Try next format
            self._data = None
        return self._data     
    
    @data.setter
    def data(self, value):
        self._data = value

    @property
    def is_data_loaded(self):
        """Check if the data is loaded."""
        return self._data is not None
    
    @property
    def target_data(self):
        """Return the selected sources by self.select_sources()."""
        if self._target_data is None:
            return self.data
        return self._target_data
    
    @target_data.setter
    def target_data(self, value):
        """Set the target data and update the info."""
        self._target_data = value
    
    @property
    def is_exists(self):
        """Check if the catalog file exists."""
        return self.path.exists()
    
    @property
    def is_saved(self):
        """Check if the catalog has been saved."""
        if self.savepath.savepath is None:
            return False
        return self.savepath.savepath.exists()
    
    @property
    def nsources(self):
        """Number of sources in the catalog."""
        if self.is_data_loaded:
            return len(self.data)
        return None
    
    @property
    def nselected(self):
        """Number of selected sources in the target data."""
        if self._target_data is not None:
            return len(self._target_data)
        return None
    

#%%
if __name__ == "__main__":
    catalog_path = '/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/NGC6121/7DT01/m400/calib_7DT01_NGC6121_20240722_010055_m400_100.com.fits.cat'
    self = Catalog(path=catalog_path, catalog_type ='all', load = True)

    #sources =  self.data[(self.data['X_IMAGE'] < 4600) & (self.data['Y_IMAGE'] < 3100) & (self.data['X_IMAGE'] > 4400) & (self.data['Y_IMAGE'] > 2900)]
    # source = sources[4]
    # target_ra = source['X_WORLD']
    # target_dec = source['Y_WORLD']
    # matching_radius_arcsec = 3
    #self.show_source(target_ra=target_ra, target_dec=target_dec, downsample=4, zoom_radius_pixel=50)
    #target_img = ScienceImage(self.target_path, telinfo = self.helper.estimate_telinfo(self.target_path), load = False)
    #tbl = self.search_sources(x = 233.890152, y = -67.523614423, unit = 'coord')
    #self.load_from_target_path()
    #self.save_info()
    #A = self._find_corresponding_fits()
    
# %%
