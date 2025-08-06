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
from tippy.imageobjects import ScienceImage, ReferenceImage, Mask
from tippy.helper import Helper


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


class TIPCatalog(Helper):
    
    def __init__(self, path: Union[Path, str], catalog_type: str = 'all', info: Info = None, load: bool = True):
        path = Path(path)
        super().__init__()
        
        if catalog_type not in ['all', 'reference', 'valid', 'transient', 'forced']:
            raise ValueError(f"Invalid catalog type: {catalog_type}")
        self.is_loaded = False
        self.path = path
        self.catalog_type = catalog_type
        self.target_img = None
        self._data = None
        self._target_data = None

        self.info = Info(path = str(path), catalog_type = catalog_type)
        if load:
            self.load_info()
            if not self.is_loaded:
                self.load_target_img()
            
        if info is not None:
            self.info = info
            self.info.path = str(self.path)
            self.info.catalog_type = catalog_type
            if self.info.target_img is not None:
                if Path(self.info.target_img).exists():
                    self.target_img = ScienceImage(self.info.target_img, telinfo = self.estimate_telinfo(self.info.target_img), load = True)

    def __repr__(self):
        return f"TIPCatalog( N_selected/N_sources = {self.nselected}/{self.nsources}, is_exists={self.is_exists}, catalog_type={self.catalog_type}, path={self.path})"
    
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
    
    @property
    def savepath(self):
        savedir = self.path.parent
        savedir.mkdir(parents=True, exist_ok=True)
        
        filename = self.path.name
        return SimpleNamespace(
            savedir = savedir,
            savepath = savedir / filename,
            refcatalogpath = self.path.with_suffix('.refcat'),
            transientcatalogpath = self.path.with_suffix('.transient'),
            candidatecatalogpath = self.path.with_suffix('.candidate'),
            stamppath = self.path.with_suffix('.stamp'),
            infopath=savedir / (filename + '.info'))
        
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
        return self._data is not None
    
    @property
    def target_data(self):
        """Return a copy of the target data."""
        if self._target_data is None:
            return self.data
        return self._target_data
    
    @target_data.setter
    def target_data(self, value):
        """Set the target data and update the info."""
        self._target_data = value
    
    @property
    def is_exists(self):
        return self.path.exists()
    
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
    
    def copy(self) -> "TIPCatalog":
        """
        Return an in-memory deep copy of this TIPCatalog instance,

        """

        new_instance = TIPCatalog(
            path=self.path,
            catalog_type=self.catalog_type,
            info=Info.from_dict(self.info.to_dict()),
            load=False
        )

        # Manually copy loaded data and header
        new_instance.data = None if self.data is None else self.data.copy()
        new_instance.target_img = None if self.target_img is None else ScienceImage(self.target_img.path, telinfo=self.target_img.telinfo, load=True)

        return new_instance
    
    def find_corresponding_fits(self) -> Optional[Path]:
        """
        Find the corresponding .fits file for a given .cat file.
        Assumes the .fits filename is the prefix of the .cat file, ending at '.fits'.
        Searches both the current and parent directory.
        """
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
        
    def load_target_img(self, target_img: Union[ScienceImage, ReferenceImage] = None):
        # Load the catalog from a target image path.
        if target_img is None:
            target_path = self.find_corresponding_fits()
            if target_path is None:
                print(f"[ERROR] No corresponding FITS file found for {self.path}")
                return False
            target_img = ScienceImage(target_path, telinfo = self.estimate_telinfo(target_path), load = True)
        
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

    def save_info(self, verbose = False):
        """ Save processing info to a JSON file """
        with open(self.savepath.infopath, 'w') as f:
            json.dump(self.info.to_dict(), f, indent=4)
        self.print(f"Saved: {self.savepath.infopath}", verbose)
    
    def load_info(self, verbose = False):
        """ Load processing info from a JSON file """
        if not self.savepath.infopath.exists():
            self.print(f"Info file does not exist: {self.savepath.infopath}", verbose)
            return self.info
        
        with open(self.savepath.infopath, 'r') as f:
            data = json.load(f)
        
        self.info = Info.from_dict(data)
        if self.info.target_img is not None:
            target_path = Path(self.info.target_img)
            if not target_path.exists():
                self.print(f"Target image does not exist: {target_path}", verbose)
                return self.info
            
            self.target_img = ScienceImage(target_path, telinfo=self.estimate_telinfo(target_path), load=True)
            self.is_loaded = True
        self.print(f"Loaded: {self.savepath.infopath}", verbose)
    
    # def search_sources(self, 
    #                    x: Union[float, list, np.ndarray],
    #                    y: Union[float, list, np.ndarray],
    #                    unit='coord',
    #                    matching_radius: float = 5.0):
    #     """
    #     Match input positions to a catalog using cKDTree (fast) for both pixel and sky coordinates.
    #     Returns matched catalog sorted by separation, along with separations (arcsec) and indices.
    #     """
    #     x = np.atleast_1d(x)
    #     y = np.atleast_1d(y)
    #     target_catalog = self.data

    #     if unit == 'pixel':
    #         catalog_coords = np.vstack((target_catalog['X_IMAGE'], target_catalog['Y_IMAGE'])).T
    #         input_coords = np.vstack((x, y)).T

    #         tree = cKDTree(catalog_coords)
    #         sep, idx = tree.query(input_coords, distance_upper_bound=matching_radius)

    #         valid = sep != np.inf
    #         sep = sep[valid]
    #         idx = idx[valid]
    #         matched_catalog = target_catalog[idx]

    #         # Sort by separation
    #         sort_idx = np.argsort(sep)
    #         return matched_catalog[sort_idx], sep[sort_idx], idx[sort_idx]

    #     elif unit == 'coord':
    #         cat_sky = SkyCoord(ra=target_catalog['X_WORLD'], dec=target_catalog['Y_WORLD'], unit='deg')
    #         input_sky = SkyCoord(ra=x, dec=y, unit='deg')

    #         cat_xyz = np.vstack(cat_sky.cartesian.xyz).T
    #         input_xyz = np.vstack(input_sky.cartesian.xyz).T

    #         tree = cKDTree(cat_xyz)
    #         matching_radius_rad = (matching_radius / 3600.0) * (np.pi / 180)
    #         sep, idx = tree.query(input_xyz, distance_upper_bound=matching_radius_rad)

    #         valid = sep != np.inf
    #         sep = sep[valid]
    #         idx = idx[valid]
    #         matched_catalog = target_catalog[idx]
    #         sep_arcsec = np.rad2deg(sep) * 3600

    #         # Sort by separation
    #         sort_idx = np.argsort(sep_arcsec)
    #         return matched_catalog[sort_idx], sep_arcsec[sort_idx], idx[sort_idx]

    #     else:
    #         raise ValueError("unit must be either 'pixel' or 'coord'")
        
    def select_sources(self,
                       x: Union[float, list, np.ndarray],
                       y: Union[float, list, np.ndarray],
                       unit='coord',
                       matching_radius: float = 5.0):
        """
        Match input positions to a catalog using cKDTree (fast) for both pixel and sky coordinates.
        Returns matched catalog sorted by separation, along with separations (arcsec) and indices.
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
        
        
    def apply_mask(self, 
                   target_ivpmask: Mask,
                   x_key: str = 'X_IMAGE',
                   y_key: str = 'Y_IMAGE'):
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
        # Convert X_WORLD and Y_WORLD to pixel coordinates and save to a stamp catalog.
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
        with open(self.savepath.stamppath, "w") as f:
            for x, y in zip(x_pix, y_pix):
                f.write(f"{round(x,3)} {round(y,3)} \n")
        return self.savepath.stamppath
    
    def to_region(self, reg_size: float = 6.0, shape : str = 'circle'):

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
    
    def write(self, format = 'ascii'):
        """Write MaskImage data to FITS file."""
        if self.data is None:
            raise ValueError("Cannot save MaskImage: data is not registered.")
        os.makedirs(self.savepath.savedir, exist_ok=True)

        # Write to disk
        self.data.write(self.savepath.savepath, format=format, overwrite=True)
        self.save_info()
        

    def remove(self, 
               remove_all: bool = True, 
               skip_exts: list =  ['.png', '.cat'],
               verbose: bool = False) -> dict:
        """
        Remove files associated with this ScienceImage.

        Parameters:
        - remove_all (bool): If True, remove all related files (FITS, .status, .info, .mask, etc.)
                            If False, remove only the FITS file.
        - verbose (bool): Print/log removed files.

        Returns:
        - dict: {filename (str): success (bool)} for each attempted removal
        """
        removed = {}

        def try_remove(p: Union[str, Path]):
            p = Path(p)
            if p.exists():
                try:
                    p.unlink()
                    self.print(f"[REMOVE] {p}", verbose)
                    return True
                except Exception as e:
                    self.print(f"[FAILED] {p} - {e}", verbse)
                    return False
            return False

        # Remove the main FITS file
        if self.path and self.path.is_file():
            removed[str(self.path)] = try_remove(self.path)

        # Remove other savepath-related files (excluding dirs)
        if remove_all:
            for attr in vars(self.savepath).values():
                if isinstance(attr, Path) and attr != self.path and attr.is_file():
                    if attr.suffix in skip_exts:
                        self.print(f"[SKIP] {attr} (skipped due to extension)", verbose)
                        continue
                    removed[str(attr)] = try_remove(attr)

        return removed   
     
    def show_source(self,
                target_ra: float,
                target_dec: float,
                downsample: int = 4,
                zoom_radius_pixel: float = 50,
                matching_radius_arcsec: float = 3.0):
        """
        Show two-panel view:
        - Left: full image with a single source marked (red).
        - Right: zoomed-in view with the requested position (red) and matched source (blue, if found).
        """
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        import matplotlib.pyplot as plt
        import numpy as np

        # Load image if not yet loaded
        if self.target_img is None:
            load_result = self.load_target_img(target_img=None)

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


#%%
if __name__ == "__main__":
    catalog_path = '/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/NGC6121/7DT01/m400/calib_7DT01_NGC6121_20240722_010055_m400_100.com.fits.cat'
    self = TIPCatalog(path=catalog_path, catalog_type ='all', load = True)

    #sources =  self.data[(self.data['X_IMAGE'] < 4600) & (self.data['Y_IMAGE'] < 3100) & (self.data['X_IMAGE'] > 4400) & (self.data['Y_IMAGE'] > 2900)]
    # source = sources[4]
    # target_ra = source['X_WORLD']
    # target_dec = source['Y_WORLD']
    # matching_radius_arcsec = 3
    #self.show_source(target_ra=target_ra, target_dec=target_dec, downsample=4, zoom_radius_pixel=50)
    #target_img = ScienceImage(self.target_path, telinfo = self.estimate_telinfo(self.target_path), load = False)
    #tbl = self.search_sources(x = 233.890152, y = -67.523614423, unit = 'coord')
    #self.load_from_target_path()
    #self.save_info()
    #A = self.find_corresponding_fits()
    
# %%
