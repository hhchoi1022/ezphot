
# %%
from astropy.io import fits
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table
from astropy.table import Row
from astropy.wcs import WCS
from astropy.io.fits import Header
import astroscrappy as cr
from typing import List, Union, Optional, Tuple
from pathlib import Path

from tqdm import tqdm
import os
import numpy as np
from astropy.table import unique
import inspect
import subprocess
import re
import warnings
import json 
from tippy.configuration import TIPConfig

# Suppress all warnings
warnings.filterwarnings('ignore')


import signal
import functools

class TimeoutError(Exception):
    pass

class ActionFailedError(Exception):
    pass

def timeout(seconds=10, error_message="Function call timed out"):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set the timeout signal
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm after the function completes
                signal.alarm(0)
            return result
        return wrapper
    return decorator

class PhotometryHelper(TIPConfig):

    def __init__(self):
        super().__init__()
        
    @property
    def configpath(self):
        return Path(self.path_global)
    
    @property
    def scamppath(self):
        return Path(self.config['SCAMP_CONFIGDIR'])
    
    @property  
    def swarppath(self):
        return Path(self.config['SWARP_CONFIGDIR'])
    
    @property
    def sexpath(self):
        return Path(self.config['SEX_CONFIGDIR'])
    
    @property
    def psfexpath(self):
        return Path(self.config['PSFEX_CONFIGDIR'])
        
    def __repr__(self):
        methods = [f'PhotometryHelper.{name}()\n' for name, method in inspect.getmembers(
            PhotometryHelper, predicate=inspect.isfunction) if not name.startswith('_')]
        txt = '[Methods]\n'+''.join(methods)
        return txt
    
    def print(self, string, do_print : bool = False):
        print(string) if do_print else None
        
    # Load information

    def get_imginfo(self, 
                    filelist: Union[List[str], List[Path]],
                    keywords: Optional[List[str]] = None,
                    normalize_key: bool = True) -> Table:
        """
        Collects FITS image metadata from all given files.

        Parameters
        ----------
        1. filelist : (list) List of FITS file paths.
        2. normalize_key : (bool) If True, normalize FITS header keywords based on required_key_variants.

        Returns
        -------
        1. all_coll : astropy.table.Table
                    Combined metadata from all FITS files.

        Notes
        -----
        - Ensures consistency in string formatting for table columns.
        - Normalizes keywords if `normalize_key=True`.
        - Handles missing FITS files gracefully.
        """
        from astropy.table import Table, vstack
        from ccdproc import ImageFileCollection
        import os
        filelist = list(Path(file) for file in filelist)

        # Get unique parent directories
        directories = list(set(file.parent for file in filelist))
        
        # Initialize an empty table
        all_coll = Table()

        # Iterate through directories and collect FITS file metadata
        for directory in directories:
            coll = ImageFileCollection(location=directory, glob_include='*.fits')
            if len(coll.files) > 0:
                print(f"Loaded {len(coll.files)} FITS files from {directory}")

                # Get metadata table
                summary = coll.summary.copy() if keywords is None else coll.summary[keywords]

                # Convert "file" column to absolute paths
                summary['file'] = [directory / f for f in summary['file']]

                if normalize_key:
                    # Normalize header keys
                    new_column_names = {}
                    seen_keys = set()
                    for colname in summary.colnames:
                        normalized_key = self.normalize_required_keys(colname)
                        if normalized_key:
                            #if normalized_key in seen_keys:
                            #    normalized_key += f"_{colname}"  # Append original name to avoid collision
                            seen_keys.add(normalized_key)
                            new_column_names[colname] = normalized_key

                    # Rename columns to normalized keys
                    new_column_names = {key: value for key, value in new_column_names.items() if value not in summary.colnames}
                    summary.rename_columns(list(new_column_names.keys()), list(new_column_names.values()))

                    # Keep only valid normalized keys
                    valid_keys = set(self.required_key_variants.keys()) | {'file'}
                    summary = summary[[col for col in summary.colnames if col in valid_keys]]

                # Ensure string consistency in table columns
                for colname in summary.colnames:
                    col_dtype = summary[colname].dtype
                    if col_dtype.kind in ('O', 'U', 'S'):  # Object, Unicode, or String types
                        summary[colname] = summary[colname].astype(str)
                    elif col_dtype.kind in ('i', 'f'):  # Integer or Float types
                        summary[colname] = summary[colname].astype(str)  # Convert to string for consistency
                        summary[colname].fill_value = ''  # Ensure NaN values are handled

                # Stack tables
                all_coll = vstack([all_coll, summary], metadata_conflicts='silent') if len(all_coll) else summary

            else:
                print(f"Warning: No FITS files found in {directory}")

        # Filter to ensure only filelist rows are returned
        filelist_inputted = filelist
        filelist_queried = np.array([Path(f) for f in all_coll['file']])
        all_coll = all_coll[np.isin(filelist_queried, filelist_inputted)]

        # Sort the final result to match the original filelist order
        file_order = {f: i for i, f in enumerate(filelist)}
        sort_idx = np.argsort([file_order.get(f, float('inf')) for f in all_coll['file']])
        all_coll = all_coll[sort_idx]

        # Check final count of combined FITS files
        print(f"Total FITS files combined: {len(all_coll)}")
        return all_coll
    
    def normalize_required_keys(self, key: str):
        # Iterate through the dictionary to find a match
        for canonical_key, variants in self.required_key_variants.items():
            if key.lower() in variants:
                return canonical_key
        return None
    
    @property
    def required_key_variants(self):
        # Define key variants, if a word is duplicated in the same variant, posit the word with the highest priority first
        required_key_variants_lower = {
            'altitude': ['alt', 'altitude'],
            'azimuth': ['az', 'azimuth'],
            'gain': ['gain'],
            'ccd-temp': ['ccdtemp', 'ccd-temp'],
            'filter': ['filter', 'filtname', 'band'],
            'imgtype': ['imgtype', 'imagetyp', 'imgtyp'],
            'exptime': ['exptime', 'exposure'],
            'obsdate': ['date-obs', 'obsdate', 'utcdate'],
            'locdate': ['date-loc', 'date-ltc', 'locdate', 'ltcdate'],
            'jd' : ['jd'],
            'mjd' : ['mjd'],
            'telescop' : ['telescop', 'telname'],
            'binning': ['binning', 'xbinning'],
            'object': ['object', 'objname', 'target', 'tarname'],
            'objctid': ['objctid', 'objid', 'id'],
            'obsmode': ['obsmode', 'mode'],
            'specmode': ['specmode'],
            'ntelescop': ['ntelescop', 'ntel'],
            'note': ['note'],
        }
        # Sort each list in the dictionary by string length (descending order)
        sorted_required_key_variants = {
            key: sorted(variants, key=len, reverse=True)
            for key, variants in required_key_variants_lower.items()
        }
        return sorted_required_key_variants
    
    def get_sexconfigpath(self, 
                          telescope: str,
                          ccd: Optional[str] = None,
                          binning: int = 1,
                          readoutmode: Optional[str] = None,
                          for_scamp: bool = False,
                          for_psfex: bool = False) -> Path:

        file_key = f'{telescope.upper()}'
        if ccd:
            file_key += f'_{ccd.upper()}'
        if readoutmode:
            file_key += f'_{readoutmode.upper()}'
        if binning:
            file_key += f'_{binning}x{binning}'
        if for_scamp:
            file_key += '.scamp'
        if for_psfex:
            file_key += '.psfex'
        file_key += '.sexconfig'
        file_path = self.configpath / 'sextractor' / file_key
        if file_path.exists():
            return file_path
        else:
            raise FileNotFoundError(f'{file_key} not found: {file_path}')

    def get_scampconfigpath(self) -> Path:
        file_path = self.configpath / 'scamp' / 'default.scampconfig'
        if file_path.exists():
            return file_path
        else:
            raise FileNotFoundError(f'default.scampconfig not found :{file_path}')
    
    def get_psfexconfigpath(self) -> Path:
        file_path = self.configpath / 'psfex' / 'default.psfexconfig'
        if file_path.exists():
            return file_path
        else:
            raise FileNotFoundError(f'default.psfexconfig not found :{file_path}')

    def get_swarpconfigpath(self,
                            telescope: str,
                            ccd: Optional[str] = None,
                            binning: int = 1,
                            readoutmode: Optional[str] = None) -> Path:
        file_key = f'{telescope.upper()}'
        if ccd:
            file_key += f'_{ccd.upper()}'
        if readoutmode:
            file_key += f'_{readoutmode.upper()}'
        if binning:
            file_key += f'_{binning}x{binning}'
        file_key += '.swarpconfig'
        file_path = self.configpath / 'swarp' / file_key
        if file_path.exists():
            return file_path
        else:
            raise FileNotFoundError(f'{file_key} not found :{file_path}')

    def get_telinfo(self,
                    telescope: Optional[str] = None, 
                    ccd: Optional[str] = None, 
                    readoutmode: Optional[str] = None, 
                    binning: Optional[int] = None, 
                    key_observatory: str = 'obs', 
                    key_ccd: str = 'ccd', 
                    key_mode: str = 'mode', 
                    key_binning: str = 'binning',
                    obsinfo_file: Optional[Union[str, Path]] = None) -> Row:
        """
        Retrieves telescope and CCD information from an observatory information file.

        Parameters
        ----------
        telescope : str, optional
            Name of the telescope.
        ccd : str, optional
            CCD name.
        readoutmode : str, optional
            Readout mode [High, Merge, Low].
        binning : int, optional
            Binning factor.
        key_observatory : str, optional
            Column name for observatory/telescope.
        key_ccd : str, optional
            Column name for CCD type.
        key_mode : str, optional
            Column name for readout mode.
        key_binning : str, optional
            Column name for binning factor.
        obsinfo_file : str or Path, optional
            Path to the observatory information file.

        Returns
        -------
        obsinfo : astropy.table.Row
            Matched observatory/CCD information.

        Raises
        ------
        AttributeError
            If no matching telescope/CCD is found.
        """

        # Load observatory info file
        if obsinfo_file is None:
            obsinfo_file = self.configpath / 'CCD.dat'

        all_obsinfo = ascii.read(obsinfo_file, format='fixed_width')

        def filter_by_column(data, column, value):
            """Filters a table by a specific column value."""
            if value is None or column not in data.colnames:
                return data
            return data[data[column] == value]

        def prompt_choice(options, message):
            """Prompts user to select from multiple options if interactive."""
            if not options:
                raise AttributeError(f"No available options for {message}.")
            print(f"{message}: {options}")
            return input("Enter choice: ").strip()

        # Select telescope if not provided
        if telescope is None:
            telescope = prompt_choice(set(all_obsinfo[key_observatory]), "Choose the Telescope")

        # Validate telescope existence
        if telescope not in all_obsinfo[key_observatory]:
            raise AttributeError(f"Telescope {telescope} information not found. Available: {set(all_obsinfo[key_observatory])}")

        # Filter for the selected telescope
        obs_info = filter_by_column(all_obsinfo, key_observatory, telescope)
        if len(obs_info) == 0:
            raise AttributeError(f"No data found for telescope: {telescope}")

        # Select CCD if not provided and multiple options exist
        if ccd is None and len(set(obs_info[key_ccd])) > 1:
            ccd = prompt_choice(set(obs_info[key_ccd]), "Multiple CCDs found. Choose one")
        obs_info = filter_by_column(obs_info, key_ccd, ccd)

        # Select readout mode if not provided and multiple options exist
        if readoutmode is None and len(set(obs_info[key_mode])) > 1:
            readoutmode = prompt_choice(set(obs_info[key_mode]), "Multiple modes found. Choose one")
        obs_info = filter_by_column(obs_info, key_mode, readoutmode)

        # Select binning if not provided and multiple options exist
        if key_binning in obs_info.colnames and binning is None and len(set(obs_info[key_binning])) > 1:
            binning = prompt_choice(set(obs_info[key_binning]), "Multiple binning values found. Choose one")
        if binning is not None:
            obs_info = filter_by_column(obs_info, key_binning, int(binning))

        # Ensure only one row remains
        if len(obs_info) == 1:
            return obs_info[0]

        raise AttributeError(f"No matching CCD info for {telescope}. Available CCDs: {list(set(all_obsinfo[key_ccd]))}")

    def load_config(self, 
                    config_path: Union[str, Path]) -> dict:
        """ Load sextractor, swarp, scamp, psfex configuration file

        Args:
            config_path (str): absolute path of the configuration file

        Returns:
            dict: dictionary of the configuration file
        """
        config_dict = {}

        with open(config_path, 'r') as file:
            for line in file:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Split the line into key and value
                key_value = line.split(maxsplit=1)
                if len(key_value) == 2:
                    key, value = key_value
                    # Remove inline comments
                    value = value.split('#', 1)[0].strip()
                    # Attempt to convert value to appropriate type
                    try:
                        # Handle lists
                        if ',' in value:
                            value = [float(v) if '.' in v else int(v)
                                    for v in value.split(',')]
                        else:
                            # Convert to float if possible
                            value = float(value) if '.' in value else int(value)
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
                    config_dict[key] = value
        return config_dict

    # Calculation
    
    def to_skycoord(self, 
                    ra: Union[float, str], 
                    dec: Union[float, str], 
                    frame: str = 'icrs') -> SkyCoord:
        """
        Converts RA and Dec to an Astropy SkyCoord object.

        Parameters
        ----------
        ra : str or float
            Right ascension in various formats (see Notes).
        dec : str or float
            Declination in various formats (see Notes).
        frame : str, optional
            Reference frame for the coordinates, default is 'icrs'.

        Returns
        -------
        skycoord : astropy.coordinates.SkyCoord
            The corresponding SkyCoord object.

        Notes
        -----
        Supported RA/Dec formats:
        1. "15h32m10s", "50d15m01s"
        2. "15 32 10", "50 15 01"
        3. "15:32:10", "50:15:01"
        4. 230.8875, 50.5369 (Decimal degrees)
        """

        from astropy.coordinates import SkyCoord
        import astropy.units as u

        ra, dec = str(ra).strip(), str(dec).strip()

        if any(symbol in ra for symbol in [':', 'h', ' ']) and any(symbol in dec for symbol in [':', 'd', ' '] ):
            units = (u.hourangle, u.deg)
        else:
            units = (u.deg, u.deg)

        return SkyCoord(ra=ra, dec=dec, unit=units, frame=frame)


    def bn_median(self, masked_array: np.ma.MaskedArray, axis: Optional[int] = None) -> np.ndarray:
        """

        parameters
        ----------
        masked_array : `numpy.ma.masked_array`
                        Array of which to find the median.
        axis : optional, int 
                        Axis along which to perform the median. Default is to find the median of
                        the flattened array.

        returns
        ----------

        notes
        ----------
        Source code from Gregory S.H. Paek
        Perform fast median on masked array
        ----------
        """

        import numpy as np
        import bottleneck as bn
        data = masked_array.filled(fill_value=np.NaN)
        med = bn.nanmedian(data, axis=axis)
        # construct a masked array result, setting the mask from any NaN entries
        return np.ma.array(med, mask=np.isnan(med))

    # Table operation

    def cross_match(self, 
                    obj_catalog: SkyCoord, 
                    sky_catalog: SkyCoord, 
                    max_distance_second: float = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        parameters
        ----------
        1. obj_catalog : SkyCoord
                        source coordinates to be matched 
        2. sky_catalog : SkyCoord
                        coordinates of reference stars
        3. max_distance_second : float
                        separation distance in seconds which indicates tolerance for matching (5)

        returns 
        -------
        1. matched_object_idx : np.array
                        matched object catalog indices 
        2. matched_catalog_idx : np.array
                        matched sky catalog indices
        3. no_matched_object_idx : np.array
                        not matched object catalog indices

        notes 
        -----
        To extract matched objects from each catalog, 
        USE : obj_catalog[matched_object_idx] & sky_catalog[matched_catalog_idx]
        -----
        '''
        from astropy.coordinates import match_coordinates_sky
        closest_ids, closest_dists, closest_dist3d = match_coordinates_sky(obj_catalog, sky_catalog)
        max_distance = max_distance_second/3600
        matched_object_idx = []
        matched_catalog_idx = []
        no_matched_object_idx = []
        for i in range(len(closest_dists)):
            if closest_dists.value[i] < max_distance:
                matched_object_idx.append(i)
                matched_catalog_idx.append(closest_ids[i])
            else:
                no_matched_object_idx.append(i)
        return matched_object_idx, matched_catalog_idx, no_matched_object_idx

    def group_table(self, tbl: Table, key: str, tolerance: float = 0.1) -> Table:
        '''
        parameters
        ----------
        group components in the {table} with the difference of the {key} smaller than the {tolerance}

        returns 
        -------
        1. groupped tables

        notes 
        -----

        -----
        '''
        from astropy.table import Table
        from astropy.table import vstack
        i = 0
        table = tbl.copy()
        table['group'] = 0
        groupped_tbl = Table()
        while len(table) >= 1:
            group_idx = (np.abs(table[0][key] - table[key]) < tolerance)
            group_tbl = table[group_idx]
            group_tbl['group'] = i
            remove_idx = np.where(group_idx == True)
            table.remove_rows(remove_idx)
            groupped_tbl = vstack([group_tbl, groupped_tbl])
            i += 1

        return groupped_tbl

    def match_table(self, 
                    tbl1: Table, 
                    tbl2: Table, 
                    key: str, 
                    tolerance: float = 0.01) -> Table:
        '''
        parameters
        ----------
        {two tables} to combine with the difference of the {key} smaller than the {tolerance}

        returns 
        -------
        1. combined table
        2. phase

        notes 
        -----
        Combined table have both columns of original tables. 
        They are horizontally combined in the order of tbl1, tbl2
        -----
        '''

        from astropy.table import vstack, hstack

        matched_tbl = Table()
        for obs in tbl1:
            ol_idx = (np.abs(obs[key] - tbl2[key]) < tolerance)
            if True in ol_idx:
                closest_idx = np.argmin(np.abs(obs[key]-tbl2[key]))
                compare_tbl = tbl2[closest_idx]
                # join(obs, compare_tbl, keys = 'observatory', join_type = 'outer')
                compare_tbl = hstack([obs, compare_tbl])
                matched_tbl = vstack([matched_tbl, compare_tbl])

        return matched_tbl

    def binning_table(self, 
                      tbl: Table, 
                      key: str, 
                      tolerance: float = 0.01) -> Table:
        '''
        Parameters
        ----------
        tbl : Astropy.Table
                The input table to be binned.
        key : str
                The column name to apply the binning on.
        tolerance : float, optional
                The tolerance within which to bin the rows. Default is 0.01.

        Returns
        -------
        Astropy.Table
                The binned table with duplicates removed based on the specified tolerance.
        '''
        import pandas as pd
        table = tbl.to_pandas()
        # Sort the table by the key for efficient processing
        table = table.sort_values(by=key).reset_index(drop=True)

        binned_rows = []
        start_idx = 0

        while start_idx < len(table):
            end_idx = start_idx
            while (end_idx < len(table)) and (table[key].iloc[end_idx] - table[key].iloc[start_idx] < tolerance):
                end_idx += 1

            compare_table = table.iloc[start_idx:end_idx]

            # Aggregate the values within the tolerance range
            row = []
            for col in table.columns:
                if pd.api.types.is_numeric_dtype(table[col]):
                    result_val = round(compare_table[col].mean(), 4)
                else:
                    result_val = compare_table[col].iloc[0]
                row.append(result_val)

            binned_rows.append(row)
            start_idx = end_idx

        binned_table = pd.DataFrame(binned_rows, columns=tbl.columns)
        binned_tbl = Table().from_pandas(binned_table)
        return binned_tbl

    def remove_rows_table(self, 
                          tbl: Table, 
                          column_key: str, 
                          remove_keys: Union[str, List[str]]) -> Table:
        '''
        Parameters
        ----------
        tbl : astropy.table.Table
                The input table from which rows need to be removed.
        column_key : str
                The column name based on which rows will be removed.
        remove_keys : str or list
                The value or list of values to be removed from the specified column in the table.

        Returns
        -------
        astropy.table.Table
                The table with specified rows removed.

        Notes
        -----
        This function removes rows from the input table where the values in the specified column
        match any of the values in `remove_keys`. `remove_keys` can be a single value (string)
        or a list of values. The function modifies the table in place and returns the modified table.
        -----
        '''
        if isinstance(remove_keys, str):
            remove_mask = tbl[column_key] == remove_keys
            remove_idx = np.where(remove_mask == True)
            tbl.remove_rows(remove_idx)
        else:
            for remove_key in remove_keys:
                remove_mask = tbl[column_key] == remove_key
                remove_idx = np.where(remove_mask == True)
                tbl.remove_rows(remove_idx)
        return tbl

    # Image processing
    def calculate_rotang(self, 
                         target_img: Union[str, Path], 
                         update_header: bool = False, 
                         print_output: bool = False):
        from astropy.io import fits
        from astropy.wcs import WCS
        import numpy as np
        # Load the FITS file with the astrometry solution
        target_img = Path(target_img)
        hdul = fits.open(target_img)
        wcs = WCS(hdul[0].header)

        # Define a pixel at the center of the image
        ny, nx = hdul[0].data.shape
        center_pixel = [nx // 2, ny // 2]

        # Get the pixel coordinates offset along the y-axis (to simulate up direction)
        north_pixel = [center_pixel[0], center_pixel[1] + 1]

        # Convert these pixel positions to celestial coordinates (RA, Dec)
        center_coord = wcs.pixel_to_world(*center_pixel)
        north_coord = wcs.pixel_to_world(*north_pixel)

        # Calculate the position angle (PA) between the north pixel and the center pixel
        delta_ra = np.deg2rad(north_coord.ra.deg - center_coord.ra.deg) * np.cos(np.deg2rad(center_coord.dec.deg))
        delta_dec = np.deg2rad(north_coord.dec.deg - center_coord.dec.deg)

        pa_radians = np.arctan2(delta_ra, delta_dec)
        pa_degrees = np.rad2deg(pa_radians)

        # Adjust the angle to get the position angle from north
        if pa_degrees < 0:
            pa_degrees += 360
        
        if update_header:
            hdul[0].header['ROTANG'] = pa_degrees
            hdul.writeto(target_img, overwrite=True)
        hdul.close()
        self.print(f"Camera rotation angle (Position Angle) toward North: {pa_degrees:.2f} degrees", print_output)
        
    def img_cutout(self, 
                   target_img: Union[str, Path, np.ndarray], 
                   target_header: Optional[Header] = None,  # When target_img is np.ndarray
                   output_path: Optional[str] = None,
                   xsize: float = 0.9, 
                   ysize: float = 0.9,
                   xcenter: Optional[float] = None, 
                   ycenter: Optional[float] = None, 
                   print_output: bool = True):
        '''
        parameters
        ----------
        1. target_img : str or np.ndarray
                        (str) absolute path of the target image
                        (np.ndarray) image data
        2. target_header : astropy.io.fits.Header (optional)
                           Required if target_img is np.ndarray
        3. xsize, ysize : float or int
                        (float) ratio of the cut image (0.9 by default)
                        (int) size of the cut image in pixels
        4. xcenter, ycenter : optional, int or (float or str)
                        (int) pixel coordinate of the center
                        (float or str) RA/Dec coordinate of the center
        5. print_output : bool
                        If True, prints progress messages
                        
        returns 
        -------
        1. If target_img is str:
            output_path : str
                {abspath} of the cutout image
        2. If target_img is np.ndarray:
            (cutouted.data, cutouted_header) : tuple
                (np.ndarray, astropy.io.fits.Header)
        '''
        from astropy.wcs import WCS
        from astropy.nddata import Cutout2D
        from astropy.io import fits
        import os
        
        self.print('Start image cutout... \n', print_output)
        
        if isinstance(target_img, (str, Path)):
            target_img = Path(target_img)
            
            if not target_img.is_file():
                raise FileNotFoundError(f"File {target_img} does not exist.")
            
            hdul = fits.open(target_img)
            hdu = hdul[0]
            target_data = hdu.data
            wcs = WCS(hdu.header)

        elif isinstance(target_img, np.ndarray):
            # Input is an image array
            if target_header is None:
                raise ValueError("Header must be provided when target_img is a numpy array.")
            target_data = target_img
            wcs = WCS(target_header)
        else:
            raise TypeError("target_img must be either a str or an np.ndarray.")
        
        # Calculate cutout size
        size = (int(ysize * target_data.shape[0]), int(xsize * target_data.shape[1])) if (xsize < 1 and ysize < 1) else (xsize, ysize)
        
        # Determine the cutout center
        if xcenter is None or ycenter is None:
            xcenter, ycenter = target_data.shape[1] // 2, target_data.shape[0] // 2
        
        # Perform the cutout
        if not isinstance(xcenter, int) or not isinstance(ycenter, int):
            center_coords = self.to_skycoord(xcenter, ycenter)
            cutouted = Cutout2D(data=target_data, position=center_coords, size=size, wcs=wcs)
        else:
            cutouted = Cutout2D(data=target_data, position=(xcenter, ycenter), size=size, wcs=wcs)

        if isinstance(target_img, Path):  # If input was a file path
            # Save the cutout image as a FITS file
            cutouted_hdu = fits.PrimaryHDU(data=cutouted.data, header=cutouted.wcs.to_header())
            cutouted_hdu.header['CUTOUT'] = (True, 'Image has been cut out.')
            cutouted_hdu.header['CUTOTIME'] = (Time.now().isot, 'Time of cutout operation.')
            cutouted_hdu.header['CUTOFILE'] = (str(target_img), 'Original file path before cutout')

            if not output_path:
                output_path = target_img.parent / f'cutout_{target_img.name}'
            cutouted_hdu.writeto(output_path, overwrite=True)

            hdul.close()
            self.print(f'Image cutout complete: {output_path}\n', print_output)
            return str(output_path)
        else:  # If input was a NumPy array
            cutouted_header = target_header.copy()
            cutouted_header.update(cutouted.wcs.to_header())
            cutouted_header['NAXIS1'] = cutouted.data.shape[1]
            cutouted_header['NAXIS2'] = cutouted.data.shape[0]
            cutouted_header['CUTOUT'] = (True, 'Image has been cut out.')
            cutouted_header['CUTOTIME'] = (Time.now().isot, 'Time of cutout operation.')
            cutouted_header['CUTOFILE'] = ('Array input', 'Original data was an array')
            self.print('Image cutout complete \n', print_output)
            return cutouted.data, cutouted_header

    def img_astroalign(self, 
                       target_img: Union[str, Path, np.ndarray], 
                       reference_img: Union[str, Path, np.ndarray], 
                       target_header: Optional[Header] = None, 
                       reference_header: Optional[Header] = None, 
                       output_path: Optional[str] = None,
                       detection_sigma: float = 5, 
                       print_output: bool = True):

        """
        WARNING: Astroalign fails when the image size is too large and distortion exists in the images.
        parameters
        ----------
        1. target_img : str or np.ndarray
                        (str) Absolute path of the target image 
                        (np.ndarray) Image data
        2. reference_img : str or np.ndarray
                        (str) Absolute path of the reference image
                        (np.ndarray) Image data
        3. target_header : astropy.io.fits.Header (optional)
                        Required if target_img is np.ndarray
        4. reference_header : astropy.io.fits.Header (optional)
                        Required if reference_img is np.ndarray
        5. detection_sigma : float
                        Detection threshold for astroalign (default: 5)
        6. print_output : bool
                        If True, prints progress messages (default: True)

        returns
        ----------
        1. If target_img is str:
            output_path : str
                Absolute path of the aligned image
        2. If target_img is np.ndarray:
            (aligned_data, aligned_header) : tuple
                (np.ndarray, astropy.io.fits.Header)
        """

        import astroalign as aa
        from ccdproc import CCDData
        from astropy.wcs import WCS
        from astropy.io import fits
        import os

        self.print('Start image alignment... \n', print_output)

        # Convert paths
        if isinstance(target_img, (str, Path)):
            target_img = Path(target_img)
            if not target_img.is_file():
                raise FileNotFoundError(f"File {target_img} does not exist.")
            target_hdul = fits.open(target_img)
            target_data = target_hdul[0].data
            target_header = target_hdul[0].header
            target_hdul.close()
        elif isinstance(target_img, np.ndarray):
            # Input is an image array
            if target_header is None:
                raise ValueError("target_header must be provided when target_img is a numpy array.")
            target_data = target_img
            target_header = target_header
        else:
            raise TypeError("target_img must be either a str or an np.ndarray.")

        if isinstance(reference_img, (str, Path)):
            reference_img = Path(reference_img)
            if not reference_img.is_file():
                raise FileNotFoundError(f"File {reference_img} does not exist.")
            reference_hdul = fits.open(reference_img)
            reference_data = reference_hdul[0].data
            reference_header = reference_hdul[0].header
            reference_hdul.close()
        elif isinstance(reference_img, np.ndarray):
            if reference_header is None:
                raise ValueError("reference_header must be provided when reference_img is a numpy array.")
            reference_data = reference_img
        else:
            raise TypeError("reference_img must be either a str, Path, or an np.ndarray.")

        # Prepare WCS and header update
        reference_wcs = WCS(reference_header)
        wcs_hdr = reference_wcs.to_header(relax=True)
        for key in ['DATE-OBS', 'MJD-OBS', 'RADESYS', 'EQUINOX']:
            wcs_hdr.remove(key, ignore_missing=True)
        
        target_header.update(wcs_hdr)
        target_data = np.array(target_data, dtype=target_data.dtype.newbyteorder('<'))
        reference_data = np.array(reference_data, dtype=reference_data.dtype.newbyteorder('<'))


        try:
            # Perform image alignment using astroalign
            aligned_data, footprint = aa.register(target_data, reference_data, 
                                                fill_value=0, 
                                                detection_sigma=detection_sigma, 
                                                max_control_points=30,
                                                min_area=10)

            if isinstance(target_img, Path):
                # Save the aligned image as a FITS file
                aligned_target = CCDData(aligned_data, header=target_header, unit='adu')
                aligned_target.header['ALIGN'] = (True, 'Image has been aligned.')
                aligned_target.header['ALIGTIME'] = (Time.now().isot, 'Time of alignment operation.')
                aligned_target.header['ALIGFILE'] = (str(target_img), 'Original file path before alignment')
                aligned_target.header['ALIGREF'] = (str(reference_img), 'Reference image path')

                if not output_path:
                    output_path = target_img.parent / f'align_{target_img.name}'
                os.makedirs(output_path.parent, exist_ok=True)
                fits.writeto(output_path, aligned_target.data, aligned_target.header, overwrite=True)

                self.print('Image alignment complete \n', print_output)
                return str(output_path)
            else:
                # Return the aligned data and header
                aligned_header = target_header.copy()
                aligned_header['NAXIS1'] = aligned_data.shape[1]
                aligned_header['NAXIS2'] = aligned_data.shape[0]
                aligned_header['ALIGN'] = (True, 'Image has been aligned.')
                aligned_header['ALIGTIME'] = (Time.now().isot, 'Time of alignment operation.')
                aligned_header['ALIGFILE'] = ('Array input', 'Original data was an array')
                aligned_header['ALIGREF'] = ('Array input', 'Reference data was an array')

                self.print('Image alignment complete \n', print_output)
                return aligned_data, aligned_header

        except Exception as e:
            self.print('Failed to align the image. Check the image quality and the detection_sigma value.', print_output)
            raise e

    def img_scale(self,
                  target_img: Union[str, Path, np.ndarray],
                  target_header: Optional[Header] = None,
                  output_path: Optional[str] = None,
                  zp_target: Optional[float] = None,
                  zp_reference: float = 25,
                  zp_key: str = 'ZP_AUTO',
                  print_output: bool = True):

        """
        Scale the input image data to a desired reference zeropoint.
        """

        self.print(f"Start image scaling to ZP={zp_reference}...", print_output)

        # Convert target_img to Path if it's a string
        if isinstance(target_img, (str, Path)):
            target_img = Path(target_img)

            if not target_img.is_file():
                raise FileNotFoundError(f"File {target_img} does not exist.")

            # Read data and header
            with fits.open(target_img) as hdul:
                target_data = hdul[0].data
                target_header = hdul[0].header

            # Determine zp_target from the header
            if zp_key in target_header:
                zp_target = float(target_header[zp_key])
            else:
                raise ValueError(
                    f"Zeropoint not found in the FITS header under key '{zp_key}'. "
                    "Please provide zp_target or ensure the header has this keyword."
                )

            # Calculate scale factor
            zp_diff = zp_target - zp_reference
            scaling_factor = 100 ** (-zp_diff / 5)
            self.print(f"Applying scaling factor: {scaling_factor:.6f} "
                    f"(zp_target={zp_target}, zp_reference={zp_reference})", print_output)

            scaled_data = target_data * scaling_factor

            # Update header with new ZP
            target_header[zp_key] = zp_reference
            target_header['ZPSCALE'] = (True, 'Image has been scaled to a new zeropoint.')
            target_header['ZPSCUNIT'] = (zp_key, 'Zeropoint unit for scaling.')
            target_header['ZPSCTIME'] = (Time.now().isot, 'Time of ZP scaling operation.')
            target_header['ZPSCFILE'] = (str(target_img), 'Original file path before scaling')

            # Write scaled data to a new FITS file
            if not output_path:
                output_path = target_img.parent / f"scaled_{target_img.name}"
            os.makedirs(output_path.parent, exist_ok=True)
            fits.writeto(output_path, scaled_data, target_header, overwrite=True)

            self.print(f"Image scaling complete. Output: {output_path}", print_output)
            return str(output_path)

        elif isinstance(target_img, np.ndarray):
            target_data = target_img

            if target_header is not None:
                if zp_key in target_header:
                    zp_target = float(target_header[zp_key])
                else:
                    raise ValueError(
                        f"{zp_key} not found in the provided target_header, "
                        "and 'zp_target' was also not supplied."
                    )

                # Calculate scale factor
                zp_diff = zp_target - zp_reference
                scaling_factor = 100 ** (-zp_diff / 5)
                self.print(f"Applying scaling factor: {scaling_factor:.6f} (zp_target={zp_target}, zp_reference={zp_reference})", print_output)

                scaled_data = target_data * scaling_factor

                # Update the header's ZP
                target_header[zp_key] = zp_reference
                target_header['ZPSCALE'] = (True, 'Image has been scaled to a new zeropoint.')
                target_header['ZPSCUNIT'] = (zp_key, 'Zeropoint unit for scaling.')
                target_header['ZPSCTIME'] = (Time.now().isot, 'Time of ZP scaling operation.')

                self.print("Image scaling complete (returning array and updated header).")
                return scaled_data, target_header

            else:
                if zp_target is None:
                    raise ValueError(
                        "When providing a NumPy array without a header, you must supply zp_target."
                    )

                # Calculate scale factor
                zp_diff = zp_target - zp_reference
                scaling_factor = 100 ** (-zp_diff / 5)
                self.print(f"Applying scaling factor: {scaling_factor:.6f} (zp_target={zp_target}, zp_reference={zp_reference})", print_output)

                scaled_data = target_data * scaling_factor
                self.print("Image scaling complete (returning array only).", print_output)
                return scaled_data

        else:
            raise TypeError("target_img must be either a FITS file path (string, Path) or a NumPy array.")

    def img_convolve(self,
                     target_img: Union[str, Path, np.ndarray],
                     target_header: Optional[Header] = None,
                     output_path: Optional[str] = None,
                     fwhm_target: Optional[float] = None,
                     fwhm_reference: Optional[float] = None,
                     fwhm_key: str = 'PEEING',
                     method: str = 'gaussian',
                     print_output: bool = True):

        """
        Parameters
        ----------
        target_img : str, Path, or np.ndarray
            Path to the FITS file or image data as a NumPy array.
        target_header : astropy.io.fits.Header, optional
            FITS header associated with the image (only when target_img is a NumPy array).
        fwhm_target : float, optional
            FWHM of the target image in pixel scale. If None, will be read from header using fwhm_key.
        fwhm_reference : float
            Desired FWHM after convolution.
        fwhm_key : str
            Header keyword to fetch the FWHM in pixel scale when target_img is a FITS file (default: 'FWHM_AUTO').
        method : str
            Convolution method, currently only supports 'gaussian' (default: 'gaussian').
        print_output : bool
            If True, prints progress messages (default: True).

        Returns
        -------
        str or (np.ndarray, astropy.io.fits.Header) or np.ndarray
            - If target_img is a string or Path, returns the path to the convolved FITS file.
            - If target_img is an array and header is provided, returns (convolved_data, header).
            - If target_img is an array without header, returns convolved_data.
        """
        from pathlib import Path
        import numpy as np
        from astropy.io import fits
        from astropy.convolution import convolve, Gaussian2DKernel
        from astropy.time import Time
        import os

        self.print(f'Start convolution...', print_output)

        # Convert target_img to Path if it's a string
        if isinstance(target_img, (str, Path)):
            target_img = Path(target_img)

            if not target_img.is_file():
                raise FileNotFoundError(f"File {target_img} does not exist.")

            # Load data and header from FITS file
            with fits.open(target_img) as hdul:
                data = hdul[0].data
                header = hdul[0].header

            # If fwhm_target is not provided, try to get it from the header
            if fwhm_key in header:
                fwhm_target = float(header[fwhm_key])
            elif fwhm_target is None:
                raise ValueError(f"FWHM not found in header using key '{fwhm_key}', and 'fwhm_target' is not provided.")

        elif isinstance(target_img, np.ndarray):
            data = target_img

            if target_header is not None:
                header = target_header
                if fwhm_key in header:
                    fwhm_target = float(header[fwhm_key])
                elif fwhm_target is None:
                    raise ValueError(f"{fwhm_key} not found in target_header and 'fwhm_target' is not provided.")
            else:
                header = None
        else:
            raise TypeError("target_img must be either a Path, string (FITS file path), or a NumPy array.")

        self.print(f'Running convolution with the following values = (FWHM_TARGET = {fwhm_target}, FWHM_REFERENCE = {fwhm_reference})', print_output)

        # Calculate the convolution kernel sigma
        sigma_tgt = fwhm_target / 2.355
        sigma_ref = fwhm_reference / 2.355
        sigma_conv = np.sqrt(max(0, sigma_ref**2 - sigma_tgt**2))

        self.print(f"Calculated convolution sigma: {sigma_conv:.6f} (method: {method})", print_output)

        # Create a Gaussian kernel and convolve the image
        if method.lower() == 'gaussian':
            kernel = Gaussian2DKernel(sigma_conv)
            convolved_image = convolve(data, kernel, normalize_kernel=True)
        # Add more convolution methods here
        else:
            raise ValueError(f"Unsupported convolution method: {method}. Currently only 'gaussian' is supported.")

        # Output based on the input type
        if isinstance(target_img, Path):
            # Save the convolved image to a new FITS file
            if not output_path:
                output_path = target_img.parent / f'conv_{target_img.name}'
            os.makedirs(output_path.parent, exist_ok=True)

            # Update header
            header['CONVOLVE'] = (True, 'Image has been convolved.')
            header['CONVMTD'] = (method, 'Convolution method used.')
            header['CONVTIME'] = (Time.now().isot, 'Time of convolution operation.')
            header['CONVFILE'] = (str(target_img), 'Original file path before convolution')

            hdu = fits.PrimaryHDU(convolved_image, header=header)
            hdu.writeto(output_path, overwrite=True)
            
            self.print(f"Image convolution complete. Output: {output_path}", print_output)
            return str(output_path)

        elif isinstance(target_img, np.ndarray):
            if target_header is not None:
                self.print('Image convolution complete with header.\n', print_output)
                return convolved_image, target_header
            else:
                self.print('Image convolution complete.\n', print_output)
                return convolved_image

    def img_crdetection(self,
                        target_img: Union[str, Path, np.ndarray],
                        target_header: Optional[Header] = None,
                        output_path: Optional[str] = None,
                        gain: float = 1.0,
                        readnoise: float = 6.0,
                        sigclip: float = 4.5,
                        sigfrac: float = 0.5,
                        objlim: float = 2.0,
                        niter: int = 4,
                        cleantype: str = 'medmask',
                        fsmode: str = 'median',
                        verbose: bool = True,
                        print_output: bool = True):

        """
        Detect and clean cosmic rays in the input image using the astroscrappy package.
        """
        from pathlib import Path
        import numpy as np
        from astropy.io import fits
        import astroscrappy as cr
        import os

        self.print(f"Start cosmic ray detection...", print_output)

        # Convert target_img to Path if it's a string
        if isinstance(target_img, (str, Path)):
            target_img = Path(target_img)

            if not target_img.is_file():
                raise FileNotFoundError(f"File {target_img} does not exist.")

            # Read data and header from FITS file
            with fits.open(target_img) as hdul:
                target_data = hdul[0].data
                target_header = hdul[0].header

            # Perform cosmic ray detection and cleaning
            mask, clean_image = cr.detect_cosmics(
                target_data, gain=gain, readnoise=readnoise, 
                sigclip=sigclip, sigfrac=sigfrac, 
                objlim=objlim, niter=niter, 
                cleantype=cleantype, fsmode=fsmode, 
                verbose=verbose
            )

            # Prepare output path
            if not output_path:
                output_path = target_img.parent / f"crclean_{target_img.name}"
            os.makedirs(output_path.parent, exist_ok=True)

            # Write the cleaned image to a new FITS file
            fits.writeto(output_path, clean_image, target_header, overwrite=True)

            self.print(f"Cosmic ray cleaning complete. Output: {output_path}", print_output)
            return str(output_path)

        elif isinstance(target_img, np.ndarray):
            target_data = target_img

            # Perform cosmic ray detection and cleaning
            mask, clean_image = cr.detect_cosmics(
                target_data, gain=gain, readnoise=readnoise, 
                sigclip=sigclip, sigfrac=sigfrac, 
                objlim=objlim, niter=niter, 
                cleantype=cleantype, fsmode=fsmode, 
                verbose=verbose
            )

            if target_header is not None:
                self.print("Cosmic ray cleaning complete (returning array and header).", print_output)
                return clean_image, target_header
            else:
                self.print("Cosmic ray cleaning complete (returning array only).", print_output)
                return clean_image

        else:
            raise TypeError("target_img must be either a Path, string (FITS file path), or a NumPy array.")


    def img_subtractbkg(self, 
                        target_img: Union[str, Path, np.ndarray],
                        target_header: Optional[Header] = None,
                        output_path: Optional[str] = None,
                        apply_2D_bkg: bool = False,
                        use_header : bool = False,
                        bkg_key: str = 'SKYVAL',
                        bkgsig_key: str = 'SKYSIG',
                        mask_sources: bool = False,
                        mask_source_size_in_pixel: int = 30,
                        bkg_estimator: str = 'median',
                        sigma: float = 5.0, 
                        box_size: int = 100, 
                        filter_size: int = 3, 
                        visualize: bool = False,
                        save_bkgmap: bool = False,
                        print_output: bool = True):

        """
        target_img: Union[str, Path, np.ndarray] = filelist[0]
        target_header: Optional[Header] = None
        output_path: Optional[str] = None
        apply_2D_bkg: bool = False
        bkg_key: str = 'SKYVAL'
        bkgsig_key: str = 'SKYSIG'
        mask_sources: bool = False
        mask_source_size_in_pixel: int = 10
        bkg_estimator: str = 'median'
        sigma: float = 5.0
        box_size: int = 100
        filter_size: int = 3
        visualize: bool = False
        print_output: bool = True
        Subtract background from the image using sigma-clipped statistics.
        """
        from pathlib import Path
        import numpy as np
        from astropy.io import fits
        from astropy.stats import SigmaClip, sigma_clipped_stats
        from photutils.background import Background2D, MedianBackground, MeanBackground, SExtractorBackground
        from photutils.segmentation import detect_threshold, detect_sources
        from photutils.utils import circular_footprint
        from astropy.time import Time
        import os
        import matplotlib.pyplot as plt

        self.print(f"Start background subtraction... [2D_bkg ={apply_2D_bkg}, Mask_sources ={mask_sources}, Use_header ={use_header}]", print_output)

        # Convert target_img to Path if it's a string
        if isinstance(target_img, (str, Path)):
            target_img = Path(target_img)

            if not target_img.is_file():
                raise FileNotFoundError(f"File {target_img} does not exist.")

            # Read data and header from FITS file
            with fits.open(target_img) as hdul:
                target_data = hdul[0].data
                target_header = hdul[0].header

        elif isinstance(target_img, np.ndarray):
            target_data = target_img
        else:
            raise TypeError("target_img must be either a Path, string (FITS file path), or a NumPy array.")

        # If the background value is already in the header, use it
        if use_header:
            bkg_value = float(target_header[bkg_key])
            bkg_value_median = bkg_value
            bkg_rms = float(target_header.get(bkgsig_key, 0))
        # Otherwise, estimate the background using sigma-clipped statistics
        else:
            # Create a mask for sources in the image
            mask = None
            if mask_sources:
                self.print('Masking sources before background estimation...', print_output)
                sigma_clip = SigmaClip(sigma=sigma)
                threshold = detect_threshold(target_data, nsigma=sigma, sigma_clip=sigma_clip)
                segment_img = detect_sources(target_data, threshold, npixels=mask_source_size_in_pixel)
                footprint = circular_footprint(radius=mask_source_size_in_pixel)
                mask = segment_img.make_source_mask(footprint=footprint)

            # Estimate background using sigma-clipped statistics
            bkg_estimator_dict = {
                'mean': MeanBackground,
                'median': MedianBackground,
                'sextractor': SExtractorBackground
            }
            bkg_estimator_function = bkg_estimator_dict[bkg_estimator.lower()]

            if apply_2D_bkg:
                self.print('Applying 2D background subtraction...', print_output)
                bkg = Background2D(target_data, (box_size, box_size), mask=mask,
                                filter_size=(filter_size, filter_size),
                                sigma_clip=SigmaClip(sigma=sigma),
                                bkg_estimator=bkg_estimator_function())
                bkg_value = bkg.background
                bkg_value_median = bkg.background_median
                bkg_rms = bkg.background_rms_median
            else:
                self.print('Applying 1D background subtraction...', print_output)
                # Estimate background using sigma-clipped statistics
                clipped_data = sigma_clipped_stats(target_data, sigma=sigma, mask=mask)
                bkg_value = clipped_data[1] if bkg_estimator == 'median' else clipped_data[0]
                bkg_value_median = clipped_data[0]
                bkg_rms = clipped_data[2]

        # Subtract background
        self.print('Subtracting background...', print_output)
        data_bkg_subtracted = target_data - bkg_value

        if visualize:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            fig, ax = plt.subplots(1, 3, figsize=(12, 6))
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im0 = ax[0].imshow(target_data, origin='lower', cmap='Greys_r', vmin=bkg_value_median, vmax=bkg_value_median + bkg_rms)
            ax[0].set_title('Original Image')
            fig.colorbar(im0, cax=cax, orientation='vertical')

            if apply_2D_bkg:
                bkg_img = bkg.background
            else:
                bkg_img = np.full(target_data.shape, bkg_value)

            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im1 = ax[1].imshow(bkg_img, origin='lower', cmap='Greys_r', vmin=bkg_value_median, vmax=bkg_value_median + bkg_rms)
            ax[1].set_title('Background')
            fig.colorbar(im1, cax=cax, orientation='vertical')

            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im2 = ax[2].imshow(data_bkg_subtracted, origin='lower', cmap='Greys_r', vmin=0, vmax=bkg_rms)
            ax[2].set_title('Background-Subtracted Image')
            fig.colorbar(im2, cax=cax, orientation='vertical')
            plt.tight_layout()
            plt.show()

        # Output handling
        if isinstance(target_img, Path):
            # Update FITS header
            target_header['SUBBKG'] = (True, 'Background subtracted')
            target_header['SUBBTIME'] = (Time.now().isot, 'Time of background subtraction')
            target_header['SUBBVALU'] = (bkg_value_median, 'Background median value')
            target_header['SUBBSIG'] = (bkg_rms, 'Background standard deviation')
            target_header['SUBBFILE'] = (str(target_img), 'Original file path before background subtraction')
            target_header['SUBBIS2D'] = (apply_2D_bkg, '2D background subtraction')
            target_header['SUBBMASK'] = (mask_sources, 'Mask sources before background estimation')
            target_header['SUBBTYPE'] = (bkg_estimator, 'Background estimator')

            if not output_path:
                output_path = target_img.parent / f"subbkg_{target_img.name}"
            os.makedirs(output_path.parent, exist_ok=True)
            fits.writeto(output_path, data_bkg_subtracted, target_header, overwrite=True)
            if save_bkgmap:
                bkg_filename = output_path.stem + '.bkgmap.fits'
                bkg_path = output_path.parent / bkg_filename
                fits.writeto(bkg_path, bkg_img, target_header, overwrite=True)
                self.print(f"Background map saved to {bkg_path}", print_output)

            self.print(f"Background subtraction completed. Output saved to {output_path}", print_output)
            return str(output_path)

        elif isinstance(target_img, np.ndarray):
            if target_header:
                target_header['SUBBKG'] = (True, 'Background subtracted')
                target_header['SUBBTIME'] = (Time.now().isot, 'Time of background subtraction')
                target_header['SUBBVALU'] = (bkg_value_median, 'Background median value')
                target_header['SUBBSIG'] = (bkg_rms, 'Background standard deviation')
                target_header['SUBBIS2D'] = (apply_2D_bkg, '2D background subtraction')
                target_header['SUBBMASK'] = (mask_sources, 'Mask sources before background estimation')
                target_header['SUBBTYPE'] = (bkg_estimator, 'Background estimator')

                self.print("Background subtraction complete (returning array and header).", print_output)
                
                if save_bkgmap:
                    return data_bkg_subtracted, target_header, bkg_img
                else:
                    return data_bkg_subtracted, target_header
            else:
                self.print("Background subtraction complete (returning array only).", print_output)
                if save_bkgmap:
                    return data_bkg_subtracted, bkg_img
                else:
                    return data_bkg_subtracted

    def img_combine(self,
                    filelist: List[Union[str, Path]],
                    output_path: Optional[str] = None,

                    # Combine parameters
                    combine_method: str = 'median',
                    clip: str = 'extrema',
                    clip_sigma_low: int = 2,
                    clip_sigma_high: int = 5,
                    clip_minmax_min: int = 3,
                    clip_minmax_max: int = 3,
                    clip_extrema_nlow: int = 1,
                    clip_extrema_nhigh: int = 1,

                    # Background subtraction parameters
                    subbkg: bool = True,
                    apply_2D_bkg: bool = False,
                    use_header : bool = True,
                    bkg_key: str = 'SKYVAL',
                    bkgsig_key: str = 'SKYSIG',
                    mask_sources: bool = False,
                    mask_source_size_in_pixel: int = 10,
                    bkg_estimator: str = 'median',
                    sigma: float = 5.0,
                    box_size: int = 100,
                    filter_size: int = 3,

                    # ZP scaling parameters
                    scale: bool = True,
                    zp_key: str = 'ZP_AUTO',
                    zp_reference: float = None,
                    
                    # Alignment parameters
                    align: bool = False,
                    

                    print_output: bool = True):

        from pathlib import Path
        from ccdproc import CCDData, combine
        import psutil
        import os
        from astropy.time import Time
        import numpy as np
        from astropy.io import fits
        from tqdm import tqdm

        def print_memory_usage(output_string='Memory usage'):
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            print(f"{output_string}: {mem_info.rss / 1024**2:.2f} MB")

        if len(filelist) <= 3:
            clip = None
            self.print('Number of filelist is lower than the minimum. Skipping clipping process... \n', print_output)

        self.print('Start image combine... \n', print_output)

        # Ensure filelist contains Paths
        filelist = [Path(file) for file in filelist]

        ccdlist = []
        for filename in tqdm(filelist, desc='Reading files...'):
            if not filename.is_file():
                raise FileNotFoundError(f"File {filename} does not exist.")

            with fits.open(filename, memmap=False) as hdul:
                data = hdul[0].data
                header = hdul[0].header

            ccdlist.append(CCDData(data, unit='adu', meta=header))

        # Background subtraction
        if subbkg:
            self.print('Applying background subtraction...', print_output)
            for idx, inim in tqdm(enumerate(ccdlist), desc='Background subtraction...'):
                data_bkg_subtracted, inim.header = self.img_subtractbkg(
                    target_img=inim.data,
                    target_header=inim.header,
                    apply_2D_bkg=apply_2D_bkg,
                    bkg_key=bkg_key,
                    bkgsig_key=bkgsig_key,
                    mask_sources=mask_sources,
                    mask_source_size_in_pixel=mask_source_size_in_pixel,
                    bkg_estimator=bkg_estimator,
                    sigma=sigma,
                    box_size=box_size,
                    filter_size=filter_size,
                    print_output=print_output
                )
                inim.data = data_bkg_subtracted

        print_memory_usage(output_string='Memory usage after subbkg')

        # Scaling
        if scale:
            self.print('Applying image scaling...', print_output)
            if not zp_reference:
                zp_reference = np.min([inim.header[zp_key] for inim in ccdlist])
            for inim in tqdm(ccdlist, desc='Image scaling...'):
                scaled_data, inim.header = self.img_scale(
                    target_img=inim.data,
                    target_header=inim.header,
                    zp_target=inim.header[zp_key],
                    zp_reference=zp_reference,
                    zp_key=zp_key,
                    print_output=print_output
                )
                inim.data = scaled_data

        print_memory_usage(output_string='Memory usage after scaling')
        
        # Align
        if align:
            self.print('Aligning images...', print_output)
            reference_image = ccdlist[0]
            for idx, inim in tqdm(enumerate(ccdlist), desc='Image alignment...'):
                aligned_data, aligned_header = self.img_astroalign(
                    target_img = inim.data,
                    reference_img = reference_image.data,
                    target_header = inim.header,
                    reference_header = reference_image.header,
                    print_output = print_output
                )
                inim.data = aligned_data

        print_memory_usage(output_string='Memory usage after alignment')

        hdr = ccdlist[0].header.copy()

        for i, file in enumerate(filelist):
            hdr[f'COMBIM{i+1}'] = file.name

        hdr['NCOMBINE'] = len(filelist)
        if 'JD' in hdr:
            hdr['JD'] = Time(np.mean([inim.header['JD'] for inim in ccdlist]), format='jd').value
        if 'DATE-OBS' in hdr:
            hdr['DATE-OBS'] = Time(np.mean([Time(inim.header['DATE-OBS']).jd for inim in ccdlist]), format='jd').isot
        hdr['TOTALEXP'] = (float(np.sum([inim.header['EXPTIME'] for inim in ccdlist])), 'Total exposure time of the combined image')

        # Combine with appropriate method and clipping
        combine_kwargs = {}
        if clip == 'minmax':
            combine_kwargs['minmax_clip'] = True
            combine_kwargs['minmax_clip_min'] = clip_minmax_min
            combine_kwargs['minmax_clip_max'] = clip_minmax_max
        elif clip == 'sigma':
            combine_kwargs['sigma_clip'] = True
            combine_kwargs['sigma_clip_low_thresh'] = clip_sigma_low
            combine_kwargs['sigma_clip_high_thresh'] = clip_sigma_high
        elif clip == 'extrema':
            combine_kwargs['nlow'] = clip_extrema_nlow
            combine_kwargs['nhigh'] = clip_extrema_nhigh

        combined = combine(ccdlist, method=combine_method, **combine_kwargs)
        combined.header = hdr
        combined.data = combined.data.astype(np.float32)

        if not output_path:
            output_path = filelist[0].parent / f"com_{filelist[0].name}"
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)

        fits.writeto(output_path, combined.data, header=hdr, overwrite=True)

        print_memory_usage(output_string='Memory usage after writing')

        self.print('Combine complete \n', print_output)
        self.print('Combine information', print_output)
        self.print(60 * '=', print_output)
        self.print(f'Ncombine = {len(filelist)}', print_output)
        self.print(f'method   = {clip}(clipping), {combine_method}(combining)', print_output)
        self.print(f'image path = {output_path.name}', print_output)

        return str(output_path)

    def run_psfex(self, 
                  target_img: Union[str, Path], 
                  sex_configfile: Union[str, Path], 
                  psfex_configfile: Union[str, Path],                   
                  output_path: Optional[Union[str, Path]] = None,
                  sex_params: Optional[dict] = None,
                  psfex_params: Optional[dict] = None,
                  print_output: bool = True) -> None:

        """
        Run SExtractor followed by PSFEx on the specified image using the provided configuration and parameters.
        """
        from pathlib import Path
        import os
        import subprocess
        import datetime

        self.print('Start PSFEx process...=====================', print_output)
        current_dir = Path.cwd()
        target_path = Path(target_img)
        psfex_config_path = Path(psfex_configfile)

        # Run SExtractor
        output_file = self.run_sextractor(target_img=str(target_path), 
                                          sex_configfile=str(sex_configfile), 
                                          sex_params=sex_params, 
                                          return_result=False, 
                                          print_output=False)

        # Load default PSFEx config
        all_params = self.load_config(psfex_config_path)

        # Set up history directory for outputs
        if not output_path:
            output_path = target_path.parent
        if output_path.is_file():
            output_path = output_path.parent
        
        # Handle CHECKIMAGE_NAME parameter
        if not psfex_params:
            psfex_params = {}

        fits_files = (psfex_params.get('CHECKIMAGE_NAME') or all_params.get('CHECKIMAGE_NAME', '')).split(',')
        abspath_fits_files = []
        for file_ in fits_files:
            filename = target_path.stem + "." + file_
            abspath_fits_files.append(str(output_path / filename))
        abspath_fits_files_str = ','.join(abspath_fits_files)
        psfex_params['CHECKIMAGE_NAME'] = abspath_fits_files_str

        # Build PSFEx parameter string
        psfexparams_str = ' '.join([f"-{key} {value}" for key, value in psfex_params.items()])

        command = f"psfex {output_file} -c {psfex_configfile} {psfexparams_str}"

        try:
            os.chdir(self.psfexpath)
            self.print('RUN COMMAND: ', command)
            result = subprocess.run(command, shell=True, check=True, 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if print_output:
                self.print(result.stdout.decode(), print_output)
                self.print(result.stderr.decode(), print_output)

            self.print("PSFEx process finished=====================", print_output)
            return abspath_fits_files
        except subprocess.CalledProcessError as e:
            self.print(f"Error during PSFEx execution: {e.stderr.decode()}", print_output)
            return None
        finally:
            os.chdir(current_dir)  # Ensure directory is reset
    
    def run_hotpants(self,
                     target_img: Union[str, Path],
                     reference_img: Union[str, Path],
                     output_path: Optional[Union[str, Path]] = None,
                     convolve_path: Optional[Union[str, Path]] = None,
                     target_mask: Optional[Union[str, Path]] = None,
                     reference_mask: Optional[Union[str, Path]] = None,
                     stamp: Optional[Union[str, Path]] = None,
                     
                     # Subtract background
                     subbkg : bool = True, 
                     apply_2D_bkg: bool = False,
                     use_header : bool = True,
                     bkg_key: str = 'SKYVAL',
                     bkgsig_key: str = 'SKYSIG',
                     mask_sources: bool = False,
                     mask_source_size_in_pixel: int = 10,
                     bkg_estimator: str = 'median',
                     sigma: float = 5.0,
                     box_size: int = 100,
                     filter_size: int = 3,
                        
                     # Zeropoint scaling
                     scale: bool = True,
                     zp_key: str = 'ZP_AUTO',
                     zp_reference: float = None,    
                     
                     # Alignment
                     align: bool = True,
                     detection_sigma: float = 5, 

                     # Hotpants config
                     convim: str = 't',
                     normim: str = 'i',
                     iu: int = 60000,
                     il: int = -10000,
                     tu: int = 6000000,
                     tl: int = -10000,
                     nrx: int = 3,
                     nry: int = 2,
                     v: int = 0,
                     print_output: bool = True) -> str:
        """
        Run Hotpants for image subtraction.
        """
        from pathlib import Path
        import subprocess

        target_path = Path(target_img)
        reference_path = Path(reference_img)
        current_dir = Path.cwd()

        if not target_path.is_file():
            raise FileNotFoundError(f"Target image {target_path} does not exist.")
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference image {reference_path} does not exist.")
        
        if subbkg | scale | align:
            target_data = fits.getdata(target_path)
            target_header = fits.getheader(target_path)
            ref_data = fits.getdata(reference_path)
            ref_header = fits.getheader(reference_path)
        
        if subbkg:
            target_data, target_header = self.img_subtractbkg(target_img = target_data, target_header = target_header, 
                                                              apply_2D_bkg = apply_2D_bkg, use_header = use_header,
                                                              bkg_key = bkg_key, bkgsig_key = bkgsig_key, mask_sources = mask_sources,
                                                              mask_source_size_in_pixel = mask_source_size_in_pixel, bkg_estimator = bkg_estimator,
                                                              sigma = sigma, box_size = box_size, filter_size = filter_size, print_output = print_output)
            ref_data, ref_header = self.img_subtractbkg(target_img = ref_data, target_header = ref_header,
                                                        apply_2D_bkg = apply_2D_bkg, use_header = use_header,
                                                        bkg_key = bkg_key, bkgsig_key = bkgsig_key, mask_sources = mask_sources,
                                                        mask_source_size_in_pixel = mask_source_size_in_pixel, bkg_estimator = bkg_estimator,
                                                        sigma = sigma, box_size = box_size, filter_size = filter_size, print_output = print_output)
        if scale:
            if zp_reference is None:
                zp_target = target_header[zp_key]
                zp_reference = ref_header[zp_key]
            else:
                zp_target = zp_reference
                zp_reference = zp_reference
            # When zp_target is larger than zp_reference, scale zp_reference to zp_target
            if zp_target > zp_reference:
                target_data, target_header = self.img_scale(target_img = target_data, target_header = target_header,
                                                            zp_reference = zp_reference, zp_key = zp_key)
            # When zp_reference is larger than zp_target, scale zp_target to zp_reference
            elif zp_target < zp_reference:
                ref_data, ref_header = self.img_scale(target_img = ref_data, target_header = ref_header,
                                                      zp_reference = zp_target, zp_key = zp_key)
            # When zp_reference is given, so we need to scale both images to the same zeropoint
            else:
                target_data, target_header = self.img_scale(target_img = target_data, target_header = target_header,
                                                            zp_reference = zp_reference, zp_key = zp_key)
                ref_data, ref_header = self.img_scale(target_img = ref_data, target_header = ref_header,
                                                        zp_reference = zp_reference, zp_key = zp_key)
        if align:
            target_data, target_header = self.img_astroalign(target_img = target_data, reference_img = ref_data,
                                                             target_header = target_header, reference_header = ref_header,
                                                             detection_sigma = detection_sigma, print_output = print_output)            
        
        if subbkg | scale | align:
            target_path = target_path.parent / f'sci_{target_path.name}'
            ref_path = reference_path.parent / f'ref_{reference_path.name}'
            fits.writeto(target_path, target_data, target_header, overwrite = True)
            fits.writeto(ref_path, ref_data, ref_header, overwrite = True)    

        if not output_path:
            output_path = target_path.parent / f'sub_{target_path.name}'
        else:
            output_path = Path(output_path)

        self.print(f'Starting image subtraction with hotpants on {target_path.name}...', print_output)

        # Build the command
        command = [
            'hotpants',
            '-c', convim,
            '-n', normim,
            '-inim', str(target_path),
            '-tmplim', str(reference_path),
            '-outim', str(output_path),
            '-iu', str(iu),
            '-il', str(il),
            '-tu', str(tu),
            '-tl', str(tl),
            '-v', str(v),
            '-nrx', str(nrx),
            '-nry', str(nry)
        ]

        if convolve_path:
            convolve_path = Path(convolve_path)
            command.extend(['-oci', str(convolve_path)])
        if target_mask:
            target_mask = Path(target_mask)
            command.extend(['-imi', str(target_mask)])
        if reference_mask:
            reference_mask = Path(reference_mask)
            command.extend(['-tmi', str(reference_mask)])
        if stamp:
            stamp = Path(stamp)
            command.extend(['-ssf', str(stamp)])

        self.print(f"RUN COMMAND: {' '.join(command)}", print_output)

        try:
            result = subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
                timeout=900
            )
            if print_output:
                self.print(result.stdout, print_output)
                self.print(result.stderr, print_output)

            self.print(f"Image subtraction completed successfully. Output saved to {output_path}", print_output)
            return str(output_path)

        except subprocess.CalledProcessError as e:
            self.print(f"Error during hotpants execution: {e.stderr}", print_output)
            return ""

        except subprocess.TimeoutExpired:
            self.print(f"Hotpants process timed out after 900 seconds.", print_output)
            return ""

    def run_astrometry(self,
                       target_img: Union[str, Path], 
                       sex_configfile: Union[str, Path],
                       output_path: Optional[Union[str, Path]] = None,
                       ra: Optional[float] = None,
                       dec: Optional[float] = None,
                       radius: Optional[float] = None,
                       scalelow: float = 0.6, 
                       scalehigh: float = 0.8, 
                       remove: bool = True,
                       print_output: bool = True):

        """
        Run the Astrometry.net process to solve WCS coordinates.
        """
        import os
        import subprocess

        target_path = Path(target_img)
        target_dir = target_path.parent
        sexconfig_path = Path(sex_configfile)
        current_dir = Path.cwd()

        if not target_path.is_file():
            raise FileNotFoundError(f"Target image {target_path} does not exist.")
        if not sexconfig_path.is_file():
            raise FileNotFoundError(f"SExtractor config file {sexconfig_path} does not exist.")

        try:
            self.print('Start Astrometry process...=====================', print_output)

            # Set up directories and copy configuration files
            os.chdir(self.sexpath)
            os.system(f'cp {sexconfig_path} {self.sexpath}/*.param {self.sexpath}/*.conv {self.sexpath}/*.nnw {target_dir}')
            os.chdir(target_dir)
            self.print(f'Solving WCS using Astrometry with RA/Dec of {ra}/{dec} and radius of {radius} arcmin', print_output)

            # Define output path
            if not output_path:
                output_path = target_dir / f'astrometry_{target_path.name}'
            else:
                output_path = Path(output_path)

            # Build the command
            command = [
                'solve-field',
                str(target_path),
                '--cpulimit', '300',
                '--use-source-extractor',
                '--source-extractor-config', str(sexconfig_path),
                '--x-column', 'X_IMAGE',
                '--y-column', 'Y_IMAGE',
                '--sort-column', 'MAG_AUTO',
                '--sort-ascending',
                '--scale-unit', 'arcsecperpix',
                '--scale-low', str(scalelow),
                '--scale-high', str(scalehigh),
                '--no-remove-lines',
                '--uniformize', '0',
                '--no-plots',
                '--new-fits', str(output_path),
                '--temp-dir'
            ]

            if ra is not None and dec is not None:
                command.extend(['--ra', str(ra), '--dec', str(dec)])
            if radius is not None:
                command.extend(['--radius', str(radius)])

            # Run astrometry with timeout
            result = subprocess.run(command, timeout=900, check=True, text=True, capture_output=True)
            if print_output:
                self.print(result.stdout, print_output)
                self.print(result.stderr, print_output)

            # Check the number of output files
            orinum = int(subprocess.check_output("ls C*.fits | wc -l", shell=True).strip())
            resnum = int(subprocess.check_output("ls a*.fits | wc -l", shell=True).strip())

            # Clean up intermediate files
            if remove:
                os.system(f'rm -f tmp* *.conv default.nnw *.wcs *.rdls *.corr *.xyls *.solved *.axy *.match check.fits *.param {sexconfig_path.name}')

            self.print('Astrometry process finished=====================', print_output)
            return str(output_path)

        except subprocess.TimeoutExpired:
            self.print("The astrometry process exceeded the timeout limit.", print_output)
            return None
        except subprocess.CalledProcessError as e:
            self.print(f"An error occurred while running the astrometry process: {e}", print_output)
            return None
        except Exception as e:
            self.print(f"An unknown error occurred while running the astrometry process: {e}", print_output)
            return None
        finally:
            os.chdir(current_dir)


    def run_sextractor(self, 
                       target_img: Union[str, Path], 
                       sex_configfile: Union[str, Path], 
                       image_mask: Optional[Union[str, Path]] = None,
                       sex_params: Optional[dict] = None, 
                       return_result: bool = True, 
                       print_output: bool = True):

        """
        Parameters
        ----------
        1. target_img : str
                Absolute path of the target image.
        2. sex_params : dict
                Configuration parameters in dict format. Can be loaded by load_sexconfig().
        3. sex_configfile : str
                Path to the SExtractor configuration file.
        4. return_result : bool
                If True, returns the result as an astropy table.

        Returns
        -------
        1. result : astropy.table.Table or str
                    Source extractor result as a table or the catalog file path.

        Notes
        -------
        This method runs SExtractor on the specified image using the provided configuration and parameters.
        """
        self.print('Start SExtractor process...=====================', print_output)

        # Switch to the SExtractor directory
        current_path = os.getcwd()
        os.chdir(self.sexpath)
        
        # Load and apply SExtractor parameters
        all_params = self.load_config(sex_configfile)
        sexparams_str = ''

        if sex_params:
            if "CATALOG_NAME" not in sex_params.keys():
                sex_params['CATALOG_NAME'] = f"{os.path.join(self.config['SEX_HISTORYDIR'], os.path.basename(sex_params['CATALOG_NAME']))}"
            else:
                pass
        else:
            sex_params = dict()
            sex_params['CATALOG_NAME'] = f"{os.path.join(self.config['SEX_HISTORYDIR'], all_params['CATALOG_NAME'])}"

        if image_mask:
            sex_params['FLAG_IMAGE'] = image_mask

        for key, value in sex_params.items():
            sexparams_str += f'-{key} {value} '
            all_params[key] = value
        

        # Command to run SExtractor
        command = f"source-extractor {target_img} -c {sex_configfile} {sexparams_str}"
        #os.makedirs(os.path.dirname(all_params['CATALOG_NAME']), exist_ok=True)
        print('RUN COMMAND: ', command)

        try:
            # Run the SExtractor command using subprocess.run
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.print("SExtractor process finished=====================", print_output)

            if return_result:
                # Read the catalog produced by SExtractor
                sexresult = ascii.read(all_params['CATALOG_NAME'])
                os.chdir(current_path)
                return sexresult
            else:
                return all_params['CATALOG_NAME']
        except:
            self.print(f"Error during SExtractor execution", print_output)
            os.chdir(current_path)
            return None

    def run_scamp(self, 
                  target_img: Union[str, List[str], Path, List[Path]], 
                  sex_configfile: Union[str, Path], 
                  scamp_configfile: Union[str, Path],                   
                  sex_params: Optional[dict] = None,
                  scamp_params: Optional[dict] = None,
                  update_files: bool = True, 
                  print_output: bool = True):

        """
        Run SCAMP for astrometric calibration on a set of images.
        """
        from pathlib import Path
        import os
        import subprocess
        import re
        from tqdm import tqdm
        from astropy.io import fits

        # Ensure target_img is a list
        if isinstance(target_img, (str, Path)):
            target_img = [Path(target_img)]
        else:
            target_img = [Path(img) for img in target_img]

        if not target_img:
            raise ValueError("No valid images provided for SCAMP.")

        self.print(f'Start SCAMP process on {len(target_img)} images...=====================', print_output)

        # Run SExtractor on each image
        sex_output_images = {}
        for image in tqdm(target_img, desc='Running Source Extractor...'):
            if not image.is_file():
                self.print(f"Warning: Image {image} does not exist. Skipping...", print_output)
                continue

            basename = image.stem
            history_dir = Path(self.config['SCAMP_HISTORYDIR']) / basename
            history_dir.mkdir(parents=True, exist_ok=True)
            history_path = history_dir / f"{basename}_scamp.cat"

            # Ensure sex_params is a dictionary and update it
            sex_params = sex_params or {}
            sex_params.update({
                'CATALOG_NAME': str(history_path),
                'PARAMETERS_NAME': str(Path(self.sexpath) / 'scamp.param')
            })

            output_file = self.run_sextractor(target_img=str(image), 
                                              sex_configfile=str(sex_configfile), 
                                              sex_params=sex_params, 
                                              return_result=False, 
                                              print_output=False)
            if output_file:
                sex_output_images[str(image)] = output_file

        # Filter out images that failed
        if not sex_output_images:
            self.print("No valid SExtractor catalogs generated. Aborting SCAMP.", print_output)
            return None

        scamp_output_images = {key: value.replace('.cat', '.head') for key, value in sex_output_images.items()}
        all_images_str = ' '.join(sex_output_images.values())

        # Load and apply SCAMP parameters
        all_params = self.load_config(scamp_configfile)
        scamp_params = scamp_params or {}
        scamp_params.update(all_params)
        scampparams_str = ' '.join([f'-{key} {value}' for key, value in scamp_params.items()])

        # SCAMP command
        command = f'scamp {all_images_str} -c {scamp_configfile} {scampparams_str}'

        try:
            current_path = Path.cwd()
            result_dir = Path(self.scamppath) / 'result'
            result_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(result_dir)

            self.print(f'RUN COMMAND: {command}', print_output)
            subprocess.run(command, shell=True, check=True, text=True, capture_output=True)

            self.print("SCAMP process finished=====================", print_output)

            if update_files:
                def sanitize_header(header: fits.Header) -> fits.Header:
                    """
                    Remove non-ASCII and non-printable characters from a FITS header.
                    """
                    sanitized_header = fits.Header()
                    for card in header.cards:
                        key, value, comment = card
                        if isinstance(value, str):
                            value = re.sub(r'[^\x20-\x7E]+', '', value)
                        sanitized_header[key] = (value, comment)
                    return sanitized_header

                def update_fits_with_head(image_file: Path, head_file: Path):
                    """
                    Update the FITS image header with WCS info from SCAMP-generated .head file.
                    """
                    with open(head_file, 'r') as head:
                        head_content = head.read()
                    head_header = fits.Header.fromstring(head_content, sep='\n')
                    head_header = sanitize_header(head_header)

                    with fits.open(image_file, mode='update') as hdul:
                        hdul[0].header.update(head_header)
                        hdul.flush()

                    self.print(f"Updated WCS for {image_file} using {head_file}", print_output)

                for image, header in scamp_output_images.items():
                    update_fits_with_head(Path(image), Path(header))

                return list(scamp_output_images.keys())
            else:
                return list(scamp_output_images.values())

        except subprocess.CalledProcessError as e:
            self.print(f"Error during SCAMP execution: {e}", print_output)
            return None
        finally:
            os.chdir(current_path)


    def run_swarp(self,
                  target_img: Union[str, List[str], Path, List[Path]], 
                  output_path: Union[str, Path],
                  swarp_configfile: Union[str, Path],
                  swarp_params: Optional[dict] = None,
                  weight_inpath: Optional[Union[str, List[str], Path, List[Path]]] = None,
                  weight_outpath: Optional[str] = None,
                  center_ra: Optional[float] = None,
                  center_dec: Optional[float] = None,
                  combine: bool = False,
                  combine_type: str = 'median',
                  print_output: bool = True) -> None:

        """_summary_

        Args:
            target_img = glob.glob('/home/hhchoi1022/data/test/swarp/calib*100.fits')
            output_path = '/home/hhchoi1022/data/test/swarp/coadd.fits'
            swarp_configfile = self.get_swarpconfigpath('7DT', 'C361K', 1, 'HIGH')
            swarp_params = None
            weight_inpath = None
            weight_outpath = None
            center_ra = None
            center_dec = None
            combine = True
            combine_type = 'median'
            print_output = True

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        
        
        from pathlib import Path
        import os
        import subprocess
        import re
        from tqdm import tqdm   
        from astropy.io import fits

        # Ensure target_img is a list
        if isinstance(target_img, (str, Path)):
            target_img = [Path(target_img)]
        else:
            target_img = [Path(img) for img in target_img]

        if not target_img:
            raise ValueError("No valid images provided for SCAMP.")
        
        output_path = Path(output_path)
        swarp_config = Path(swarp_configfile)
        weight_inpath = Path(weight_inpath) if weight_inpath else None
        weight_outpath = Path(weight_outpath) if weight_outpath else None
        
        # Load and apply SWARP parameters
        all_params = self.load_config(swarp_configfile)
        swarpparams_str = ''
        if not swarp_params:
            swarp_params = dict()
        swarp_params['IMAGEOUT_NAME'] = path_outim
        swarp_params['WEIGHTOUT_NAME'] = os.path.splitext(path_outim)[0] + '.weight.fits'
        if swarp_params:
            for key, value in swarp_params.items():
                swarpparams_str += f'-{key} {value} '   
                all_params[key] = value
        
        # Command to run SWARP
        all_images_str = ' '.join(succeeded_images)
        command = f'SWarp {all_images_str} -c {swarp_configfile} {swarpparams_str}'
        
        try:
            current_path = os.getcwd()
            os.chdir(os.path.join(self.swarppath,'result'))
            # Run the SExtractor command using subprocess.run
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.print("SWARP process finished=====================", print_output)
            return path_outim
        except:
            return None

    def run_ds9(self, filelist, shell: str = '/bin/bash'):
        import subprocess
        import numpy as np
        '''
		parameters
		----------
		1. filelist : str or list or np.array
				(str) abspath of the fitsfile for visualization
				(list or np.array) the list of abspath of the fitsfiles for visualization
		
		returns 
		-------
		
		notes 
		-----
		-----
		'''
        ds9_options = "-scalemode zscale -scale lock yes -frame lock image "
        names = ""
        if (type(filelist) == str) | (type(filelist) == np.str_):
            names = filelist
        else:
            for file in filelist:
                names += file+" "
        ds9_command = "ds9 "+ds9_options+names+" &"
        print('Running "'+ds9_command+'" in the terminal...')
        sp = subprocess.Popen([shell, "-i", "-c", ds9_command])
        sp.communicate()
        # os.system(ds9_command)

    def to_regions(self, reg_ra, reg_dec, reg_size: float = 5.0, output_file_name: str = 'regions.reg'):
        from regions import CircleSkyRegion, write_ds9
        import astropy.units as u
        # Check if ra and dec are single float values or lists
        if isinstance(reg_ra, float) and isinstance(reg_dec, float):
            ra_list = [reg_ra]
            dec_list = [reg_dec]
        elif isinstance(reg_ra, list) and isinstance(reg_dec, list):
            ra_list = reg_ra
            dec_list = reg_dec
        else:
            ra_list = reg_ra
            dec_list = reg_dec

        regions = []
        for ra, dec in zip(ra_list, dec_list):
            center = SkyCoord(ra, dec, unit='deg', frame='icrs')
            radius = reg_size * u.arcsec  # Example radius
            region = CircleSkyRegion(center, radius)
            regions.append(region)
        # Write the regions to a DS9 region file
        output_file_path = os.path.join(self.configpath, output_file_name)
        write_ds9(regions, output_file_path)
        return output_file_path
    
    def visualize_image(self, filename : str):
        from astropy.visualization import ImageNormalize, ZScaleInterval
        import matplotlib.pyplot as plt
        data = fits.getdata(filename)
        zscale_interval = ZScaleInterval()
        norm = ImageNormalize(data, interval=zscale_interval)

        # Plot the normalized image
        plt.imshow(data, cmap='gray', norm=norm, origin='lower')
        plt.colorbar()
        plt.title(f'{os.path.basename(filename)}')
        plt.show()
        

#%% Initialization
if __name__ == '__main__':
    A = PhotometryHelper()
    print_output = True
    psfex_sexconfigfile = A.get_sexconfigpath(telescope = '7DT', ccd = 'C361K', readoutmode = 'HIGH', binning = 1, for_psfex = True)
    psfex_configfile = A.get_psfexconfigpath()
# %% Test get_imginfo

if __name__ == '__main__':
    import glob
    filelist = glob.glob('/home/hhchoi1022/data/test/swarp/*.fits')
    A = PhotometryHelper()
    tbl = A.get_imginfo(filelist)
#%% Load configuration
if __name__ == '__main__':
    import glob
    filelist = glob.glob('/home/hhchoi1022/data/test/swarp/*.fits')
    psfex_configpath = A.get_psfexconfigpath()
    scamp_configpath = A.get_scampconfigpath()
    sexconfig = A.get_sexconfigpath(telescope = '7DT', ccd = 'C361K', readoutmode = 'HIGH', binning = 1, for_psfex = False)
    sexconfig_psfex = A.get_sexconfigpath(telescope = '7DT', ccd = 'C361K', readoutmode = 'HIGH', binning = 1, for_psfex = True)
    sexconfig_scamp = A.get_sexconfigpath(telescope = '7DT', ccd = 'C361K', readoutmode = 'HIGH', binning = 1, for_scamp = True)
    swarpconfig = A.get_swarpconfigpath(telescope = '7DT', ccd = 'C361K', readoutmode = 'HIGH', binning = 1)
    print(A.load_config(psfex_configpath))
    print(A.load_config(scamp_configpath))
    print(A.load_config(sexconfig))
    print(A.load_config(sexconfig_psfex))
    print(A.load_config(sexconfig_scamp))
    print(A.load_config(swarpconfig))
#%% astrolaign 
if __name__ == '__main__':
    filelist = list(Path('/home/hhchoi1022/data/test/astroalign/').glob('*.fits'))
    # Path-based running
    A.img_astroalign(target_img = filelist[0], reference_img = filelist[1], output_path = None, print_output = True)
    # Data-based running
    target_img = fits.getdata(filelist[0])
    reference_img = fits.getdata(filelist[1])
    target_header = fits.getheader(filelist[0])
    reference_header = fits.getheader(filelist[1])
    aligned_data, aligned_heeader = A.img_astroalign(target_img = target_img, reference_img = reference_img, target_header = target_header, reference_header = reference_header, output_path = None, print_output = True)
    fits.writeto('/home/hhchoi1022/data/test/astroalign/aligned_database.fits', aligned_data, header = aligned_heeader, overwrite = True)
#%% cutout
if __name__ == '__main__':
    filelist = list(Path('/home/hhchoi1022/data/test/cutout/').glob('*.fits'))
    # Path-based running
    ## Cutout based on pixel
    A.img_cutout(target_img = filelist[0], output_path = None, xsize = 0.5, ysize = 0.5, xcenter = 4632, ycenter = 3194)
    ## Cutout based on WCS
    A.img_cutout(target_img = filelist[0], output_path = None, xcenter = '03:35:39', ycenter = '-82:21:39', xsize = 50, ysize = 50)
    ## Data-based running with pixel
    target_img = fits.getdata(filelist[0])
    target_header = fits.getheader(filelist[0])
    cutouted_data, cutouted_header = A.img_cutout(target_img = target_img, target_header = target_header, xsize = 0.5, ysize = 0.5, xcenter = 4632, ycenter = 3194)
    fits.writeto(Path(filelist[0]).with_name('cutout_data_bsased.fits'), cutouted_data, cutouted_header, overwrite = True)
    ## Data-based running with WCS
    cutouted_data, cutouted_header = A.img_cutout(target_img = target_img, target_header = target_header, xcenter = '03:35:39', ycenter = '-82:21:39', xsize = 50, ysize = 50)
    fits.writeto(Path(filelist[0]).with_name('cutout_data_bsased_wcs.fits'), cutouted_data, cutouted_header, overwrite = True)

#%% scaling
if __name__ == '__main__':
    filelist = list(Path('/home/hhchoi1022/data/test/scale/').glob('*.fits'))
    # Path-based running
    A.img_scale(target_img = filelist[0], zp_target = 25, zp_reference = 25, zp_key = 'ZP_AUTO', print_output = True)
    # Data-based running with the header
    target_img = fits.getdata(filelist[0])
    target_header = fits.getheader(filelist[0])
    scaled_data, scaled_header = A.img_scale(target_img = target_img, target_header = target_header, zp_target = 25, zp_reference = 25, zp_key = 'ZP_AUTO', print_output = True)
    fits.writeto(Path(filelist[0]).with_name('scaled_data_based.fits'), scaled_data, scaled_header, overwrite = True)

    # Data-based running without the header
    target_img = fits.getdata(filelist[0])
    scaled_data = A.img_scale(target_img = target_img, zp_target = 27, zp_reference = 25, zp_key = 'ZP_AUTO', print_output = True)
    fits.writeto(Path(filelist[0]).with_name('scaled_data_based_nohead.fits'), scaled_data, overwrite = True)

#%% convolution
if __name__ == '__main__':
    filelist = list(Path('/home/hhchoi1022/data/test/convolve/').glob('*.fits'))
    # Path-based running
    convolved_file = A.img_convolve(target_img = filelist[0], fwhm_target = 3, fwhm_reference = 3.95, output_path = None, print_output = True)
    # Data-based running
    target_img = fits.getdata(filelist[0])
    target_header = fits.getheader(filelist[0])
    convolved_data, convolved_header = A.img_convolve(target_img = target_img, target_header = target_header, fwhm_reference = 3.95, output_path = None, print_output = True)
    fits.writeto(Path(filelist[0]).with_name('convolved_data_based.fits'), convolved_data, header = convolved_header, overwrite = True)
    convolved_data = A.img_convolve(target_img = target_img, fwhm_target = 3, fwhm_reference = 3.95, output_path = None, print_output = True)
    fits.writeto(Path(filelist[0]).with_name('convolved_data_based_nopix.fits'), convolved_data, overwrite = True)
    # Check convolution is applied well
    import matplotlib.pyplot as plt
    convolved_file = Path(filelist[0]).with_name('convolved_data_based.fits')
    tbl = A.run_sextractor(target_img = filelist[0], sex_configfile = A.get_sexconfigpath(telescope = '7DT', ccd = 'C361K', readoutmode = 'HIGH', binning = 1, for_psfex = False))
    peeing_sources = tbl[(tbl['MAG_APER'] > -13) & (tbl['MAG_APER'] < -10) & (tbl['CLASS_STAR'] > 0.9)]
    plt.scatter(tbl['MAG_APER'], tbl['FWHM_IMAGE'], label = f'Peeing= {np.mean(peeing_sources["FWHM_IMAGE"]):.2f}')
    
    tbl = A.run_sextractor(target_img = convolved_file, sex_configfile = A.get_sexconfigpath(telescope = '7DT', ccd = 'C361K', readoutmode = 'HIGH', binning = 1, for_psfex = False))
    peeing_sources = tbl[(tbl['MAG_APER'] > -13) & (tbl['MAG_APER'] < -10) & (tbl['CLASS_STAR'] > 0.9)]
    plt.scatter(tbl['MAG_APER'], tbl['FWHM_IMAGE'], label = f'Peeing= {np.mean(peeing_sources["FWHM_IMAGE"]):.2f}')
    plt.legend()
    
#%% crdetection
if __name__ == '__main__':
    filelist = list(Path('/home/hhchoi1022/data/test/crdetection/').glob('*.fits'))
    # Path-based running
    telinfo = A.get_telinfo('7DT', 'C361K', 'HIGH', 1)
    A.img_crdetection(target_img = filelist[0], gain = telinfo['gain'], readnoise = telinfo['readnoise'], output_path = None, print_output = True)
    # Data-based running
    target_img = fits.getdata(filelist[0])
    target_header = fits.getheader(filelist[0])
    crdetection_data, crdetection_header = A.img_crdetection(target_img = target_img, target_header = target_header, gain = telinfo['gain'], readnoise = telinfo['readnoise'], output_path = None, print_output = True)
    fits.writeto(Path(filelist[0]).with_name('crdetection_data_based.fits'), crdetection_data, header = crdetection_header, overwrite = True)
    # Data-based running without header
    target_img = fits.getdata(filelist[0])
    crdetection_data = A.img_crdetection(target_img = target_img, gain = telinfo['gain'], readnoise = telinfo['readnoise'], output_path = None, print_output = True)
    fits.writeto(Path(filelist[0]).with_name('crdetection_data_based_nohead.fits'), crdetection_data, overwrite = True)
#%% subbkg
if __name__ == '__main__':
    filelist = list(Path('/home/hhchoi1022/data/test/subbkg/').glob('*.fits'))
    # Path-based running
    ## 1D bkg with no masking
    A.img_subtractbkg(target_img = filelist[0], apply_2D_bkg = False, mask_sources = False, use_header = False, visualize = True, output_path = Path(filelist[0]).with_name('subbkg_1D_30nomask.fits'), print_output = True, save_bkgmap = True)
    ## 1D bkg with masking
    A.img_subtractbkg(target_img = filelist[0], apply_2D_bkg = False, mask_sources = True, use_header = False, visualize = True, output_path = Path(filelist[0]).with_name('subbkg_1D_30mask.fits'), print_output = True, save_bkgmap = True)
    ## 2D bkg with no masking
    A.img_subtractbkg(target_img = filelist[0], apply_2D_bkg = True, mask_sources = False, use_header = False, visualize = True, output_path = Path(filelist[0]).with_name('subbkg_2D_30nomask.fits'), print_output = True, save_bkgmap = True)
    ## 2D bkg with masking
    A.img_subtractbkg(target_img = filelist[0], apply_2D_bkg = True, mask_sources = True, use_header = False, visualize = True, output_path = Path(filelist[0]).with_name('subbkg_2D_30mask.fits'), print_output = True, save_bkgmap = True)

    # data-based running
    target_img = fits.getdata(filelist[0])
    target_header = fits.getheader(filelist[0])
    ## 2D bkg with masking
    subbkg_data, subbkg_header, bkg_map = A.img_subtractbkg(target_img = target_img, target_header = target_header, apply_2D_bkg = True, mask_sources = True, use_header = False, visualize = True, output_path = None, print_output = True, save_bkgmap = True)
    fits.writeto(Path(filelist[0]).with_name('subbkg_2D_data_based_mask.fits'), subbkg_data, header = subbkg_header, overwrite = True)
    fits.writeto(Path(filelist[0]).with_name('subbkg_2D_data_based_mask_bkgmap.fits'), bkg_map, target_header, overwrite = True)
#%% combine
if __name__ == '__main__':
    filelist = list(Path('/home/hhchoi1022/data/test/combine/').glob('*20250104*100.fits'))
    # Path-based running
    A.img_combine(filelist = filelist, subbkg = True, apply_2D_bkg = True, mask_sources = True, mask_source_size_in_pixel = 30, zp_reference = None, align = True)
    # Data-based running
    A.img_combine(filelist = filelist, )
    
#%% PSFex
if __name__ == '__main__':
    filelist = list(Path('/home/hhchoi1022/data/test/psfex/').glob('*.fits'))
    # Path-based running
    A.run_psfex(target_img = filelist[0], sex_configfile = psfex_sexconfigfile, psfex_configfile = psfex_configfile)
#%% hotpants
if __name__ == '__main__':
    filelist = list(Path('/home/hhchoi1022/data/test/subtract/').glob('calib*.com.fits'))
    masklist = list(Path('/home/hhchoi1022/data/test/subtract/').glob('calib*.mask.fits'))
    stamplist = list(Path('/home/hhchoi1022/data/test/subtract/').glob('*.ssf.txt'))
    A.run_hotpants(target_img = filelist[0], reference_img = filelist[1], 
                   convolve_path = str(filelist[0]).replace('calib', 'conv_calib'),
                   target_mask = masklist[0], reference_mask = masklist[1],
                   stamp = stamplist[0],
                   subbkg = False, scale = False, align = False, normim = 't', convim = 'i', iu = 60000, il = -3
                   )
    
#%% sextractor
if __name__ == '__main__':
    filelist = list(Path('/home/hhchoi1022/data/test/sextractor/').glob('*com.fits'))
    masklist = list(Path('/home/hhchoi1022/data/test/sextractor/').glob('*.mask.fits'))
    sci_catalog = A.run_sextractor(target_img = filelist[0], sex_configfile = sexconfig, image_mask = masklist[0])
    
#%% Visualization for sextractor
if __name__ == '__main__': 
    from tippy.catalog import Catalog
    cat = Catalog(objname = 'T00176', catalog_type = 'GAIAXP')
    ref_catalog, _ = cat.get_reference_sources(mag_lower = 13, mag_upper = 16)
    ref_catalog_coord = SkyCoord(ra = ref_catalog['ra'], dec = ref_catalog['dec'], unit = 'deg')
    obj_catalog_coord = SkyCoord(ra = sci_catalog['ALPHA_J2000'], dec = sci_catalog['DELTA_J2000'], unit = 'deg')
    matched_object_idx, matched_catalog_idx, _ = A.cross_match(obj_catalog_coord, ref_catalog_coord, 1)
    obj_matched = sci_catalog[matched_object_idx]
    ref_matched = ref_catalog[matched_catalog_idx]
    # Seeing plotplt.figure()
    plt.scatter(sci_catalog['MAG_APER'], sci_catalog['FWHM_IMAGE'], alpha = 0.1, c = 'k')
    plt.scatter(obj_matched['MAG_APER'], obj_matched['FWHM_IMAGE'], c = 'r')
    # ZP plot
    plt.figure()
    zp = obj_matched['MAG_APER'] - ref_matched['r_mag']
    plt.scatter(ref_matched['r_mag'], zp)
#%%
if __name__ == '__main__':
    A.get_imginfo(filelist)
    test_folder = '/home/hhchoi1022/data/test/test4/'
    
    reference_img = os.path.join(test_folder, 'calib_7DT15_T00176_20250227_034830_r_300.com.fits')
    target_img = os.path.join(test_folder, 'calib_7DT15_T00176_20241220_022038_r_300.com.fits')
    target_mask = target_img.replace('.fits', '.mask.fits')
    ssf_path = target_img.replace('.fits', '.ssf.txt')
    
    tgt_dat = fits.getdata(target_img)
    ref_dat = fits.getdata(reference_img)
    tgt_hdr = fits.getheader(target_img)
    ref_hdr = fits.getheader(reference_img)
    
    #tgt_scaled = A.img_scale(target_img = target_img, target_header  =tgt_hdr,  zp_reference = 23)
    #ref_scaled = A.img_scale(target_img = reference_img, target_header = ref_hdr, zp_reference = 23)
    #B = A.img_subtractbkg(target_img = tgt_image, apply_2D_bkg = False, mask_sources = True, visualize = True)
    psfexconfigpath = A.get_psfexconfigpath()#telescope = '7DT', ccd = 'C361K', readoutmode = 'HIGH', binning = 1)
    psfex_config = A.load_config(psfexconfigpath)
    sex_configfile = A.get_sexconfigpath(telescope = '7DT', ccd = 'C361K', readoutmode = 'HIGH', binning = 1, for_psfex = False)
    #convimg=  A.img_convolve(target_img = tgt_scaled,  fwhm_reference = 5)
    #A.run_psfex(target_img = target_img, sex_configfile = psfex_config, psfex_configfile = psfexconfigpath)
    output_hotpants = os.path.join(test_folder, 'sub2.fits')
    output_conv = os.path.join(test_folder, 'conv2.fits')
    # 0
    #A.run_hotpants(target_img = target_img, reference_img = reference_img, output_path = output_hotpants, convolved_img = output_conv, target_mask = None, reference_mask = None, stamp = None, convim = 't')
    # 1 with mask
    #A.run_hotpants(target_img = target_img, reference_img = reference_img, output_path = output_hotpants, convolved_img = output_conv, target_mask = target_mask, reference_mask = None, stamp = None, convim = 't')
    # 2 with stamp
    #A.run_hotpants(target_img = target_img, reference_img = reference_img, output_path = output_hotpants, convolved_img = output_conv, target_mask = None, reference_mask = None, stamp = ssf_path, convim = 't')
    
    # Test SCAMP
    scampconfig = A.get_scampconfigpath()
    sex_configfile = A.get_sexconfigpath(telescope = '7DT', ccd = 'C361K', readoutmode = 'HIGH', binning = 1, for_psfex = False, for_scamp = True)
    #A.run_scamp(target_img = target_img, sex_configfile = sexampconfig, scamp_configfile = scampconfig)
    
    

    #A.run_astrometry(image = target_img, sex_configfile = sexconfig)
    #B = A.img_subtractbkg(imlist[0])
    #A.img_scale(target_img = B, zp_target = 25, zp_reference = 25, zp_key = 'ZP_AUTO', print_output = True)
    # A.img_combine(
    #     imlist,
    #     output_path = None, #'/home/hhchoi1022/data/test/test.fits',
    #     combine_method = 'median', 
    #     clip = 'extrema', 
    #     clip_sigma_low = 2,
    #     clip_sigma_high = 5,
    #     clip_minmax_min = 3,
    #     clip_minmax_max = 3,
    #     clip_extrema_nlow = 1,
    #     clip_extrema_nhigh = 1,
    #     subbkg = True,
    #     apply_2D_bkg = False,
    #     bkg_key = 'SKYBKG',
    #     zp_key = 'ZP_AUTO',
    #     mask_sources = False,
    #     mask_source_size_in_pixel = 10,
    #     bkg_estimator = 'median',
    #     sigma = 5.0,
    #     box_size = 100,
    #     filter_size = 3,
    #     scale = True,
    #     zp_reference = None,#25.0,
    #     print_output = True
    # )

    #A.run_sextractor(image = '/mnt/data1/7DT/calib_7DT02_S240422ed_20240423_013036_r_120.fits', sex_configfile = sexconfigpath)
    # file_ = '/data1/supernova_rawdata/SN2023rve/analysis/RASA36/reference_image/com_align_Calib-RASA36-NGC1097-20210719-091118-r-60.fits'
    # #file1 = '/data1/supernova_rawdata/SN2023rve/analysis/KCT_STX16803/r/align_com_align_cutoutmost_Calib-KCT_STX16803-NGC1097-20230801-075323-r-120.fits'
    # #file2 = '/data1/supernova_rawdata/SN2023rve/analysis/KCT_STX16803/reference_image/cut_Ref-KCT_STX16803-NGC1097-r-5400.com.fits'
    # sex_configfile = '/home/hhchoi1022/hhpy/Research/photometry/sextractor/RASA36_HIGH.scampconfig'
    # filelist = glob.glob('/mnt/data1/supernova_rawdata/SN2023rve/analysis/RASA36/reference_image_tmp/cutout*.fits')
    # #sex_configfile = '/home/hhchoi1022/hhpy/Research/photometry/sextractor/KCT.config'
    # #A.run_astrometry(image = file1, sex_configfile = sex_configfile)
    # #A.run_scamp(filelist = file_, sex_configfile = sex_configfile)
    
    # #sex_params = dict()
    # #sex_params['CATALOG_NAME'] = f"{A.scamppath}/catalog/{os.path.basename(file_).split('.')[0]}.cat"
    # #sex_params['PARAMETERS_NAME'] = f'{A.sexpath}/scamp.param'
    # target_img = glob.glob('/mnt/data1/supernova_rawdata/SN2023rve/analysis/KCT_STX16803/g/Calib*.fits')[10]
    # reference_img = '/mnt/data1/supernova_rawdata/SN2023rve/analysis/KCT_STX16803/g/Calib-KCT_STX16803-NGC1097-20230927-063834-g-120.fits'
    # A.visualize_image(target_img)
    # A.visualize_image(reference_img)
    #A.align_img(target_img, reference_img)

# %%
