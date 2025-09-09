# %%
# Standard library
import os
import re
import shutil
import inspect
import warnings
import subprocess
import signal
import functools
from datetime import datetime
from pathlib import Path
from typing import List, Union, Optional, Tuple

# External libraries
import numpy as np
from pympler import asizeof

# Astropy
import astropy.units as u
import astropy.io.fits as fits
from astropy.io.fits import Header
from astropy.io import ascii
from astropy.time import Time
from astropy.table import Table, Row
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.convolution import convolve, Gaussian2DKernel
# Custom config
from ezphot.configuration import Configuration

# Suppress all warnings
warnings.filterwarnings('ignore')


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

class PhotometryHelper(Configuration):

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
        cls = self.__class__
        methods = [
            f'{cls.__name__}.{name}()\n'
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction)
            if not name.startswith('_') and method.__qualname__.startswith(cls.__name__)
        ]
        return '[Methods]\n' + ''.join(methods)

    def print(self, string, do_print : bool = False):
        print(string) if do_print else None
        
    # Load information
    def load_fits(self,
                  target_img: Union[str, Path, np.ndarray]):
        if isinstance(target_img, (str, Path)):
            target_img = Path(target_img)
            with fits.open(target_img) as hdul:
                target_data = hdul[0].data
                target_header = hdul[0].header
            return target_data, target_header
        elif isinstance(target_img, np.ndarray):
            target_data = target_img
            return target_data, None
        else:
            raise TypeError("target_img must be a Path, string, or NumPy array.")
    
    def get_imginfo(self, 
                    filelist: Union[List[str], List[Path]],
                    pattern: Optional[str] = '*.fits',
                    keywords: Optional[List[str]] = None,
                    normalize_key: bool = True,
                    verbose: bool = True) -> Table:
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
            coll = ImageFileCollection(location=directory, glob_include= pattern)
            if len(coll.files) > 0:
                self.print(f"Loaded {len(coll.files)} FITS files from {directory}", verbose)

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
                self.print(f"Warning: No FITS files found in {directory}", verbose)

        # Filter to ensure only filelist rows are returned
        filelist_inputted = filelist
        filelist_queried = np.array([Path(f) for f in all_coll['file']])
        all_coll = all_coll[np.isin(filelist_queried, filelist_inputted)]

        # Sort the final result to match the original filelist order
        file_order = {str(f): i for i, f in enumerate(filelist)}
        sort_idx = np.argsort([file_order.get(f, float('inf')) for f in all_coll['file']])
        all_coll = all_coll[sort_idx]

        # Check final count of combined FITS files
        self.print(f"Total FITS files combined: {len(all_coll)}", verbose)
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


    def merge_header(self, 
                    new_header, 
                    original_header, 
                    include_keys=['*'], 
                    exclude_keys=['PV*']):
        """
        Merge original header metadata into a new header, preserving WCS from the new header.

        Parameters
        ----------
        new_header : astropy.io.fits.Header
            Header from the reprojected (SWarped) image (valid WCS).
        original_header : astropy.io.fits.Header
            Original image header with instrument metadata.
        include_keys : list of str, optional
            List of keys to include from the original header. Wildcards like '*' are supported.
            Default is ['*'] (include all).
        exclude_keys : list of str, optional
            List of keys to exclude from the original header. Wildcards like '*' are supported.
            Default is [''] (exclude none).

        Returns
        -------
        merged_header : astropy.io.fits.Header
        """
        from astropy.wcs import WCS
        from fnmatch import fnmatch

        # WCS keywords to preserve from reprojected header
        wcs_keys = WCS(new_header).to_header(relax=True).keys()

        # Start from a copy of new_header (with valid WCS)
        merged_header = new_header.copy()

        # Check if a key should be included
        def should_include(key):
            return (any(fnmatch(key, pattern) for pattern in include_keys) and
                    not any(fnmatch(key, pattern) for pattern in exclude_keys))

        # Add non-WCS original keywords
        for key, val in original_header.items():
            if (key not in wcs_keys and key not in merged_header and should_include(key)):
                if original_header[key] is not None:
                    merged_header[key] = val

        return merged_header

    def estimate_telinfo(self, 
                         path: Union[str, Path],
                         update_header: bool = True):
        path = Path(path)
        # For 7DT
        if '7DT' in str(path):
            header = fits.getheader(path)
            telescope = '7DT'
            if header['NAXIS1'] < 13000:
                ccd = 'C361K'
            else:
                ccd = 'C516k'
            if 'GAIN' in header.keys():
                if str(header['GAIN']) == '2750':
                    readoutmode = 'HIGH'
                else:
                    readoutmode = 'LOW'
            else:
                readoutmode = 'HIGH'
                
            if 'XBINNING' in header.keys():
                if str(header['XBINNING']) == '1':
                    binning = 1
                else:
                    binning = 2
            else:
                binning = 1
        # For KCT
        elif 'KCT' in str(path):
            header = fits.getheader(path)
            telescope = 'KCT'
            if 'INSTRUME' in header.keys():
                if 'STX' in header['INSTRUME']:
                    ccd = 'STX16803'
                else:
                    ccd = 'ASI1600MM'
            readoutmode = None
            binning = 1
        
        # For RASA36
        elif 'RASA' in str(path):
            telescope = 'RASA36'
            ccd = 'KL4040'
            binning = 1
            
            ############### ADD HERE FOR IDENTIFYING RASA36 RNMODE ###############
            with fits.open(path, mode = 'update') as hdul:
                header = hdul[0].header
                if 'CAMMODE' in header.keys():
                    if header['CAMMODE'] == 'HIGH':
                        readoutmode = 'HIGH'
                    else:
                        readoutmode = 'MERGE'
                else:
                    data = hdul[0].data
                    zero = np.mean(np.sort((data[~np.isnan(data)].flatten()))[-10:-5])
                    if zero > 80000:
                        readoutmode = 'MERGE'
                    else:
                        readoutmode = 'HIGH'                

                    if update_header:
                        header['CAMMODE'] = readoutmode
        
        elif ('LSGT' in str(path)) or ('sophia' in str(path).lower()):
            telescope = 'LSGT'
            with fits.open(path, mode = 'update') as hdul:
                header = hdul[0].header
                if 'INSTRUME' in header.keys():
                    if 'andor' in header['INSTRUME'].lower():
                        ccd = 'SNUCAMII'
                    else:
                        ccd = 'ASI1600MM'
            readoutmode = None
            binning = 1
            
        elif ('CBNUO' in str(path)):
            telescope = 'CBNUO'
            ccd = 'STX16803'
            binning = 1
            readoutmode = 'None'

                
        telinfo = self.get_telinfo(telescope = telescope, ccd = ccd, readoutmode = readoutmode, binning = binning)
        return telinfo

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
            obsinfo_file = self.config['OBSERVATORY_TELESCOPEINFO']

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
        elif len(obs_info) == 1:
            return obs_info[0]

        # Select CCD if not provided and multiple options exist
        if ccd is None and len(obs_info[key_ccd]) > 1:
            ccd = prompt_choice(set(obs_info[key_ccd]), "Multiple CCDs found. Choose one")
        obs_info = filter_by_column(obs_info, key_ccd, ccd)

        # Select readout mode if not provided and multiple options exist
        if readoutmode is None and len(obs_info[key_mode]) > 1:
            readoutmode = prompt_choice(set(obs_info[key_mode]), "Multiple modes found. Choose one")
        obs_info = filter_by_column(obs_info, key_mode, readoutmode)
        if len(obs_info) == 1:
            return obs_info[0]

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

    def flexible_time_parser(self, value) -> Time:
        # If already a Time instance
        if isinstance(value, Time):
            return value

        # If datetime object
        if isinstance(value, datetime):
            return Time(value)

        # Convert to string
        value = str(value).strip()

        # Handle raw digits: YYMMDD or YYYYMMDD
        if value.isdigit():
            if len(value) == 6:  # YYMMDD
                value = '20' + value
            if len(value) == 8:  # YYYYMMDD
                value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
            return Time(value, format='iso')

        # Handle ISO or ISO with time
        try:
            return Time(value, format='iso', scale='utc')
        except ValueError:
            pass

        # Fallback to automatic parsing
        return Time(value)

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

    def to_native(self, value):
        """Ensure array is a NumPy array with native byte order. Return as-is if not an array."""
        if not isinstance(value, np.ndarray):
            return value
        return value.astype(value.dtype.newbyteorder('=')) if value.dtype.byteorder not in ('=', '|') else value

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

    def report_memory(self, threshold_mb: float = 1.0, sort: bool = True, top_n: int = 20):
        frame = inspect.currentframe().f_back
        var_sizes = []

        for name, val in frame.f_locals.items():
            try:
                size_mb = asizeof.asizeof(val) / 1024**2
                if size_mb > threshold_mb:
                    var_sizes.append((name, size_mb))
            except Exception as e:
                print(e)

        if sort:
            var_sizes.sort(key=lambda x: -x[1])

        print(f"[Memory usage by variable > {threshold_mb} MB]")
        for name, size in var_sizes[:top_n]:
            print(f"{name}: {size:.2f} MB")

    def report_memory_process(self):
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 ** 2)
        print(f"[MEMORY REPORT] Memory usage: {mem:.2f} MB")

    def cross_match(self, 
                    obj_coords: SkyCoord, 
                    sky_coords: SkyCoord, 
                    max_distance_second: float = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast cross-match using vectorized operations.
        """
        from astropy.coordinates import match_coordinates_sky
        import numpy as np

        # Match coordinates
        closest_ids, closest_dists, _ = match_coordinates_sky(obj_coords, sky_coords)

        # Convert distance threshold to degrees (SkyCoord uses degrees internally)
        max_distance_deg = max_distance_second / 3600.0

        # Vectorized masking (no Python loops)
        matched_mask = closest_dists.value < max_distance_deg

        # Extract indices
        matched_object_idx = np.where(matched_mask)[0]
        matched_catalog_idx = closest_ids[matched_mask]
        no_matched_object_idx = np.where(~matched_mask)[0]

        return matched_object_idx, matched_catalog_idx, no_matched_object_idx


    def group_table(self, tbl: Table, key: str, tolerance: float = 0.1):
        """
        Group rows in the table where values of the specified key are within a given tolerance.

        Parameters
        ----------
        tbl : Table
            The input astropy table.
        key : str
            The column name to group by, using value proximity within tolerance.
        tolerance : float
            The maximum difference between values to consider them in the same group.

        Returns
        -------
        Table
            Table with an additional 'group' column indicating group ID.
        """
        import numpy as np
        from astropy.table import Table, vstack
        import pandas as pd
        table = tbl.copy()
        table[key] = pd.to_numeric(table[key], errors='coerce')
        table.sort(key)  # Sort by key to make grouping faster
        group_ids = np.full(len(table), -1, dtype=int)

        current_group = 0
        i = 0

        while np.any(group_ids == -1):
            idx_unassigned = np.where(group_ids == -1)[0]
            ref_idx = idx_unassigned[0]
            ref_val = table[key][ref_idx]

            # Assign all unassigned rows close to ref_val
            close_idx = idx_unassigned[np.abs(table[key][idx_unassigned] - ref_val) < tolerance]
            group_ids[close_idx] = current_group

            current_group += 1
            i += 1

        table['group'] = group_ids
        return table.group_by('group')

    def match_table(self, 
                    tbl1: Table, 
                    tbl2: Table, 
                    key1: str,
                    key2: str, 
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
            ol_idx = (np.abs(obs[key1] - tbl2[key2]) < tolerance)
            if True in ol_idx:
                closest_idx = np.argmin(np.abs(obs[key1]-tbl2[key2]))
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

    def is_wcs_equal(self, wcs1, wcs2, tolerance = 1e-4, check_sip=False):
        """
        Compare whether two WCS headers represent the same projection.

        Parameters
        ----------
        header1, header2 : astropy.io.fits.Header
            FITS headers to compare.
        rtol : float
            Relative tolerance for comparing matrix values.
        atol : float
            Absolute tolerance for comparing CRVAL and CRPIX.
        check_sip : bool
            Whether to compare SIP distortion terms.

        Returns
        -------
        bool
            True if WCS match within tolerances.
        """

        # Check CRVAL (sky center)
        if not np.allclose(wcs1.wcs.crval, wcs2.wcs.crval, atol=tolerance):
            return False

        # Check CRPIX (reference pixel)
        if not np.allclose(wcs1.wcs.crpix, wcs2.wcs.crpix, atol=tolerance):
            return False

        # Check PC/CD matrix
        if wcs1.wcs.has_pc() and wcs2.wcs.has_pc():
            matrix1 = wcs1.wcs.pc
            matrix2 = wcs2.wcs.pc
        elif wcs1.wcs.has_cd() and wcs2.wcs.has_cd():
            matrix1 = wcs1.wcs.cd
            matrix2 = wcs2.wcs.cd
        else:
            return False  # Incompatible or missing projection matrix

        if not np.allclose(matrix1, matrix2, atol=tolerance):
            return False

        # # Check projection type
        # if wcs1.wcs.ctype != wcs2.wcs.ctype:
        #     return False

        # Check pixel scale
        scale1 = proj_plane_pixel_scales(wcs1)
        scale2 = proj_plane_pixel_scales(wcs2)
        if not np.allclose(scale1, scale2, atol=tolerance):
            return False

        # Check SIP distortion terms
        if check_sip:
            sip1 = getattr(wcs1.sip, 'a', None)
            sip2 = getattr(wcs2.sip, 'a', None)
            if (sip1 is not None) or (sip2 is not None):
                if (sip1 is None) or (sip2 is None):
                    return False
                if not np.allclose(sip1, sip2, atol=tolerance):
                    return False
        return True
        
    def img_astroalign(self, 
                       target_img: Union[str, Path, np.ndarray], 
                       reference_img: Union[str, Path, np.ndarray], 
                       target_header: Optional[Header] = None, 
                       reference_header: Optional[Header] = None, 
                       target_outpath: Optional[str] = None,
                       detection_sigma: float = 5.0, 
                       verbose: bool = True):

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
        6. verbose : bool
                        If True, prints progress messages (default: True)

        returns
        ----------
        1. If target_img is str:
            target_outpath : str
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

        self.print('Start image alignment... \n', verbose)

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
        wcs_hdr = reference_wcs.to_header(relax=False)
        for key in ['DATE-OBS', 'MJD-OBS', 'RADESYS', 'EQUINOX']:
            wcs_hdr.remove(key, ignore_missing=True)
        target_header.update(wcs_hdr)
        
        if reference_wcs.wcs.has_pc():
            # Use PC matrix if available
            linear_matrix = reference_wcs.wcs.pc
        elif reference_wcs.wcs.has_cd():
            # Use CD matrix if PC is not available
            linear_matrix = reference_wcs.wcs.cd
        else:
            # Fallback to CDELT and CRPIX if no CD matrix is available
            linear_matrix = np.array([[reference_wcs.wcs.cdelt[0], 0],
                                        [0, reference_wcs.wcs.cdelt[1]]])
        # Safely update header with PC matrix, explicitly set even 0s
        target_header['PC1_1'] = linear_matrix[0, 0]
        target_header['PC1_2'] = linear_matrix[0, 1]
        target_header['PC2_1'] = linear_matrix[1, 0]
        target_header['PC2_2'] = linear_matrix[1, 1]
            
        target_header['CD1_1'] = linear_matrix[0, 0]
        target_header['CD1_2'] = linear_matrix[0, 1]
        target_header['CD2_1'] = linear_matrix[1, 0]
        target_header['CD2_2'] = linear_matrix[1, 1]
        
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

                if not target_outpath:
                    target_outpath = target_img.parent / f'align_{target_img.name}'
                os.makedirs(target_outpath.parent, exist_ok=True)
                fits.writeto(target_outpath, aligned_target.data, aligned_target.header, overwrite=True)

                self.print('Image alignment complete \n', verbose)
                return str(target_outpath)
            else:
                # Return the aligned data and header
                aligned_header = target_header.copy()
                aligned_header['NAXIS1'] = aligned_data.shape[1]
                aligned_header['NAXIS2'] = aligned_data.shape[0]
                aligned_header['ALIGN'] = (True, 'Image has been aligned.')
                aligned_header['ALIGTIME'] = (Time.now().isot, 'Time of alignment operation.')
                aligned_header['ALIGFILE'] = ('Array input', 'Original data was an array')
                aligned_header['ALIGREF'] = ('Array input', 'Reference data was an array')

                self.print('Image alignment complete \n', verbose)
                return aligned_data, aligned_header, footprint

        except Exception as e:
            self.print('Failed to align the image. Check the image quality and the detection_sigma value.', verbose)
            raise e

    def img_convolve(self,
                    target_img: Union[str, Path, np.ndarray],
                    input_type: str = 'image',
                    kernel: str = 'gaussian',
                    target_header: Optional[Header] = None,
                    target_outpath: Optional[str] = None,
                    fwhm_target: Optional[float] = None,
                    fwhm_reference: Optional[float] = None,
                    fwhm_key: str = 'PEEING',
                    verbose: bool = True):
        """
        Convolve an image or error map with a Gaussian kernel to match target seeing.

        Parameters
        ----------
        target_img : str, Path, or np.ndarray
            Path to the FITS file or image data as a NumPy array.
        kernel : str
            Type of kernel to use ('gaussian').
        target_header : astropy.io.fits.Header, optional
            Header of the image (required if target_img is a NumPy array).
        target_outpath : str, optional
            Path to save the convolved FITS file.
        fwhm_target : float, optional
            Original FWHM of the image.
        fwhm_reference : float
            Desired FWHM after convolution.
        fwhm_key : str
            Header keyword used to read FWHM from FITS header.
        input_type : 'image' or 'error'
            Determines whether to convolve an image or an error map.
        verbose : bool
            Print progress messages.

        Returns
        -------
        str or (np.ndarray, astropy.io.fits.Header) or np.ndarray
            Convolved image or path depending on input.
        """

        self.print(f'Start convolution...', verbose)

        # Load image data and header
        if isinstance(target_img, (str, Path)):
            target_img = Path(target_img)
            if not target_img.is_file():
                raise FileNotFoundError(f"File {target_img} does not exist.")
            with fits.open(target_img) as hdul:
                data = hdul[0].data
                header = hdul[0].header

            if fwhm_target is None:
                if fwhm_key in header:
                    fwhm_target = float(header[fwhm_key])
                else:
                    raise ValueError(f"FWHM not found in header using key '{fwhm_key}', and 'fwhm_target' is not provided.")

        elif isinstance(target_img, np.ndarray):
            data = target_img
            if target_header is not None:
                header = target_header
                if fwhm_target is None:
                    if fwhm_key in header:
                        fwhm_target = float(header[fwhm_key])
                    else:
                        raise ValueError(f"{fwhm_key} not found in target_header and 'fwhm_target' is not provided.")
            else:
                header = None
        else:
            raise TypeError("target_img must be either a Path, string (FITS file path), or a NumPy array.")

        # Create Gaussian kernel
        if isinstance(kernel, str):
            if kernel.lower() == 'gaussian':
                sigma_tgt = fwhm_target / 2.355
                sigma_ref = fwhm_reference / 2.355
                diff_fwhm = fwhm_reference - fwhm_target

                if diff_fwhm < 0.1:
                    self.print(f"FWHM difference ({diff_fwhm:.3f}) is small; applying minimal smoothing kernel (?=0.1)", verbose)
                    sigma_conv = 0.1
                else:
                    sigma_conv = np.sqrt(sigma_ref**2 - sigma_tgt**2)

                self.print(f"Calculated convolution sigma: {sigma_conv:.6f} (method: {kernel})", verbose)
                kernel = Gaussian2DKernel(sigma_conv)
            else:
                raise ValueError(f"Unsupported convolution kernel '{kernel}'. Only 'gaussian' is supported.")
        else:
            # user-defined kernel
            kernel = kernel  # already given as array

        # Perform convolution
        if input_type == 'image':
            convolved_image = convolve(data, kernel, normalize_kernel=True)
        elif input_type == 'error':
            kernel_array = kernel.array if isinstance(kernel, Gaussian2DKernel) else kernel
            var = data**2
            var_convolved = convolve(var, kernel_array**2, normalize_kernel=True) ## kernel_array or kernel_array**2 ???
            convolved_image = np.sqrt(var_convolved)
            self.print("Error map convolved using variance propagation.", verbose)
        else:
            raise ValueError("input_type must be either 'image' or 'error'.")

        # Return result
        if isinstance(target_img, Path):
            if not target_outpath:
                target_outpath = target_img.parent / f'conv_{target_img.name}'
            os.makedirs(target_outpath.parent, exist_ok=True)

            # Update header
            header['CONVOLVE'] = (True, 'Image has been convolved.')
            header['CONVTIME'] = (Time.now().isot, 'Time of convolution operation.')
            header['CONVFILE'] = (str(target_img), 'Original file path before convolution')

            hdu = fits.PrimaryHDU(convolved_image, header=header)
            hdu.writeto(target_outpath, overwrite=True)

            self.print(f"Image convolution complete. Output: {target_outpath}", verbose)
            return str(target_outpath), None

        elif isinstance(target_img, np.ndarray):
            if target_header is not None:
                self.print('Image convolution complete with header.\n', verbose)
                return convolved_image, target_header
            else:
                self.print('Image convolution complete.\n', verbose)
                return convolved_image, None

    def run_psfex(self, 
                  target_path: Union[str, Path], 
                  psfex_sexconfigfile: Union[str, Path], 
                  psfex_configfile: Union[str, Path],                   
                  psfex_sexparams: Optional[dict] = None,
                  psfex_params: Optional[dict] = None,
                  target_outpath: Optional[Union[str, Path]] = None,
                  verbose: bool = True) -> None:

        """
        Run SExtractor followed by PSFEx on the specified image using the provided configuration and parameters.
        """
        from pathlib import Path
        import os
        import subprocess
        import datetime

        self.print('Start PSFEx process...=====================', verbose)
        current_dir = Path.cwd()
        target_path = Path(target_path)
        psfex_config_path = Path(psfex_configfile)

        # Run SExtractor
        sex_result, output_file, _, _ = self.run_sextractor(
            target_path=target_path, 
            sex_configfile=psfex_sexconfigfile, 
            sex_params=psfex_sexparams, 
            return_result=False, 
            verbose=False)

        # Load default PSFEx config
        all_params = self.load_config(psfex_config_path)

        # Set up history directory for outputs
        if not target_outpath:
            target_outpath = target_path.parent
        if target_outpath.is_file():
            target_outpath = target_outpath.parent
        
        # Handle CHECKIMAGE_NAME parameter
        if not psfex_params:
            psfex_params = dict

        # Check image
        psfex_params['CHECKIMAGE_NAME'] = psfex_params.get('CHECKIMAGE_NAME') or all_params.get('CHECKIMAGE_NAME', '')
        abspath_fits_files = []
        checkimage_filelist = psfex_params['CHECKIMAGE_NAME'].split(',')
        for filename in checkimage_filelist:
            file_ = Path(filename)
            output_file_path = Path(output_file)
            abspath = output_file_path.parent / (file_.stem + '_' + output_file_path.stem + file_.suffix)
            abspath_fits_files.append(abspath)
            
        # Build PSFEx parameter string
        psfexparams_str = ' '.join([f"-{key} {value}" for key, value in psfex_params.items()])

        command = f"psfex {output_file} -c {psfex_configfile} {psfexparams_str}"

        try:
            os.chdir(target_path.parent)
            self.print(f'RUN COMMAND: {command}', verbose)
            result = subprocess.run(command, shell=True, check=True, 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if verbose:
                self.print(result.stdout.decode(), verbose)
                self.print(result.stderr.decode(), verbose)

            self.print("PSFEx process finished=====================", verbose)
            return abspath_fits_files
        except subprocess.CalledProcessError as e:
            self.print(f"Error during PSFEx execution: {e.stderr.decode()}", verbose)
            return None
        finally:
            os.chdir(current_dir)  # Ensure directory is reset
    
    def run_hotpants(self,
                     target_path: Union[str, Path],
                     reference_path: Union[str, Path],
                     convolve_path: Optional[Union[str, Path]] = None,
                     target_mask: Optional[Union[str, Path]] = None,
                     reference_mask: Optional[Union[str, Path]] = None,
                     stamp: Optional[Union[str, Path]] = None,
                     target_outpath: Optional[Union[str, Path]] = None,

                     # Hotpants config
                     verbose: bool = True,
                     
                     convim: str = 't',
                     normim: str = 'i',
                     nrx: int = 3,
                     nry: int = 2,
                     iu: float = 60000,
                     il: float = -10000,
                     tu: float = 60000,
                     tl: float = -10000,
                     **hotpants_kwargs) -> str:
        """
        Run Hotpants for image subtraction.
        """
        from pathlib import Path
        import subprocess

        target_path = Path(target_path)
        reference_path = Path(reference_path)
        current_dir = Path.cwd()

        if not target_path.is_file():
            raise FileNotFoundError(f"Target image {target_path} does not exist.")
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference image {reference_path} does not exist.")
        
        if not target_outpath:
            target_outpath = target_path.parent / f'sub_{target_path.name}'
        else:
            target_outpath = Path(target_outpath)

        self.print(f'Starting image subtraction with hotpants on {target_path.name}...', verbose)

        command = ['hotpants']

        # Required base options
        command.extend([
            '-c', convim,
            '-n', normim,
            '-inim', str(target_path),
            '-tmplim', str(reference_path),
            '-outim', str(target_outpath),
            '-nrx', str(nrx),
            '-nry', str(nry),
            '-iu', str(iu),
            '-il', str(il),
            '-tu', str(tu),
            '-tl', str(tl),
        ])

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

        # Additional hotpants kwargs
        for key, value in hotpants_kwargs.items():
            # Only include keys that look like valid flags
            command.extend([f'-{key}', str(value)])

        self.print(f"RUN COMMAND: {' '.join(command)}", verbose)

        try:
            result = subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
                timeout=900
            )
            if verbose:
                self.print(result.stdout, verbose)
                self.print(result.stderr, verbose)

            self.print(f"Image subtraction completed successfully. Output saved to {target_outpath}", verbose)
            return str(target_outpath)

        except subprocess.CalledProcessError as e:
            self.print(f"Error during hotpants execution: {e.stderr}", verbose)
            return ""

        except subprocess.TimeoutExpired:
            self.print(f"Hotpants process timed out after 900 seconds.", verbose)
            return ""
        
        except Exception as e:
            self.print(f"An unexpected error occurred: {str(e)}", verbose)
            return ""

    def run_astrometry(self,
                       target_path: Union[str, Path], 
                       astrometry_sexconfigfile: Union[str, Path],
                       ra: Optional[float] = None,
                       dec: Optional[float] = None,
                       radius: Optional[float] = 1,
                       pixelscale: Optional[float] = None,
                       target_outpath: Optional[Union[str, Path]] = None,
                       verbose: bool = True):

        """
        Run the Astrometry.net process to solve WCS coordinates.
        """
        import os
        import subprocess
        import tempfile

        current_dir = Path.cwd()
        target_path = Path(target_path)
        target_dir = target_path.parent
        sexconfig_path = Path(astrometry_sexconfigfile)
        hdr = fits.getheader(target_path)
        if not ra:
            ra_keys = ['RA', 'OBJCTRA', 'CRVAL1']
            for ra_key in ra_keys:
                if ra_key in hdr.keys():
                    ra = hdr[ra_key]
                    self.print(f'RA key found in header: {ra}', verbose)
                    break
        if not dec:
            dec_keys = ['DEC', 'OBJCTDEC', 'CRVAL2']
            for dec_key in dec_keys:
                if dec_key in hdr.keys():
                    dec = hdr[dec_key]
                    self.print(f'DEC key found in header: {dec}', verbose)
                    break

        if not target_path.is_file():
            raise FileNotFoundError(f"Target image {target_path} does not exist.")
        if not sexconfig_path.is_file():
            raise FileNotFoundError(f"SExtractor config file {sexconfig_path} does not exist.")

        try:
            self.print('Start Astrometry process...=====================', verbose)

            with tempfile.TemporaryDirectory(dir = target_dir) as tmpdir:
                tmpdir = Path(tmpdir)
                os.chdir(tmpdir)
                # Copy configs into tmpdir
                for ext in ['.param', '.conv', '.nnw']:
                    for file in Path(self.sexpath).glob(f'*{ext}'):
                        shutil.copy(file, tmpdir)
                        
                # # Set up directories and copy configuration files
                self.print(f'Solving WCS using Astrometry with RA/Dec of {ra}/{dec} and radius of {radius} deg', verbose)

                # Define output path
                if not target_outpath:
                    target_outpath = target_dir / f'astrometry_{target_path.name}'
                else:
                    target_outpath = Path(target_outpath)
                    
                # Rename target_path if target_outpath and target_path are the same
                remove_tmpfile = False
                if target_outpath.resolve() == target_path.resolve():
                    tmp_target_path = target_path.with_name(target_path.stem + '_tmp.fits')
                    target_path.rename(tmp_target_path)  # Rename the actual file
                    target_path = tmp_target_path  # Update reference in script
                    remove_tmpfile = True

                # Build the command
                command = [
                    'solve-field', str(target_path),
                    '--dir', str(tmpdir),
                    '--cpulimit', '300',
                    '--use-source-extractor',
                    '--source-extractor-config', str(sexconfig_path),
                    '--x-column', 'X_IMAGE',
                    '--y-column', 'Y_IMAGE',
                    '--sort-column', 'MAG_AUTO',
                    '--sort-ascending',
                    '--no-remove-lines',
                    '--uniformize', '0',
                    '--no-plots',
                    '--new-fits', str(target_outpath),
                    '--overwrite'
                ]

                if ra is not None and dec is not None:
                    command.extend(['--ra', str(ra), '--dec', str(dec), '--radius', str(radius)])
                if pixelscale:
                    command.extend(['--scale-unit', 'arcsecperpix','--scale-low', str(pixelscale - 0.1), '--scale-high', str(pixelscale + 0.1)])
                
                # Run astrometry with timeout
                command_str = ' '.join(command)
                self.print(f'RUN COMMAND: {command_str}', verbose)
                result = subprocess.run(command, timeout=900, check=True, text=True, capture_output=True)
                if verbose:
                    self.print(result.stdout, verbose)
                    self.print(result.stderr, verbose)
                    
                # Check if the output file was created
                solved_file = tmpdir / Path(target_path).with_suffix('.solved').name              
                if solved_file.is_file():
                    # Check the number of output files
                    #orinum = int(subprocess.check_output("ls C*.fits | wc -l", shell=True).strip())
                    #resnum = int(subprocess.check_output("ls a*.fits | wc -l", shell=True).strip())

                    # Clean up intermediate files
                    #if remove:
                    #    os.system(f'rm -f tmp* *.conv default.nnw *.wcs *.rdls *.corr *.xyls *.solved *.axy *.match check.fits *.param {sexconfig_path.name}')

                    if remove_tmpfile:
                        os.remove(tmp_target_path)
                    self.print('Astrometry process finished=====================', verbose)
                    return True, str(target_outpath)
                else:
                    if remove_tmpfile:
                        tmp_target_path.rename(target_path)
                    self.print('Astrometry process failed=====================', verbose)
                    return False, target_path
            
        except subprocess.TimeoutExpired:
            self.print("The astrometry process exceeded the timeout limit.", verbose)
            return False, target_path
        except subprocess.CalledProcessError as e:
            self.print(f"An error occurred while running the astrometry process: {e}", verbose)
            return False, target_path
        except Exception as e:
            self.print(f"An unknown error occurred while running the astrometry process: {e}", verbose)
            return False, target_path
        finally:
            if 'tmp_target_path' in locals() and tmp_target_path.exists():
                tmp_target_path.rename(target_path)
            os.chdir(current_dir)

    def run_sextractor(self, 
                       target_path: Union[str, Path], 
                       sex_configfile: Union[str, Path], 
                       sex_params: Optional[dict] = None, 
                       
                       # Sextractor parameters
                       target_mask: Optional[Union[str, Path]] = None,
                       target_weight: Optional[Union[str, Path]] = None,
                       weight_type: str = 'MAP_RMS', # BACKGROUND, MAP_RMS, MAP_VAR, MAP_WEIGHT
                       
                       # Optional parameters
                       target_outpath : Optional[Union[str, Path]] = None,
                       return_result: bool = True, 
                       verbose: bool = True):

        """
        Parameters
        ----------
        1. target_path : str
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
        self.print('Start SExtractor process...=====================', verbose)

        # Switch to the SExtractor directory
        target_path = Path(target_path)
        current_path = Path.cwd()
        os.chdir(self.sexpath)
        
        # Load and apply SExtractor parameters
        default_params = self.load_config(sex_configfile)
        if not sex_params:
            sex_params = dict()
        sexparams_str = ''
        if not target_outpath:
            target_outpath = target_path.parent / f"{target_path.stem}.cat"
        sex_params['CATALOG_NAME'] = str(target_outpath)
            
        if target_mask is not None:
            sex_params['FLAG_IMAGE'] = str(target_mask)
            sex_params['PARAMETERS_NAME'] = 'sexflag.param'
            
        if target_weight is not None:
            sex_params['WEIGHT_GAIN'] = 'N'
            sex_params['WEIGHT_IMAGE'] = str(target_weight)
            sex_params['WEIGHT_TYPE'] = str(weight_type)

        for key, value in sex_params.items():
            sexparams_str += f'-{key} {value} '        

        # Command to run SExtractor
        command = f"source-extractor {target_path} -c {sex_configfile} {sexparams_str}"
        #os.makedirs(os.path.dirname(all_params['CATALOG_NAME']), exist_ok=True)
        self.print(f'RUN COMMAND: {command}', verbose)

        try:
            # Run the SExtractor command using subprocess.run
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stderr = result.stderr.decode('utf-8', errors='ignore')

            # Pattern to match background and RMS
            match = re.search(r'Background:\s*([-+eE0-9.]+)\s+RMS:\s*([-+eE0-9.]+)', stderr)
            global_bkgval = match.group(1) if match else None
            global_bkgrms = match.group(2) if match else None

            result_file = Path(target_outpath)
            self.print("SExtractor process finished=====================", verbose)
            self.print(f'Output path: {target_outpath}', verbose)
            if return_result:
                # Read the catalog produced by SExtractor
                sexresult = ascii.read(sex_params['CATALOG_NAME'])
                os.chdir(current_path)
                return True, sexresult, global_bkgval, global_bkgrms
            else:
                os.chdir(current_path)
                return True, sex_params['CATALOG_NAME'], global_bkgval, global_bkgrms
        except Exception as e:
            self.print(f"Error during SExtractor execution: {e}", verbose)
            os.chdir(current_path)
            return False, None, None, None

    def run_scamp(self, 
                  target_path: Union[str, List[str], Path, List[Path]], 
                  scamp_sexconfigfile: Union[str, Path], 
                  scamp_configfile: Union[str, Path],                   
                  scamp_sexparams: Optional[dict] = None,
                  scamp_params: Optional[dict] = None,
                  output_dir: Optional[str] = None,
                  
                  # Other parameters
                  overwrite: bool = True, 
                  verbose: bool = True,
                  ):

        """
                  target_path: Union[str, List[str], Path, List[Path]] = target_path
                  sex_configfile: Union[str, Path] = scamp_sexconfigfile
                  scamp_configfile: Union[str, Path] = scamp_configfile
                  scamp_sexparams: Optional[dict] = None
                  scamp_params: Optional[dict] = None
                  target_mask : Optional[Union[str, Path]] = None
                  target_outpath: Optional[Union[str, Path]] = None
                  update_files: bool = True
                  verbose: bool = True
                  remove : bool = True
        
        Run SCAMP for astrometric calibration on a set of images.
        """
        from pathlib import Path
        import os
        import subprocess
        import re
        from tqdm import tqdm
        from astropy.io import fits

        # Ensure target_path is a list
        if isinstance(target_path, (str, Path)):
            target_path = [Path(target_path)]
        else:
            target_path = [Path(img) for img in target_path]
            
        if not target_path:
            raise ValueError("No valid images provided for SCAMP.")
                    
        if output_dir is None:
            output_dir = target_path[0].parent
        
        self.print(f'Start SCAMP process on {len(target_path)} images...=====================', verbose)

        sex_output_images = {}
        if verbose:
            iterator = tqdm(target_path, desc='Running Source Extractor...')
        else:
            iterator = target_path
        for image in iterator:
            if not image.is_file():
                self.print(f"Warning: Image {image} does not exist. Skipping...", verbose)
                continue
            
            hdr = fits.getheader(image)
            
            if 'CRVAL1' not in hdr:
                raise RuntimeError(f'CRVAL1 or CRVAL2 is not included in the header. Run astrometry first.')
            
            # Ensure sex_params is a dictionary and update it
            scamp_sexparams = scamp_sexparams or {}
            scamp_sexparams.update({
                'PARAMETERS_NAME': str(Path(self.sexpath) / 'scamp.param'),
                'CATALOG_TYPE': 'FITS_LDAC'
            })

            sex_result, output_file, _, _ = self.run_sextractor(
                target_path= image, 
                sex_configfile= scamp_sexconfigfile, 
                sex_params= scamp_sexparams, 
                target_outpath= output_dir / (image.name + ".scamp.cat"),
                return_result= False, 
                verbose= False)
            
            if sex_result:
                sex_output_images[str(image)] = output_file

        # Filter out images that failed
        if not sex_output_images:
            self.print("No valid SExtractor catalogs generated. Aborting SCAMP.", verbose)
            return None

        scamp_output_images = {key: value.replace('.cat', '.head') for key, value in sex_output_images.items()}
        all_images_str = ' '.join(sex_output_images.values())

        # Load and apply SCAMP parameters
        scamp_params = scamp_params or {}
        scampparams_str = ' '.join([f'-{key} {value}' for key, value in scamp_params.items()])

        # SCAMP command
        command = f'scamp {all_images_str} -c {scamp_configfile} {scampparams_str}'

        try:
            current_path = Path.cwd()
            result_dir = Path(self.scamppath) / 'result'
            result_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(result_dir)

            self.print(f'RUN COMMAND: {command}', verbose)
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            
            is_succeeded = [Path(scamp_output).is_file() for scamp_output in scamp_output_images.values()]
            
            if all(is_succeeded):    
                self.print("SCAMP process finished=====================", verbose)
            else:
                self.print("SCAMP process failed. Some output files are missing.", verbose)
                return None
            
            if overwrite:
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

                    self.print(f"Updated WCS for {image_file} using {head_file}", verbose)

                for (image, header), result in zip(scamp_output_images.items(), is_succeeded):
                    update_fits_with_head(Path(image), Path(header))
                    # Remove header file
                    os.remove(header)
                # When updating files, return the updated file names
                return is_succeeded, list(scamp_output_images.keys())
            else:
                # When not updating files, return the output files and headers
                return is_succeeded, scamp_output_images

        except subprocess.CalledProcessError as e:
            self.print(f"Error during SCAMP execution: {e}", verbose)
            return None
        finally:
            # Remove cat file
            for cat in sex_output_images.values():
                os.remove(cat)
            os.chdir(current_path)

    def run_swarp(self,
                  # Input parameters
                  target_path: Union[str, List[str], Path, List[Path]],  
                  swarp_configfile: Union[str, Path],
                  swarp_params: Optional[dict] = None,
                  target_outpath: Union[str, Path] = None,
                  weight_inpath: Optional[Union[str, List[str], Path, List[Path]]] = None,
                  weight_outpath: Optional[str] = None,
                  weight_type: str = 'MAP_RMS', # BACKGROUND, MAP_RMS, MAP_VAR, MAP_WEIGHT
                  
                  # Resampling configuration
                  resample: bool = True,
                  resample_type: bool = 'LANCZOS3',
                  center_ra: Optional[float] = None,
                  center_dec: Optional[float] = None,
                  x_size: Optional[int] = None,
                  y_size: Optional[int] = None,
                  pixelscale: Optional[float] = None,
                  
                  # Combine configuration
                  combine: bool = True,
                  combine_type: str = 'MEDIAN',
                  
                  # Background subtraction configuration (From SWARP, Not recommended)
                  subbkg: bool = False,
                  box_size: int = 512,
                  filter_size: int = 3,
                  
                  # Fill to nan value 
                  fill_zero_tonan: bool = True,                  
                  verbose: bool = True) -> None:

        """_summary_

        Args:
            target_path = '/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T00176/7DT16/g/subbkg_calib_7DT16_T00176_20250315_025251_g_100.fits'
            target_outpath = '/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T00176/7DT16/g/subbkg_calib_7DT16_T00176_20250315_025251_g_100.coadd.fits'
            swarp_configfile = self.get_swarpconfigpath('7DT', 'C361K', 1, 'HIGH')
            swarp_params = None
            weight_inpath: Optional[Union[str, List[str], Path, List[Path]]] = '/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T00176/7DT16/g/calib_7DT16_T00176_20250315_025251_g_100.fits.errormap'
            weight_outpath: Optional[str] = '/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T00176/7DT16/g/calib_7DT16_T00176_20250315_025251_g_100.fits.coadd.errormap'
            weight_type: str = 'MAP_RMS' # BACKGROUND, MAP_RMS, MAP_VAR, MAP_WEIGHT
            
            # Resampling configuration
            resample: bool = True
            center_ra: Optional[float] = None
            center_dec: Optional[float] = None
            x_size: Optional[int] = None
            y_size: Optional[int] = None
            
            # Combine configuration
            combine: bool = False
            combine_type: str = 'median'
            
            # Background subtraction configuration (From SWARP, Not recommended)
            subbkg: bool = False
            box_size: int = 512
            filter_size: int = 3
            
            verbose: bool = True
            
            
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
            
        # swarp_configfile check
        if Path(swarp_configfile).is_file():
            swarp_config = Path(swarp_configfile)
        else:
            raise ValueError("SWARP configuration file does not exist.")
        
        # target_inpath setting
        if isinstance(target_path, (str, Path)):
            target_path = [Path(target_path)]
        else:
            target_path = [Path(img) for img in target_path]
        if not target_path:
            raise ValueError("No valid images provided for SWARP.")
        
        # target_outpath check
        target_outpath = Path(target_outpath) if target_outpath is not None else target_path[0].with_suffix('.com.fits')
        output_dir = target_outpath.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # weight_inpath check
        if isinstance(weight_inpath, (str, Path)):
            weight_inpath = [Path(weight_inpath)]
        elif isinstance(weight_inpath, list):
            weight_inpath = [Path(img) for img in weight_inpath]
        else:
            weight_inpath = None
            
        # weight_outpath check
        weight_outpath = Path(weight_outpath) if weight_outpath else None

        # center_ra/dec check
        iterator = tqdm(target_path, desc='Reading headers...') if verbose else target_path
        target_header = [fits.getheader(img) for img in iterator]
        if not center_ra or not center_dec:
            if 'OBJCTRA' in target_header[0].keys():
                center_ra = target_header[0]['OBJCTRA']
                center_dec = target_header[0]['OBJCTDEC']
                coord = self.to_skycoord(center_ra, center_dec)
                center_ra = coord.ra.to_string(unit=u.hourangle, sep=':')
                center_dec = coord.dec.to_string(unit=u.degree, sep=':', alwayssign=True)
                self.print(f'Center RA/Dec not provided. Using the mean of OBJCTRA/OBJCTDEC: {center_ra}/{center_dec}', verbose)    
            else:
                center_ra = np.mean([hdr['CRVAL1'] for hdr in target_header])
                center_dec = np.mean([hdr['CRVAL2'] for hdr in target_header])
                self.print(f'Center RA/Dec not provided. Using the mean of CRVAL1/CRVAL2: {center_ra}/{center_dec}', verbose)    
                center_coord = SkyCoord(center_ra, center_dec, unit='deg')
                center_ra = center_coord.ra.to_string(unit=u.hourangle, sep=':')
                center_dec = center_coord.dec.to_string(unit=u.degree, sep=':', alwayssign=True)

        if x_size and y_size:
            x_size = int(x_size)
            y_size = int(y_size)
        
        if not swarp_params:
            swarp_params = dict()
        
        # Input and output file settings
        all_params = self.load_config(swarp_configfile)
        swarp_params['CENTER_TYPE'] = 'MANUAL'
        swarp_params['CENTER'] = f'{center_ra},{center_dec}'
        if x_size and y_size:
            swarp_params['IMAGE_SIZE'] = f'{x_size},{y_size}'
        swarp_params['RESAMPLE'] = 'Y' if resample else 'N'
        swarp_params['RESAMPLE_DIR'] = str(output_dir)
        swarp_params['RESAMPLING_TYPE'] = resample_type.upper()
        if pixelscale:
            swarp_params['PIXEL_SCALE'] = str(pixelscale)
        
        swarp_params['IMAGEOUT_NAME'] = str(target_outpath)
        if weight_outpath is not None:
            swarp_params['WEIGHTOUT_NAME'] = weight_outpath
        else:
            swarp_params['WEIGHTOUT_NAME'] = str(Path(target_outpath).with_suffix('.weight.fits'))
        if weight_inpath is not None:
            swarp_params['WEIGHT_IMAGE'] = ','.join([str(img) for img in weight_inpath])
            swarp_params['WEIGHT_TYPE'] = weight_type.upper()
        else:
            swarp_params['WEIGHT_TYPE'] = 'NONE'
        
        # When combine is False, just resample the images to the specified coordinate
        if not combine:
            swarp_params['COMBINE'] = 'N'
            swarp_params['DELETE_TMPFILES'] = 'N'
            target_outlist = [Path(swarp_params['RESAMPLE_DIR']) / (file_.stem + all_params['RESAMPLE_SUFFIX']) for file_ in target_path]
            weights_outlist = [Path(swarp_params['RESAMPLE_DIR']) / (file_.stem + '_resamp.fits')for file_ in weight_inpath] if weight_inpath is not None else []
            output_filelist = [target_outlist, weights_outlist]
        # When combine is True, combine the images. If combine_type == 'weighted', use the weighted mean
        else:
            swarp_params['COMBINE'] = 'Y'
            swarp_params['COMBINE_TYPE'] = combine_type.upper()
            swarp_params['DELETE_TMPFILES'] = 'Y'              
            output_filelist = [swarp_params['IMAGEOUT_NAME'], swarp_params['WEIGHTOUT_NAME']]    

        # Input and output file settings
        if subbkg:
            swarp_params['SUBTRACT_BACK'] = 'Y'
            swarp_params['BACK_SIZE'] = box_size
            swarp_params['BACK_FILTERSIZE'] = filter_size
            
        swarpparams_str = ''
        
        for key, value in swarp_params.items():
            swarpparams_str += f'-{key} {value} '   
        
        # Command to run SWARP
        all_images_str = ' '.join([str(img) for img in target_path])
        command = f'SWarp {all_images_str} -c {swarp_configfile} {swarpparams_str}'
        
        if fill_zero_tonan:
            for path in target_path:
                image = fits.open(path)
                # If data is 0 value, fill it with nan_value
                image[0].data[np.where(image[0].data == 0)] = np.nan
                image.writeto(path, overwrite=True)
        
        try:
            self.print(f'RUN COMMAND: {command}', verbose)
            current_path = os.getcwd()
            # Run the SExtractor command using subprocess.run
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.print("SWARP process finished=====================", verbose)
            return output_filelist
        except Exception as e:
            self.print(f"Error during SWARP execution: {e}", verbose)
            return [None, None]

    def run_ds9(self, filelist: Union[str, Path, List[Union[str, Path]], np.ndarray], shell: str = '/bin/bash'):
        '''
        Parameters
        ----------
        filelist : str, Path, list, or np.ndarray
            Path or list of paths to FITS files for visualization.

        shell : str
            Shell to execute the ds9 command.

        Returns
        -------
        None

        Notes
        -----
        Opens ds9 with given FITS files using zscale and image/frame locking options.
        '''

        ds9_options = "-scalemode zscale -scale lock yes -frame lock image "
        
        # Normalize input to a list of strings
        if isinstance(filelist, (str, Path, np.str_)):
            filelist = [filelist]

        filelist = [str(f) for f in filelist]  # Convert Path or np.str_ to string
        names = " ".join(filelist)

        ds9_command = f"ds9 {ds9_options}{names} &"
        print(f'Running "{ds9_command}" in the terminal...')

        subprocess.Popen([shell, "-i", "-c", ds9_command])
        
    def to_regions(self, 
                reg_x, 
                reg_y, 
                reg_a=None,
                reg_b=None,
                reg_theta=None,
                reg_size: float = 6.0, 
                output_file_path: str = None):
        import astropy.units as u
        from regions import CirclePixelRegion, EllipsePixelRegion, PixCoord, Regions

        # Normalize input
        if isinstance(reg_x, (int, float)) and isinstance(reg_y, (int, float)):
            x_list = [reg_x]
            y_list = [reg_y]
        elif isinstance(reg_x, list) and isinstance(reg_y, list):
            x_list = reg_x
            y_list = reg_y
        else:
            x_list = list(reg_x)
            y_list = list(reg_y)
            
        N = len(x_list)

        # Normalize ellipse parameters
        def normalize_param(p, default=1.0):
            if p is None:
                return [default] * N
            elif isinstance(p, (int, float)):
                return [p] * N
            elif isinstance(p, list):
                return p
            else:
                return list(p)

        use_ellipse = reg_a is not None and reg_b is not None and reg_theta is not None
        if use_ellipse:
            a_list = normalize_param(reg_a)
            b_list = normalize_param(reg_b)
            theta_list = normalize_param(reg_theta, default=0.0)

        # Create regions
        region_list = []
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            center = PixCoord(x=x, y=y)

            if use_ellipse:
                a = reg_size * a_list[i]
                b = reg_size * b_list[i]
                theta = theta_list[i]
                region = EllipsePixelRegion(center=center, width=a, height=b, angle=theta * u.deg)
            else:
                radius = float(reg_size)
                region = CirclePixelRegion(center=center, radius=radius)

            region_list.append(region)

        # Write to file
        reg = Regions(region_list)
        if output_file_path is not None:
            reg.write(output_file_path, format='ds9', overwrite=True)
            return reg
        else:
            return reg