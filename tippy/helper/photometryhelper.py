
# %%
from astropy.io import fits
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table
from astropy.wcs import WCS

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
        return self.path_global
    
    @property
    def scamppath(self):
        return self.config['SCAMP_CONFIGDIR']
    
    @property  
    def swarppath(self):
        return self.config['SWARP_CONFIGDIR']
    
    @property
    def sexpath(self):
        return self.config['SEX_CONFIGDIR']
    
    @property
    def psfexpath(self):
        return self.config['PSFEX_CONFIGDIR']
        
    def __repr__(self):
        methods = [f'PhotometryHelper.{name}()\n' for name, method in inspect.getmembers(
            PhotometryHelper, predicate=inspect.isfunction) if not name.startswith('_')]
        txt = '[Methods]\n'+''.join(methods)
        return txt
    
    def print(self, string, do_print : bool = False):
        print(string) if do_print else None
        
    # Load information

    def get_imginfo(self, filelist, keywords : list = None, normalize_key: bool = True):
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

        # Get unique parent directories
        directories = list(set(os.path.dirname(file) for file in filelist))
        
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
                summary['file'] = [os.path.join(directory, f) for f in summary['file']]

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
        filelist_queried = np.array(all_coll['file'])
        all_coll = all_coll[np.isin(filelist_queried, filelist_inputted)]

        # Sort the final result to match the original filelist order
        file_order = {f: i for i, f in enumerate(filelist)}
        sort_idx = np.argsort([file_order.get(f, float('inf')) for f in all_coll['file']])
        all_coll = all_coll[sort_idx]

        # Check final count of combined FITS files
        print(f"Total FITS files combined: {len(all_coll)}")
        return all_coll


    # def get_imginfo(self, 
    #                 filelist, 
    #                 keywords=['jd', 'group', 'filter', 'exptime', 'object', 'telescop', 'instrume', 'ra', 'dec', 'xbinning', 'ybinning', 'imagetyp'],
    #                 normalize_key: bool = True):
    #     '''
    #     Parameters
    #     ----------
    #     1. filelist: (list) List of FITS image file paths.

    #     2. keywords : (list) List of FITS header keywords to extract.

    #     3. normalize_key: (bool) If True, normalize keywords based on required_key_variants.

    #     Returns 
    #     -------
    #     1. result_tbl : astropy.table
    #                     Table containing FITS header information.

    #     Notes 
    #     -----
    #     - "comment" and "history" are removed due to excessive length.
    #     - If `normalize_key=True`, returned keys will be standardized.
    #     - If `normalize_key=False`, the original keywords are retained.
    #     '''

    #     filedirs = set(os.path.dirname(filename) for filename in filelist)
    #     match_tbl = Table([filelist], names=['file'])
    #     result_tbl = Table()

    #     for filedir in filedirs:
    #         files = [os.path.basename(file) for file in filelist if os.path.dirname(file) == filedir]
    #         iminfo = ImageFileCollection(location=filedir, keywords=keywords, filenames=files).summary

    #         # Convert file paths to absolute paths
    #         iminfo['file'] = [os.path.join(filedir, f) for f in iminfo['file']]

    #         if normalize_key:
    #             # Normalize header keys
    #             new_column_names = {}
    #             for colname in iminfo.colnames:
    #                 normalized_key = self.normalize_required_keys(colname)
    #                 if normalized_key:
    #                     new_column_names[colname] = normalized_key  # Map old key to new key

    #             # Rename columns to normalized keys
    #             iminfo.rename_columns(list(new_column_names.keys()), list(new_column_names.values()))

    #             # Keep only normalized keys + 'file'
    #             valid_keys = set(self.required_key_variants.keys()) | {'file'}
    #             iminfo = iminfo[[col for col in iminfo.colnames if col in valid_keys]]

    #         result_tbl = vstack([iminfo, result_tbl])

    #     result_tbl = join(match_tbl, result_tbl, keys='file', join_type='inner')

    #     return result_tbl
    
    
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
                          ccd: str = None,
                          binning : int = 1,
                          readoutmode: str = None,
                          for_scamp : bool = False
                          ):
        file_key = f'{telescope.upper()}'
        if ccd:
            file_key += f'_{ccd.upper()}'
        if readoutmode:
            file_key += f'_{readoutmode.upper()}'
        if binning:
            file_key += f'_{binning}x{binning}'
        if for_scamp:
            file_key += '.scamp'
        file_key += '.sexconfig'
        file_path = os.path.join(self.configpath, 'sextractor', file_key)
        is_exist = os.path.exists(file_path)
        if is_exist:
            return file_path          
        else:
            raise FileNotFoundError(f'{file_key} not found: {file_path}')

    def get_scampconfigpath(self):
        file_path = os.path.join(self.configpath, 'scamp', 'default.scampconfig')
        is_exist = os.path.exists(file_path)
        if is_exist:
            return file_path
        else:
            raise FileNotFoundError(f'default.scampconfig not found :{file_path}')

    def get_swarpconfigpath(self,
                            telescope : str,
                            ccd : str = None,
                            binning : int = 1,
                            readoutmode : str = None):
        file_key = f'{telescope.upper()}'
        if ccd:
            file_key += f'_{ccd.upper()}'
        if readoutmode:
            file_key += f'_{readoutmode.upper()}'
        if binning:
            file_key += f'_{binning}x{binning}'
        file_key += '.swarpconfig'
        file_path = os.path.join(self.configpath, 'swarp', file_key)
        is_exist = os.path.exists(file_path)
        if is_exist:
            return file_path
        else:
            raise FileNotFoundError(f'{file_key} not found :{file_path}')

    def get_telinfo(self,
                    telescope=None, 
                    ccd=None, 
                    readoutmode=None, 
                    binning=None, 
                    key_observatory='obs', 
                    key_ccd='ccd', 
                    key_mode='mode', 
                    key_binning='binning',
                    obsinfo_file=None):
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
        obsinfo_file : str, optional
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
            obsinfo_file = os.path.join(self.configpath, 'CCD.dat')
        all_obsinfo = ascii.read(obsinfo_file, format='fixed_width')

        def filter_by_column(data, column, value):
            """Filters a table by a specific column value."""
            return data[data[column] == value] if value in data[column] else data

        def prompt_choice(options, message):
            """Prompts user to select from multiple options."""
            print(f"{message}: {options}")
            return input("Enter choice: ")

        # Select telescope if not provided
        if telescope is None:
            telescope = prompt_choice(set(all_obsinfo[key_observatory]), "Choose the Telescope")

        # Validate telescope existence
        if telescope not in all_obsinfo[key_observatory]:
            raise AttributeError(f"Telescope {telescope} information not found. Available: {set(all_obsinfo[key_observatory])}")

        # Filter for the selected telescope
        obs_info = filter_by_column(all_obsinfo, key_observatory, telescope)

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
        obs_info = filter_by_column(obs_info, key_binning, int(binning))

        # Ensure only one row remains
        if len(obs_info) == 1:
            return obs_info[0]
        
        raise AttributeError(f"No matching CCD info for {telescope}. Available CCDs: {list(set(all_obsinfo[key_ccd]))}")

    def load_sexconfig(self, sexconfig: str) -> dict:
        config_dict = {}

        with open(sexconfig, 'r') as file:
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
                            value = float(
                                value) if '.' in value else int(value)
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
                    config_dict[key] = value
        return config_dict

    def load_scampconfig(self, scampconfig: str) -> dict:
        config_dict = {}

        with open(scampconfig, 'r') as file:
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
                            value = float(
                                value) if '.' in value else int(value)
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
                    config_dict[key] = value
        return config_dict
    
    def load_psfexconfig(self, psfexconfig: str) -> dict:
        config_dict = {}

        with open(psfexconfig, 'r') as file:
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
                            value = float(
                                value) if '.' in value else int(value)
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
                    config_dict[key] = value
        return config_dict
    
    def load_swarpconfig(self, swarpconfig: str) -> dict:
        config_dict = {}

        with open(swarpconfig, 'r') as file:
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
                            value = float(
                                value) if '.' in value else int(value)
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
                    config_dict[key] = value
        return config_dict

    # Calculation

    def to_skycoord(self, ra, dec, frame: str = 'icrs'):
        import astropy.units as u
        '''
		parameters
		----------
		1. ra : str or float
				Right ascension in diverse format(see notes)
		2. dec : str or float
				Declination in diverse format(see notes)
		
		returns 
		-------
		1. skycoord : SkyCoord
		
		notes 
		-----
		Current supported formats
				1. 15h32m10s, 50d15m01s
				2. 15 32 10, 50 15 01
				3. 15:32:10, 50:15:01
				4. 230.8875, 50.5369
		-----
		'''
        ra = str(ra)
        dec = str(dec)
        if (':' in ra) & (':' in dec):
            skycoord = SkyCoord(ra=ra, dec=dec, unit=(
                u.hourangle, u.deg), frame=frame)
        elif ('h' in ra) & ('d' in dec):
            skycoord = SkyCoord(ra=ra, dec=dec, unit=(
                u.hourangle, u.deg), frame=frame)
        elif (' ' in ra) & (' ' in dec):
            skycoord = SkyCoord(ra=ra, dec=dec, unit=(
                u.hourangle, u.deg), frame=frame)
        else:
            skycoord = SkyCoord(ra=ra, dec=dec, unit=(
                u.deg, u.deg), frame=frame)
        return skycoord

    def bn_median(self, masked_array, axis=None):
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

    def cross_match(self, obj_catalog, sky_catalog, max_distance_second=5):
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

    def group_table(self, tbl: Table, key: str, tolerance: float = 0.1):
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

    def match_table(self, tbl1, tbl2, key, tolerance=0.01):
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

    def binning_table(self, tbl, key, tolerance=0.01):
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

    def remove_rows_table(self, tbl, column_key, remove_keys):
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
 
    def calculate_rotang(self, target_img, update_header : bool = False, print_output : bool = False):
        from astropy.io import fits
        from astropy.wcs import WCS
        import numpy as np
        #fits_file = filelist[]
        # Load the FITS file with the astrometry solution
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
        

    def cutout_img(self, target_img, size=0.9, prefix='cut_', xcenter=None, ycenter=None, print_output: bool = True):
        '''
        parameters
        ----------
        1. target_img : str
                        {abspath} of the target image
        2. size : float or int
                        (float) ratio of the cut image (0.9)
                        (int) size of the cut image in pixel 
        3. prefix : str
                        prefix of the output image
        4. xcenter : optional, int or (float or str)
                        (int) pixel coordinate of the center
                        (float or str) RA coordinate of the center
        4. ycenter : optional, int or (float or str)
                        (int) pixel coordinate of the center
                        (float or str) Dec coordinate of the center
        returns 
        -------
        1. outputname : str
                        {abspath} of the cutout image

        notes 
        -----
        if xcenter, ycenter == None, cutout image will be centered to the center of the original image
        -----
        '''
        from astropy.wcs import WCS
        from astropy.nddata import Cutout2D
            
        self.print('Start image cutout... \n', print_output)
        hdul = fits.open(target_img)
        hdu = hdul[0]
        wcs = WCS(hdu.header)
        if size < 1:
            size = size*int(len(hdu.data))
        if (xcenter == None) & (ycenter == None):
            xcenter, ycenter = len(hdu.data)//2, len(hdu.data)//2
        if (type(xcenter) != int) & (type(ycenter) != int):
            center_coords = self.to_skycoord(xcenter, ycenter)
            cutouted = Cutout2D(
                data=hdu.data, position=center_coords, size=size, wcs=wcs)
        else:
            cutouted = Cutout2D(data=hdu.data, position=(
                xcenter, ycenter), size=size, wcs=wcs)
        cutouted_hdu = hdu
        cutouted_hdu.data = cutouted.data
        cutouted_hdu.header.update(cutouted.wcs.to_header())
        cutouted_hdu.header['NAXIS1'] = int(size)
        cutouted_hdu.header['NAXIS2'] = int(size)
        outputname = f'{os.path.dirname(target_img)}/{prefix}{os.path.basename(target_img)}'
        cutouted_hdu.writeto(outputname, overwrite=True)
        hdul.close()
        self.print('Image cutout complete \n', print_output)
        return outputname

    def align_img(self, target_img, reference_img, prefix='align_', detection_sigma = 5, print_output: bool = True):
        """

        parameters
        ----------
        1. target_img : str
                        absolute path of a target image 
        2. reference_img : str
                        absolute path of a reference image
                        The image as a reference frame. All images will be aligned on the basis of this image
        3. prefix = str
                        prefix of the aligned iamge

        returns
        ----------
        1. outputname : str
                        absolute path of the aligned image
        notes
        ----------
        ----------
        """

        import astroalign as aa
        from ccdproc import CCDData
        from astropy.wcs import WCS

        self.print('Start image alignment... \n', print_output)
        tgt_hdul = fits.open(target_img)
        ref_hdul = fits.open(reference_img)
        tgt_hdu = tgt_hdul[0]
        ref_hdu = ref_hdul[0]
        tgt_data = tgt_hdu.data
        ref_data = ref_hdu.data
        tgt_hdr = tgt_hdu.header
        ref_hdr = ref_hdu.header
        
        ref_wcs = WCS(ref_hdr)
        wcs_hdr = ref_wcs.to_header(relax = True)
        wcs_hdr.remove('DATE-OBS', ignore_missing = True)
        wcs_hdr.remove('MJD-OBS', ignore_missing = True)
        wcs_hdr.remove('RADESYS', ignore_missing = True)
        wcs_hdr.remove('EQUINOX', ignore_missing = True)
        tgt_hdr.update(wcs_hdr)
        tgt_data = tgt_data.byteswap().newbyteorder()
        ref_data = ref_data.byteswap().newbyteorder()
        try:
            aligned_data, footprint = aa.register(tgt_data, ref_data, fill_value=0, detection_sigma= detection_sigma, max_control_points=30)
            aligned_tgt = CCDData(aligned_data, header=tgt_hdr, unit='adu')
            outputname = f'{os.path.dirname(target_img)}/{prefix}{os.path.basename(target_img)}'
            fits.writeto(outputname, aligned_tgt.data, aligned_tgt.header, overwrite=True)
            self.print('Image alignment complete \n', print_output)
            return outputname
        except:
            self.print('Failed to align the image. Check the image quality and the detection_sigma value.', print_output)
            raise ActionFailedError('Failed to align the image. Check the image quality and the detection_sigma value.')
        finally:    
            tgt_hdul.close()
            ref_hdul.close()
        
    def combine_img(self,
                    filelist,
                    combine_method: str = 'median',
                    scale: str = 'multiply',
                    prefix: str = 'com_',
                    output_name : str = None,
                    zp_key: str ='ZP5_1',
                    print_output: bool = True,
                    
                    # Clipping parameters
                    clip: str = 'extrema',
                    clip_sigma_low: int = 2,
                    clip_sigma_high: int = 5,
                    clip_minmax_min: int = 3,
                    clip_minmax_max: int = 3,
                    clip_extrema_nlow: int = 1,
                    clip_extrema_nhigh: int = 1,
                    ):
        '''
        parameters
        ----------
        1. filelist : list or np.array 
                        filelist to be combined
        2. clip : str
                        method for clipping [None, minmax, sigma, extrema] (sigma)
        3. combine : str
                        method for combining [mean, median, sum] (median)
        4. scale : str
                        method for scaling [None, zero, multiply] (zero)
        5. prefix : str
                        prefix of the combined image

        2.1. clip_sigma_low : optional, int
                        Threshold for rejecting pixels that deviate below the baseline value.
        2.2. clip_sigma_high : optional, int
                        Threshold for rejecting pixels that deviate above the baseline value.    
        2.3. clip_minmax_min : optional, int
                        If not None, all pixels with values below min_clip will be masked.
        2.4. clip_minmax_max : optional, int
                        If not None, all pixels with values above min_clip will be masked.
        2.5. clip_extrema_nlow : optional, int
                        If not None, the number of low values to reject from the combination.
        2.6. clip_extrema_nhigh : optional, int
                        If not None, the number of high values to reject from the combination.

        returns 
        -------
        1. outputname : str
                        absolute path of the combined image

        notes 
        -----
        For more information : https://ccdproc.readthedocs.io/en/latest/image_combination.html
        -----
        '''
        from ccdproc import CCDData
        from ccdproc import Combiner
        import psutil
        import os
        import gc

        def print_memory_usage(output_string = 'Memory usage'):
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            print(f"{output_string}: {mem_info.rss / 1024**2:.2f} MB")  # Convert bytes to MB

        if len(filelist) <3:
            clip = None
            self.print('Number of filelist is lower than the minimum. Skip clipping process... \n', print_output)
        
        self.print('Start image combine... \n', print_output)

        def read_fits_int16(filename):
            with fits.open(filename, memmap=False) as hdul:
                data = hdul[0].data.astype(np.int16)
                header = hdul[0].header
            return CCDData(data, unit='adu', meta=header)
        
        ccdlist = []        
        for file_ in tqdm(filelist, desc = 'Reading files...'):
            ccdlist.append(read_fits_int16(file_))
            print_memory_usage()
        hdr = ccdlist[0].header.copy()
        init_mean, init_std = np.mean(ccdlist[0].data), np.std(ccdlist[0].data)
        
        for i, file in enumerate(filelist):
            hdr[f'COMBIM{i+1}'] = os.path.basename(file)
        # 여기 다시 zp - zp_ref 해서 스케일링 하는걸로 변경
        hdr['NCOMBINE'] = int(len(filelist))
        if 'JD' in hdr.keys():
            hdr['JD'] = Time(np.mean([inim.header['JD'] for inim in ccdlist]), format='jd').value
        if 'DATE-OBS' in hdr.keys():
            hdr['DATE-OBS'] = Time(np.mean([Time(inim.header['DATE-OBS']).jd for inim in ccdlist]), format='jd').isot
        hdr['TOTALEXP'] = float(np.sum([inim.header['EXPTIME'] for inim in ccdlist]))
        print_memory_usage(output_string = 'Memory usage before combiner')
        combiner = Combiner(ccdlist, dtype=np.float32)
        print_memory_usage(output_string = 'Memory usage after combiner')
        if scale == 'multiply':
            zp_median = np.median([inim.header[zp_key] for inim in ccdlist]) 
            for inim in ccdlist:
                zp_diff = inim.header[zp_key] - zp_median
                inim.data *= 100 ** (-zp_diff/5)
            #for i, inim in enumerate(ccdlist):
            #    zp = inim.header[zp_key]
            #    zp_diff = zp - zp_median
            #    ccdlist[i].data = ccdlist[i].data * 100 ** (-zp_diff/5)
            
        elif (scale == 'zero'):
            averages = [np.mean(ccddata) for ccddata in ccdlist]
            delvalues = averages - averages[0]
            for i, delvalue in enumerate(delvalues):
                ccdlist[i].data = ccdlist[i].data-delvalue
                combiner = Combiner(ccdlist, dtype=np.float32)

        # Free memory 
        del ccdlist
        gc.collect()
        
        # Clipping
        print_memory_usage(output_string = 'Memory usage before clipping')
        if clip == 'minmax':
            combiner.minmax_clipping(min_clip=clip_minmax_min, max_clip=clip_minmax_max)
        if clip == 'sigma':
            combiner.sigma_clipping(low_thresh=clip_sigma_low, high_thresh=clip_sigma_high, func=np.ma.median)
        if clip == 'extrema':
            combiner.clip_extrema(nlow=clip_extrema_nlow, nhigh=clip_extrema_nhigh)
        print_memory_usage(output_string = 'Memory usage after clipping')
        # Combining
        if combine_method == 'median':
            combined = combiner.median_combine(median_func=self.bn_median)
        if combine_method == 'mean':
            combined = combiner.average_combine()
        if combine_method == 'sum':
            combined = combiner.sum_combine()
        print_memory_usage(output_string = 'Memory usage after combining')
        
        # Free memory 
        del combiner
        gc.collect()

        combined.header = hdr

        outputname = f'{os.path.dirname(filelist[0])}/{prefix}{os.path.basename(filelist[0])}'
        if output_name:
            outputname = os.path.join(os.path.dirname(filelist[0]), output_name)
        if (len(filelist) == 1):
            ccd.header = hdr
            ccd.write(outputname, overwrite=True, format='fits')
        else:
            combined.write(outputname, overwrite=True, format='fits')
        fin_mean, fin_std = np.mean(combined.data), np.std(combined.data)
        
        # Free memory 
        del combined
        del ccd
        gc.collect()
        print_memory_usage(output_string = 'Memory usage after writing')


        self.print('Combine complete \n',print_output)
        self.print('Combine information',print_output)
        self.print(60*'=',print_output)
        self.print(f'Ncombine = {len(filelist)}',print_output)
        self.print(f'method   = {clip}(clipping), {combine_method}(combining)',print_output)
        self.print(f'mean     = {round(init_mean,3)} >>> {round(fin_mean,3)}',print_output)
        self.print(f'std      = {round(init_std,3)} >>> {round(fin_std,3)}',print_output)
        self.print(f'image path = {outputname}',print_output)
        return outputname

    def convolve_img(self,
                     target_img,
                     fwhm_tgt, 
                     fwhm_ref,
                     pixelscale = 0.505,
                     method = 'gaussian'):
        print(f'Start convolution on {os.path.basename(target_img)}... \n COMMAND = (FWHM_TARGET = {fwhm_tgt}, FWHM_REFERENCE = {fwhm_ref}, PIXELSCALE = {pixelscale})')
        import numpy as np
        from astropy.io import fits
        from astropy.convolution import convolve, Gaussian2DKernel
        import matplotlib.pyplot as plt

        # Calculate sigmas
        sigma_tgt = fwhm_tgt / 2.355 / pixelscale
        sigma_ref = fwhm_ref / 2.355 / pixelscale

        # Calculate the convolution kernel sigma
        sigma_conv = np.sqrt(sigma_ref**2 - sigma_tgt**2)

        # Load the image
        image_data = fits.getdata(target_img)
        
        # Create a Gaussian kernel for convolution
        kernel = Gaussian2DKernel(sigma_conv)

        # Convolve the good image with the kernel
        convolved_image = convolve(image_data, kernel, normalize_kernel=True)

        # Save the convolved image to a new FITS file
        convolved_path = os.path.join(os.path.dirname(target_img), 'conv_' + os.path.basename(target_img))
        
        hdu = fits.PrimaryHDU(convolved_image)
        hdu.header = fits.getheader(target_img)
        hdu.writeto(convolved_path, overwrite=True)

        # Display the original and convolved images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_data, origin='lower', cmap='gray')
        plt.title('Original Good Image')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(convolved_image, origin='lower', cmap='gray')
        plt.title('Convolved Image (Matched to Bad Seeing)')
        plt.colorbar()

        plt.tight_layout()
        plt.show()

        return convolved_path

    def run_psfex(self, 
                  target_img : str, 
                  sex_configfile : str, 
                  psfex_configfile : str,                   
                  sex_params : dict = None,
                  psfex_params : dict = None,
                  print_output : bool = True):
        """
        Parameters
        ----------
        1. image : str
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
        self.print('Start PSFEx process...=====================', print_output)

        # Switch to the SExtractor directory
        current_path = os.getcwd()
        os.chdir(self.sexpath)
        
        # Command to run psfex
        output_file = self.run_sextractor(image = target_img, sex_configfile = sex_configfile, sex_params = sex_params, return_result = False, print_output = False)
        
        all_params = self.load_psfexconfig(psfex_configfile)
        psfexparams_str = ''

        if psfex_params:
            if "CHECKIMAGE_NAME" in psfex_params.keys():
                fits_files = psfex_params['CHECKIMAGE_NAME'].split(',')
                abspath_fits_files = ','.join([os.path.join(self.config['PSFEX_HISTORYDIR'], fits_file) for fits_file in fits_files])
                psfex_params['CHECKIMAGE_NAME'] = abspath_fits_files
            else:
                fits_files = all_params['CHECKIMAGE_NAME'].split(',')
                abspath_fits_files = ','.join([os.path.join(self.config['PSFEX_HISTORYDIR'], fits_file) for fits_file in fits_files])
                psfex_params['CHECKIMAGE_NAME'] = abspath_fits_files
        else:
            psfex_params = dict()
            fits_files = all_params['CHECKIMAGE_NAME'].split(',')
            abspath_fits_files = ','.join([os.path.join(self.config['PSFEX_HISTORYDIR'], fits_file) for fits_file in fits_files])
            psfex_params['CHECKIMAGE_NAME'] = abspath_fits_files

        for key, value in psfex_params.items():
            psfexparams_str += f"-{key} {value} "
            
        
        command = f"psfex {output_file} -c {psfex_configfile} {psfexparams_str}"
        os.makedirs(self.config['PSFEX_HISTORYDIR'], exist_ok=True)
        
        try:
            os.chdir(self.psfexpath)
            # Run the SExtractor command using subprocess.run
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.print("PSFEx process finished=====================", print_output)
        except:
            self.print(f"Error during PSFEx execution", print_output)
            return
        finally:
            os.chdir(current_path)

        
    def run_hotpants(self,
                     target_img,
                     reference_img,
                     output_img,
                     convolved_img = None,
                     target_mask = None,
                     reference_mask = None,

                     # hotpants config
                     convdir = 't', # t or i (t when template has better seeing, i when image has better seeing)
                     normdir = 'i', 
                     iu=60000,
                     il=-1,
                     tu=600000000,
                     tl=-10000000,
                     nrx : int = 3,
                     nry : int = 2,
                     v=0,
                     print_output=True
                     ):
        '''
        parameters
        ----------
        1. target_img : str
                {abspath} of the target image to be subtracted
        2. reference_img : str
                {abspath} of the reference image 
        3. prefix : str
                prefix of the output image (sub_)
        4. method : str
                method for subtraction (hotpants)
        5. iu : int
                upper valid data count, image (60000)
        6. tu : int
                upper valid data count, template (600000000)
        7. tl : int
                lower valid data count, template (-100000)
        8. v : int
                level of verbosity, 0-2 (0)
        9. ng : str
                'ngauss degree0 sigma0 .. degreeN sigmaN'
                : ngauss = number of gaussians which compose kernel (3)
                : degree = degree of polynomial associated with gaussian # (3 2 1)
                : sigma  = width of gaussian # (1.0 0.7 0.4)
        returns 
        -------

        notes 
        -----
        For more information : https://github.com/acbecker/hotpants
        -----
        '''
        default_command = f'hotpants -c {convdir} -n {normdir} -inim {target_img} -tmplim {reference_img} -outim {output_img} -iu {iu} -il {il} -tu {tu} -tl {tl} -v {v} -nrx {nrx} -nry {nry} '
        
        if convolved_img:
            default_command += f'-oci {convolved_img} '
        if target_mask:
            default_command += f'-imi {target_mask} '
        if reference_mask:
            default_command += f'-tmi {reference_mask} '
            
        command = default_command
        self.print(f'Start image subtraction on {os.path.basename(target_img)}... \n COMMAND = {default_command}', print_output)

        result = subprocess.run(command, shell=True, timeout=900, check=True, text=True, capture_output=True)

        self.print(f"Image subtraction completed. Output saved to {output_img}", print_output)      
        return output_img

    def subtract_background(self, 
                            target_img: str,
                            apply_2D_bkg: bool = True,
                            mask_sources: bool = False,
                            mask_source_size_in_pixel : int = 10,
                            bkg_estimator: str = 'median', # mean, median, sextractor, 
                            bkg_sigma: float = 3.0, 
                            bkg_box_size: int = 300, 
                            bkg_filter_size: int = 3, 
                            prefix : str = 'subbkg_',
                            update_header: bool = True,
                            visualize: bool = False,
                            print_output: bool = True):
        """
        target_img: str
        mask_sources: bool = True
        mask_source_size_in_pixel : int = 10
        bkg_estimator: str = 'median' # mean, median, sextractor
        bkg_sigma: float = 3.0
        bkg_box_size: int = 50
        bkg_filter_size: int = 3
        prefix : str = 'subbkg_'
        update_header: bool = True
        visualize: bool = True
        Subtract background from the image using sigma-clipped statistics.
        
        Parameters
        ----------
        bkg_sigma : float, optional
            The sigma level for sigma clipping in background estimation, by default 3.0
        bkg_box_size : int, optional
            Size of the box used for local background estimation, by default 50
        bkg_filter_size : int, optional
            Size of the filter used to smooth the background estimation, by default 3
        update_header : bool, optional
            Whether to update the FITS header with the background subtraction info, by default True
        """
        from photutils.background import Background2D, MedianBackground, MeanBackground, SExtractorBackground
        from astropy.stats import SigmaClip, sigma_clipped_stats
        from photutils.segmentation import detect_threshold, detect_sources
        from photutils.utils import circular_footprint
        import matplotlib.pyplot as plt

        self.print(f"Start background subtraction...", print_output)

        # Load the image data
        hdul = fits.open(target_img)
        hdu = hdul[0]
        data = hdu.data

        # Create a mask for sources in the image
        mask = None
        if mask_sources & apply_2D_bkg:
            sigma_clip = SigmaClip(sigma=3.0)
            threshold = detect_threshold(data, nsigma = bkg_sigma, sigma_clip=sigma_clip)
            segment_img = detect_sources(data, threshold, npixels=mask_source_size_in_pixel)
            footprint = circular_footprint(radius=mask_source_size_in_pixel)
            mask = segment_img.make_source_mask(footprint=footprint)

        # Estimate background using sigma-clipped statistics
        bkg_estimator_dict = dict(MEAN=MeanBackground, MEDIAN=MedianBackground, SEXTRACTOR=SExtractorBackground)
        bkg_estimator = bkg_estimator_dict[bkg_estimator.upper()]
        if apply_2D_bkg:
            bkg = Background2D(data, (bkg_box_size, bkg_box_size), mask = mask, 
                               filter_size=(bkg_filter_size, bkg_filter_size),
                               sigma_clip= SigmaClip(sigma=3.0), 
                               bkg_estimator=bkg_estimator())
            bkg_value = bkg.background
            bkg_value_median = bkg.background_median
            bkg_rms = bkg.background_rms_median
        else:
            # Global background estimation
            clipped_data = sigma_clipped_stats(data, sigma=bkg_sigma)
            bkg_value = clipped_data[1] if bkg_estimator == 'median' else clipped_data[0]
            bkg_value_median = clipped_data[0]
            bkg_rms = clipped_data[2]

        # Subtract background from image
        data_bkg_subtracted = data - bkg_value

        # Update FITS header (optional)
        if update_header:
            hdu.header['BKG_TIME'] = (Time.now().isot, 'Time of background subtraction')
            hdu.header['BKG_SUB'] = (True, 'Background subtracted')
            hdu.header['BKG_BOX'] = (bkg_box_size, 'Background estimation box size')
            hdu.header['BKG_FILT'] = (bkg_filter_size, 'Background filter size')
            hdu.header['BKG_SIG'] = (bkg_sigma, 'Sigma clipping level for background')

        # Save the background-subtracted image (optional, overwrite or new file)
        hdul.close()
        output_filename = f'{os.path.dirname(target_img)}/{prefix}{os.path.basename(target_img)}'
        hdu.data = data_bkg_subtracted
        hdu.writeto(output_filename, overwrite=True)

        if visualize:
            # Plot the background-subtracted image
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            import numpy as np

            fig, ax = plt.subplots(1, 3, figsize=(12, 6))
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im0 = ax[0].imshow(data, origin='lower', cmap='Greys_r', vmin=bkg_value_median, vmax=bkg_value_median + 1 * bkg_rms)
            ax[0].set_title('Original Image')
            fig.colorbar(im0, cax=cax, orientation='vertical')
            
            if apply_2D_bkg:
                bkg_img = bkg.background
            else:
                bkg_img = np.full(data.shape, bkg_value)

            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im1 = ax[1].imshow(bkg_img, origin='lower', cmap='Greys_r', vmin=bkg_value_median, vmax=bkg_value_median + 1 * bkg_rms)
            ax[1].set_title('Background')
            fig.colorbar(im1, cax=cax, orientation='vertical')

            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im2 = ax[2].imshow(data_bkg_subtracted, origin='lower', cmap='Greys_r', vmin=0, vmax= bkg_rms)
            ax[2].set_title('Background-Subtracted Image')
            fig.colorbar(im2, cax=cax, orientation='vertical')
            plt.tight_layout()
            plt.show()
            
        self.print(f"Background subtraction completed. Output saved to {output_filename}", print_output)      
        return output_filename

    # Program running
    @timeout(seconds = 15)
    def run_astrometry(self,
                       image, 
                       sex_configfile : str,
                       ra : float = None,
                       dec : float = None,
                       radius : float = None,
                       scalelow : float = 0.6, 
                       scalehigh : float = 0.8, 
                       prefix : str = 'astrometry_',
                       overwrite : bool = False,
                       remove : bool = True,
                       print_output : bool = True
                       ):
        """
        1. Description
        : Solving WCS coordinates using Astrometry.net software. For better performance in especially B band images, --use-sextractor mode is added. This mode needs SExtractor configuration files. So please posit configuration files for your working directory. cpulimit 300 is also added to prevent too long processing time for bad images.
        : scalelow and scalehigh for the range of pixscale estimation

        2. History
        2018.03    Created by G.Lim.
        2018.12.18 Edited by G.Lim. SExtractor mode is added.
        2018.12.21 Edited by G.Lim. Define SAO_astrometry function.
        2020.03.01 --backend-config is added to have the system find INDEX files.
        2021.12.29 Edited by HH.Choi.  
        """
        import os,sys
        import glob
        import subprocess
        import numpy as np
        
        """
        Running the Astrometry process with options to pass RA/Dec and a timeout.
        """
        try:
            self.print('Start Astrometry process...=====================', print_output)
            # Set up directories and copy configuration files
            current_dir = os.getcwd()
            sex_dir = self.sexpath
            image_dir = os.path.dirname(image)
            os.chdir(sex_dir)
            os.system(f'cp {sex_configfile} {sex_dir}/*.param {sex_dir}/*.conv {sex_dir}/*.nnw {image_dir}')
            
            os.chdir(image_dir)
            self.print(f'Solving WCS using Astrometry with RA/Dec of {ra}/{dec} and radius of {radius} arcmin', print_output)

            # Building the command string
            if overwrite:
                new_filename = os.path.join(image_dir,os.path.basename(image))
                com = f'solve-field {image} --cpulimit 60 --overwrite --use-source-extractor --source-extractor-config {sex_configfile} --x-column X_IMAGE --y-column Y_IMAGE --sort-column MAG_AUTO --sort-ascending --scale-unit arcsecperpix --scale-low {str(scalelow)} --scale-high {str(scalehigh)} --no-remove-lines --uniformize 0 --no-plots --new-fits {new_filename} --temp-dir .'
            else:
                new_filename = os.path.join(image_dir, prefix + os.path.basename(image))
                com = f'solve-field {image} --cpulimit 60 --use-source-extractor --source-extractor-config {sex_configfile} --x-column X_IMAGE --y-column Y_IMAGE --sort-column MAG_AUTO --sort-ascending --scale-unit arcsecperpix --scale-low {str(scalelow)} --scale-high {str(scalehigh)} --no-remove-lines --uniformize 0 --no-plots --new-fits {new_filename} --temp-dir .'
            
            if ra is not None and dec is not None:
                com += f' --ra {ra} --dec {dec}'
            if radius is not None:
                com += f' --radius {radius}'
            
            # Use subprocess.run with timeout
            result = subprocess.run(com, shell=True, timeout=900, check=True, text=True, capture_output=True)
            orinum = subprocess.check_output(f'ls C*.fits | wc -l', shell=True)
            resnum = subprocess.check_output(f'ls a*.fits | wc -l', shell=True)
            
            # Clean up
            if remove:
                os.system(f'rm tmp* *.conv default.nnw *.wcs *.rdls *.corr *.xyls *.solved *.axy *.match check.fits *.param {os.path.basename(sex_configfile)}')
            self.print('Astrometry process finished=====================', print_output)
            return new_filename

        except subprocess.TimeoutExpired:
            self.print(f"The astrometry process exceeded the timeout limit.", print_output)
            return None
        except subprocess.CalledProcessError as e:
            self.print(f"An error occurred while running the astrometry process: {e}", print_output)
            return None
        except:
            self.print(f"An unknown error occurred while running the astrometry process.", print_output)
            return None

    def run_sextractor(self, image, 
                       sex_configfile, 
                       sex_params: dict = None, 
                       return_result: bool = True, 
                       print_output : bool = True):
        """
        Parameters
        ----------
        1. image : str
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
        all_params = self.load_sexconfig(sex_configfile)
        sexparams_str = ''

        if sex_params:
            if "CATALOG_NAME" not in sex_params.keys():
                sex_params['CATALOG_NAME'] = f"{os.path.join(self.config['SEX_HISTORYDIR'], os.path.basename(sex_params['CATALOG_NAME']))}"
            else:
                sex_params['CATALOG_NAME'] = f"{os.path.join(self.config['SEX_HISTORYDIR'], os.path.basename(all_params['CATALOG_NAME']))}"
        else:
            sex_params = dict()
            sex_params['CATALOG_NAME'] = f"{os.path.join(self.config['SEX_HISTORYDIR'], all_params['CATALOG_NAME'])}"

        for key, value in sex_params.items():
            sexparams_str += f'-{key} {value} '
            all_params[key] = value

        # Command to run SExtractor
        command = f"source-extractor {image} -c {sex_configfile} {sexparams_str}"
        os.makedirs(os.path.dirname(all_params['CATALOG_NAME']), exist_ok=True)

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
                  filelist : str or list, 
                  sex_configfile : str, 
                  scamp_configfile : str,                   
                  sex_params : dict = None,
                  scamp_params : dict = None,
                  update_files : bool = True, 
                  print_output : bool = True):
        
        if isinstance(filelist, str):
            filelist = [filelist]
        
        # Run SExtractor on each image in the filelist
        self.print(f'Start SCAMP process on {len(filelist)} images...=====================', print_output)
        sex_output_images = dict()
        for image in tqdm(filelist, desc='Running Source extractor...'):
            if sex_params is None:
                sex_params = dict()
            sex_params['CATALOG_NAME'] = f"{self.scamppath}/result/{os.path.basename(image).split('.')[0]}.sexcat"
            sex_params['PARAMETERS_NAME'] = f'{self.sexpath}/scamp.param'
            output_file = self.run_sextractor(image = image, sex_configfile = sex_configfile, sex_params = sex_params, return_result = False, print_output = False)
            sex_output_images[image] = output_file
        
        # Filter out images that failed to produce a catalog
        sex_output_images = {key: value for key, value in sex_output_images.items() if value is not None}
        scamp_output_images = {key: value.replace('.sexcat', '.head') for key, value in sex_output_images.items()}
        all_images_str = ' '.join(sex_output_images.values())
        
        # Load and apply SCAMP parameters
        all_params = self.load_scampconfig(scamp_configfile)
        scampparams_str = ''
        if scamp_params:
            for key, value in scamp_params.items():
                scampparams_str += f'-{key} {value} '
                all_params[key] = value
                
        # Command to run SCAMP
        command = f'scamp {all_images_str} -c {scamp_configfile} {scampparams_str}'
        
        try:
            current_path = os.getcwd()
            os.chdir(os.path.join(self.scamppath,'result'))
            # Run the SExtractor command using subprocess.run
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.print("SCAMP process finished=====================", print_output)

            if update_files:
                def sanitize_header(header: fits.Header) -> fits.Header:
                    """
                    Sanitize a FITS header by removing or cleaning non-ASCII and non-printable characters.
                    
                    Parameters:
                    header (fits.Header): The FITS header to be sanitized.
                    
                    Returns:
                    fits.Header: The sanitized header.
                    """
                    sanitized_header = fits.Header()
                    
                    # Loop through each header card and sanitize it
                    for card in header.cards:
                        key, value, comment = card
                        if isinstance(value, str):
                            # Remove non-ASCII characters from the value
                            value = re.sub(r'[^\x20-\x7E]+', '', value)
                        
                        # Add sanitized card to the new header
                        sanitized_header[key] = (value, comment)
                    
                    return sanitized_header

                def update_fits_with_head(image_file: str, head_file: str):
                    """
                    Update the WCS and other relevant header information in a FITS file using a SCAMP-generated .head file.
                    
                    Parameters:
                    image_file (str): Path to the FITS image file to be updated.
                    head_file (str): Path to the SCAMP-generated .head file with updated WCS and other parameters.
                    """
                    # Read the header from the .head file
                    with open(head_file, 'r') as head:
                        head_content = head.read()

                    # Convert the head file content to an astropy header object
                    head_header = fits.Header.fromstring(head_content, sep='\n')
                    
                    # Sanitize the header to remove non-ASCII characters
                    head_header = sanitize_header(head_header)

                    # Open the FITS image and update its header with WCS information from the .head file
                    hdul = fits.open(image_file)
                    hdul[0].header.update(head_header)
                    hdul.flush()
                    hdul.close()
                    self.print(f"Updated WCS and relevant header information for {image_file} using {head_file}", print_output)

                
                for image, header in scamp_output_images.items():
                    update_fits_with_head(image, header)
                return scamp_output_images.keys()
            else:
                return scamp_output_images.values()
        except:
            self.print(f"Error during SCAMP execution", print_output)
            return
        finally:
            os.chdir(current_path)
            
    def run_swarp(self,
                  filelist : str or list, 
                  path_outim : str,
                  swarp_configfile : str,
                  swarp_params : dict = None,
                  do_scamp : bool = False,
                  scamp_configfile : str = None,
                  sex_configfile : str = None, 
                  scamp_params : dict = None,
                  sex_params : dict = None,
                  print_output : bool = True):
        
        if isinstance(filelist, str):
            filelist = [filelist]
        
        # Run SExtractor on each image in the filelist
        succeeded_images = filelist
        if do_scamp:
            succeeded_images = self.run_scamp(filelist = filelist, sex_configfile= sex_configfile, scamp_configfile= scamp_configfile, sex_params= sex_params, scamp_params= scamp_params, update_files = True, print_output= print_output)        
        
        # Load and apply SWARP parameters
        all_params = self.load_swarpconfig(swarp_configfile)
        swarpparams_str = ''
        if not swarp_params:
            swarp_params = dict()
        swarp_params['IMAGEOUT_NAME'] = path_outim
        swarp_params['WEIGHTOUT_NAME'] = os.path.splitext(path_outim)[0] + '.weight.fits'
        #swarp_params['CENTER'] = "07:43:15,-22:55:28"
        #swarp_params['IMAGE_SIZE'] = '10200,6800'
        #swarp_params['NTHREADS'] = 4
        #swarp_params['CENTER_TYPE'] = 'MANUAL'
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
        

# %%
if __name__ == '__main__':
    import glob
    A = PhotometryHelper()
    sexconfigpath = A.get_sexconfigpath(telescope = '7DT', ccd = 'C361K', readoutmode = 'HIGH', binning = 1, for_scamp = True)
    #sexconfig = A.load_sexconfig(sexconfigpath)
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
