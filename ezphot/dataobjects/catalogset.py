
#%%
import inspect
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Union, List
from astropy.time import Time
from astropy.table import Table
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from ezphot.dataobjects import Catalog
from ezphot.imageobjects import ScienceImage
from ezphot.helper import Helper

#%%

class CatalogSet:
    """
    CatalogSet class for managing a set of catalogs.
    
    This class provides methods 
    
    1. Search for catalogs in the given folder.
    
    2. Select catalogs with given criteria.
    
    3. Exclude catalogs with given criteria.
    
    4. Add catalogs to the set.
    
    5. Merge catalogs into single table.
        
    6. Select sources from each Catalog instance.
    
    """
    def __init__(self, catalogs: list[Catalog] = None):
        """
        Initialize the CatalogSet class.

        Parameters
        ----------
        catalogs : list[Catalog], optional
            List of Catalog instances. Defaults to None. If None, the CatalogSet will be dummy instance.
        """
        self.helper = Helper()
        self.catalogs = catalogs if catalogs is not None else []
        self.target_catalogs = self.catalogs
        self._last_filter = dict(
            file_key=None,
            filter=None,
            exptime=None,
            objname=None,
            obs_start=None,
            obs_end=None,
            seeing=None,
            depth=None,
            observatory=None,
            telname=None
        )
        self._last_mode = "select"  # <-- Track last mode (select or exclude)
    
    def __repr__(self):
        
        txt = f"CatalogSet[n_selected/n_catalogs= {len(self.target_catalogs)}/{len(self.catalogs)}] \n"
        txt += 'SELECT FILTER ============\n'
        for key, value in self._last_filter.items():
            prefix = "!" if self._last_mode == "exclude" and value is not None else ""
            txt += f"{prefix}{key:>11} = {value}\n"
        return txt

    def help(self):
        # Get all public methods from the class, excluding `help`
        methods = [
            (name, obj)
            for name, obj in inspect.getmembers(self.__class__, inspect.isfunction)
            if not name.startswith("_") and name != "help"
        ]

        # Build plain text list with parameters
        lines = []
        for name, func in methods:
            sig = inspect.signature(func)
            params = [str(p) for p in sig.parameters.values() if p.name != "self"]
            sig_str = f"({', '.join(params)})" if params else "()"
            lines.append(f"- {name}{sig_str}")

        # Final plain text output
        help_text = ""
        print(f"Help for {self.__class__.__name__}\n{help_text}\n\nPublic methods:\n" + "\n".join(lines))
    
    def search_catalogs(self,
                        target_name: str,
                        search_key: str = '.cat',
                        folder: str = None,
                        recursive: bool = True,
                        n_proc: int = 16):
        """
        Search for catalogs in the given folder.
        
        Parameters
        ----------
        target_name : str
            Name of the target.
        search_key : str
            Search key for catalogs.
        folder : str, optional
            Folder to search for catalogs.
        recursive : bool, optional
            If True, search recursively in the given folder.
        n_proc : int, optional
            Number of processes to use for loading catalogs.

        Returns
        -------
        succeeded_catalogs : list[Catalog]
            List of Catalog instances that were successfully loaded.
        failed_catalogs : list[str]
            List of paths that failed to load.
        skipped_catalogs : list[str]
            List of paths that were skipped.
        """
        if folder is None:
            folder = self.helper.config['SCIDATA_DIR']
        folder = Path(folder)

        target_dir = next(folder.rglob(f"*{target_name}*"), None)
        print(f"Folder found: {target_dir}")
        if target_dir is None or not target_dir.is_dir():
            print(f"[WARNING] No matching subdirectory for target_name: {target_name}")
            return

        catalog_files = list(target_dir.rglob(f"*{search_key}")) if recursive else list(target_dir.glob(f"*{search_key}"))
        print(f"Catalog files: {len(catalog_files)}")

        existing_paths = [str(cat.path) for cat in self.catalogs]
        args = [(catalog_file, existing_paths) for catalog_file in catalog_files]

        succeeded_catalogs = []
        failed_catalogs = []
        skipped_catalogs = []

        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            results = list(tqdm(executor.map(self._load_catalog_worker, args), total=len(args), desc="Loading catalogs"))

        for status, path, catalog in results:
            if status == 'success':
                succeeded_catalogs.append(catalog)
                print(f"[LOADED] {path}")
            elif status == 'skipped':
                skipped_catalogs.append(path)
                print(f"[SKIPPED] {path}")
            else:
                failed_catalogs.append(path)
                print(f"[FAILED] {path}")

        self.catalogs.extend(succeeded_catalogs)
        self.catalogs.sort(key=lambda x: x.path)

        return succeeded_catalogs, failed_catalogs, skipped_catalogs
    
    def merge_catalogs(self,
        max_distance_arcsec=1.0,
        ra_key='X_WORLD',
        dec_key='Y_WORLD',
        data_keys=['MAGSKY_AUTO', 'MAGERR_AUTO', 'MAGSKY_APER', 'MAGERR_APER',
                'MAGSKY_APER_1', 'MAGERR_APER_1', 'MAGSKY_APER_2', 'MAGERR_APER_2',
                'MAGSKY_APER_3', 'MAGERR_APER_3', 'MAGSKY_CIRC', 'MAGERR_CIRC'],
        join_type='outer'
        ):
        """
        Merge catalogs into single table.
        
        Parameters
        ----------
        max_distance_arcsec : float, optional
            Maximum distance in arcseconds for matching sources.
        ra_key : str, optional
            Column name for right ascension.
        dec_key : str, optional
            Column name for declination.
        data_keys : list[str], optional
            List of column names for data.
        join_type : str, optional
            Type of join to use.

        Returns
        -------
        merged_tbl : astropy.table.Table
            Merged table of catalogs.
        metadata : dict
            Metadata of catalogs.
        """
        from astropy.coordinates import SkyCoord
        import numpy as np
        import pandas as pd
        import astropy.units as u
        from tqdm import tqdm

        catalogs = self.target_catalogs
        dfs = []
        coords = []
        metadata = {}

        # Step 1: Load and preprocess all catalogs
        for i, catalog in tqdm(enumerate(catalogs), total=len(catalogs), desc="Preparing catalogs"):
            tbl = catalog.target_data.copy()
            ra = tbl[ra_key]
            dec = tbl[dec_key]
            mask = np.isfinite(ra) & np.isfinite(dec)
            tbl = tbl[mask]
            if len(tbl) == 0 or np.sum(np.isfinite(tbl[ra_key]) & np.isfinite(tbl[dec_key])) == 0:
                # Still create a dummy DataFrame with NaNs
                n_dummy = 1  # You can make this 1 or 0, depending on downstream needs
                row = {'ra': [0] * n_dummy, 'dec': [0] * n_dummy}
                for key in data_keys:
                    colname = f"{key}_idx{i}"
                    row[colname] = [np.nan] * n_dummy
                df = pd.DataFrame(row)
                df['catalog_id'] = i
                df['match_id'] = -1
                dfs.append(df)
                coords.append(SkyCoord([0]*n_dummy * u.deg, [0]*n_dummy * u.deg))  # dummy coords
                metadata[i] = catalog.info.to_dict()
                continue

            metadata[i] = catalog.info.to_dict()

            row = {'ra': tbl[ra_key], 'dec': tbl[dec_key]}
            for key in data_keys:
                colname = f"{key}_idx{i}"
                row[colname] = tbl[key] if key in tbl.colnames else np.full(len(tbl), np.nan)

            df = pd.DataFrame(row)
            df['catalog_id'] = i
            df['match_id'] = -1  # placeholder
            dfs.append(df)
            coords.append(SkyCoord(df['ra'].values * u.deg, df['dec'].values * u.deg))

        if len(dfs) == 0:
            return None, {}

        # Step 2: Initialize merged_df with first catalog
        merged_df = dfs[0].copy()
        merged_df['match_id'] = np.arange(len(merged_df))

        for i in tqdm(range(1, len(dfs)), desc="Merging catalogs"):
            c1 = SkyCoord(merged_df['ra'].values * u.deg, merged_df['dec'].values * u.deg)
            c2 = coords[i]
            df2 = dfs[i].copy()

            # Match c2 ? c1
            idx, d2d, _ = c2.match_to_catalog_sky(c1)
            sep_mask = d2d.arcsec < max_distance_arcsec
            df2.loc[sep_mask, 'match_id'] = merged_df.iloc[idx[sep_mask]]['match_id'].values

            matched = df2[df2['match_id'] >= 0].copy()
            unmatched = df2[df2['match_id'] < 0].copy()

            # Assign new match_id for unmatched
            if len(unmatched) > 0:
                unmatched['match_id'] = np.arange(
                    merged_df['match_id'].max() + 1,
                    merged_df['match_id'].max() + 1 + len(unmatched)
                )

            # Avoid duplicated column merge
            matched = matched[[col for col in matched.columns if col not in merged_df.columns or col == 'match_id']]
            merged_df = pd.merge(merged_df, matched, on='match_id', how=join_type)

            if join_type == 'outer' and len(unmatched) > 0:
                merged_df = pd.concat([merged_df, unmatched], ignore_index=True)

        # Step 3: Add detection count
        main_key = data_keys[0]
        match_cols = [col for col in merged_df.columns if col.startswith(main_key)]
        merged_df['n_detections'] = merged_df[match_cols].notna().sum(axis=1)
        # Remove columns that are completely NaN (dummy columns)
        idx_cols = [col for col in merged_df.columns if '_idx' in col]
        is_dummy_row = merged_df[idx_cols].isna().all(axis=1)
        merged_df = merged_df[~is_dummy_row].copy()
        merged_tbl = Table.from_pandas(merged_df)
        
        # Add coord column
        coord = SkyCoord(ra=merged_tbl['ra'] * u.deg, dec=merged_tbl['dec'] * u.deg)
        merged_tbl['coord'] = coord
        return merged_tbl, metadata

    def exclude_catalogs(self, 
                        file_key=None,
                        filter=None, 
                        exptime=None, 
                        objname=None, 
                        obs_start=None, 
                        obs_end=None,
                        seeing=None,
                        depth=None,
                        observatory=None,
                        telname=None):
        """
        Exclude catalogs that match the given criteria from self.catalogs.
        
        Select catalogs from self.catalogs and update self.target_catalogs.
        
        Parameters
        ----------
        file_key : str, optional
            File key to exclude.
        filter : str, optional
            Filter to exclude.
        exptime : float, optional
            Exposure time to exclude.
        objname : str, optional
            Object name to exclude.
        obs_start : str, optional
            Observation start time to exclude.
        obs_end : str, optional 
            Observation end time to exclude.
        seeing : float, optional
            Seeing to exclude.
        depth : float, optional
            Depth to exclude.
        observatory : str, optional
            Observatory to exclude.
        telname : str, optional
            Telescope name to exclude.

        Returns
        -------
        None
        """
        df = self.df

        # Convert inputs to arrays
        if file_key is not None:
            file_key = np.atleast_1d(file_key)
            for key in file_key:
                key = key.replace('*', '') if '*' in key else key
                df = df[~df['path'].str.contains(key)]

        if filter is not None:
            filter = np.atleast_1d(filter)
            df = df[~df['filter'].isin(filter)]
            
        if exptime is not None:
            exptime = np.atleast_1d(exptime)
            df = df[~df['exptime'].isin(exptime)]
            
        if objname is not None:
            objname = np.atleast_1d(objname)
            df = df[~df['objname'].isin(objname)]
            
        if obs_start is not None:
            obs_start = self.helper.flexible_time_parser(obs_start)
            df = df[Time(df['obsdate'].tolist()) < obs_start]
            
        if obs_end is not None:
            obs_end = self.helper.flexible_time_parser(obs_end)
            df = df[Time(df['obsdate'].tolist()) > obs_end]
            
        if seeing is not None:
            df = df[df['seeing'] >= seeing]
            
        if depth is not None:
            df = df[df['depth'] <= depth]
            
        if observatory is not None:
            observatory = np.atleast_1d(observatory)
            df = df[~df['observatory'].isin(observatory)]
            
        if telname is not None:
            telname = np.atleast_1d(telname)
            df = df[~df['telname'].isin(telname)]

        # Update target_catalogs
        if df.empty:
            self.target_catalogs = []
        else:
            self.target_catalogs = [self.catalogs[i] for i in df.index]

        self._last_filter = {
            'file_key': file_key,
            'filter': filter,
            'exptime': exptime,
            'objname': objname,
            'obs_start': obs_start,
            'obs_end': obs_end,
            'seeing': seeing,
            'depth': depth,
            'observatory': observatory,
            'telname': telname,
        }

        print(f"[INFO] Excluded catalogs based on given criteria. Remaining: {len(self.target_catalogs)}")

    def select_catalogs(self, 
                        file_key=None,
                        filter=None, 
                        exptime=None, 
                        objname=None, 
                        obs_start=None, 
                        obs_end=None,
                        seeing=None,
                        depth=None,
                        observatory=None,
                        telname=None):
        """
        Select catalogs that match the given criteria from self.catalogs.
        
        Select catalogs from self.catalogs and update self.target_catalogs.
        
        Parameters
        ----------
        file_key : str, optional
            File key to select.
        filter : str, optional
            Filter to select.   
        exptime : float, optional
            Exposure time to select.
        objname : str, optional
            Object name to select.
        obs_start : str, optional
            Observation start time to select.
        obs_end : str, optional
            Observation end time to select.
        seeing : float, optional
            Seeing to select.
        depth : float, optional
            Depth to select.
        observatory : str, optional
            Observatory to select.
        telname : str, optional
            Telescope name to select.

        Returns
        -------
        None
        """
        df = self.df

        # Convert inputs to arrays
        if file_key is not None:
            file_key = np.atleast_1d(file_key)
            for key in file_key:
                key = key.replace('*', '') if '*' in key else key
                df = df[df['path'].str.contains(key)]

        if filter is not None:
            filter = np.atleast_1d(filter)
            df = df[df['filter'].isin(filter)]
            
        if exptime is not None:
            exptime = np.atleast_1d(exptime)
            df = df[df['exptime'].isin(exptime)]
            
        if objname is not None:
            objname = np.atleast_1d(objname)
            df = df[df['objname'].isin(objname)]
            
        if obs_start is not None:
            obs_start = self.helper.flexible_time_parser(obs_start)
            df = df[Time(df['obsdate'].tolist()) >= obs_start]
            
        if obs_end is not None:
            obs_end = self.helper.flexible_time_parser(obs_end)
            df = df[Time(df['obsdate'].tolist()) <= obs_end]
            
        if seeing is not None:
            df = df[df['seeing'] < seeing]
            
        if depth is not None:
            df = df[df['depth'] > depth]
            
        if observatory is not None:
            observatory = np.atleast_1d(observatory)
            df = df[df['observatory'].isin(observatory)]
            
        if telname is not None:
            telname = np.atleast_1d(telname)
            df = df[df['telname'].isin(telname)]

        # Update target_catalogs
        if df.empty:
            self.target_catalogs = []
        else:
            self.target_catalogs = [self.catalogs[i] for i in df.index]

        self._last_filter = {
            'file_key': file_key,
            'filter': filter,
            'exptime': exptime,
            'objname': objname,
            'obs_start': obs_start,
            'obs_end': obs_end,
            'seeing': seeing,
            'depth': depth,
            'observatory': observatory,
            'telname': telname,
        }
        self._last_mode = "select"  # <-- mark as select

    def add_catalogs(self,
                     catalogs: Union[List[Catalog], Catalog]):
        """Add catalogs to the dataset."""
        if isinstance(catalogs, Catalog):
            catalogs = [catalogs]
            
        succeeded_catalogs = []
        failed_catalogs = []
        skipped_catalogs = []
        existing_paths = [cat.path for cat in self.catalogs]
        for catalog in catalogs:
            if catalog.path in existing_paths:
                print(f"[WARNING] Catalog already exists: {catalog.path}")
                skipped_catalogs.append(catalog.path)
                continue
            if not catalog.is_loaded:
                load_result = catalog.load_target_img(target_img = None)
                if load_result is False:
                    original_img = catalog.find_corresponding_fits()
                    if original_img is not None:
                        target_img = ScienceImage(original_img, telinfo = self.helper.estimate_telinfo(original_img), load=True)
                        load_result = catalog.load_target_img(target_img=target_img)
            if catalog.is_loaded:
                print(f"Loaded catalog: {catalog_file}")
                succeeded_catalogs.append(catalog)
            else:
                print(f"[ERROR] Failed to load catalog: {catalog_file}")
                failed_catalogs.append(catalog_file)
            
        self.catalogs.extend(succeeded_catalogs)
        self.catalogs.sort(key=lambda x: x.path)
        return succeeded_catalogs, failed_catalogs, skipped_catalogs

    def select_sources(self, ra, dec, radius = 60):
        """
        Select sources from all catalogs within the given radius around the input coordinates.
        
        Each catalog will be updated with the selected sources.
        
        Parameters
        ----------
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.
        radius : float
            Search radius in arcseconds.
        
        Returns
        -------
        None
        """
        results = []
        for cat in tqdm(self.target_catalogs, desc = 'Selecting sources...'):
            cat.select_sources(ra, dec, unit='coord', matching_radius=radius)
        
    @property
    def df(self):
        """
        Return a DataFrame containing metadata of all catalogs.
        """
        if len(self.catalogs) == 0:
            return pd.DataFrame()
        rows = []
        for cat in self.catalogs:
            info = cat.info
            rows.append({
                'catalog': cat,
                'path': info.path,
                'filter': info.filter,
                'exptime': info.exptime,
                'obsdate': info.obsdate,
                'observatory': info.observatory,
                'telname': info.telname,
                'objname': info.objname,
                'seeing': info.seeing,
                'depth': info.depth,
                'ra': info.ra,
                'dec': info.dec,
                'fov_ra': info.fov_ra,
                'fov_dec': info.fov_dec,
            })
        return pd.DataFrame(rows)

    def _load_catalog_worker(self, args):
        catalog_file, existing_paths = args
        try:
            if str(catalog_file) in existing_paths:
                return 'skipped', str(catalog_file), None

            catalog = Catalog(path=catalog_file, catalog_type='all', load=True)

            if not catalog.is_loaded:
                load_result = catalog.load_target_img(target_img=None)

            if catalog.is_loaded:
                return 'success', str(catalog_file), catalog
            else:
                return 'failed', str(catalog_file), None

        except Exception as e:
            return 'failed', str(catalog_file), None
        
    def _search_sources_worker(self, args):
        catalog, ra, dec, radius = args
        matched, sep, _ = catalog.search_sources(ra, dec, unit='coord', matching_radius=radius)
        if len(matched) > 0:
            return (catalog.info, matched, sep)
        return None

#%%
if __name__ == "__main__":
    self = CatalogSet()
    target_name = 'T01358'
    folder: str = None
    recursive: bool = True
    load: bool = True
    search_key = '100.fits.cat'  # Default search key for catalogs
    succeeded, failed, skipped = self.search_catalogs(
        target_name=target_name,
        folder=folder,
        recursive=recursive,
        search_key=search_key,
        n_proc = 16
    )
