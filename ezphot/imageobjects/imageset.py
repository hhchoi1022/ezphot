
#%%
import inspect
from typing import Union, List
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.time import Time

from ezphot.helper import Helper
from ezphot.imageobjects import ScienceImage, ReferenceImage

#%%
def _extract_info(image):
    """Extract lightweight info without forcing full lazy load."""
    info = image.info  # This uses lazy header reading internally
    return dict(
        image=image,
        path=info.SAVEPATH,
        filter=info.FILTER,     # ? FIXED typo from FILER to FILTER
        exptime=info.EXPTIME,
        obsdate=info.OBSDATE,
        observatory=info.OBSERVATORY,
        telname=info.TELNAME,
        objname=info.OBJNAME,
        seeing=info.SEEING,
        depth=info.DEPTH,
        ra=info.RA,
        dec=info.DEC,
        fov_ra=info.FOVX,
        fov_dec=info.FOVY,
    )

def _load_bkgmap(image):
    """Load background map from image."""
    return image.bkgmap

def _load_bkgrms(image):
    """Load background map from image."""
    return image.bkgrms

def _load_sourcemask(image):
    """Load source mask from image."""
    return image.sourcemask

def _load_catalog(image):
    """Load catalog from image."""
    return image.catalog

def _load_refcatalog(image):
    """Load reference catalog from image."""
    return image.refcatalog

class ImageSet:
    """
    Class representing a set of images.
    
    A container for managing and processing a list of ScienceImage or ReferenceImage objects.

    This class provides an interface to store a set of images and efficiently extract, filter,
    and manipulate background maps, source masks, and catalogs in parallel using multiprocessing.
    
    It supports image selection and exclusion based on header metadata such as filter, exposure time,
    observation date, seeing, depth, telescope, and observatory information.

    Parameters
    ----------
    images : List[ScienceImage] or List[ReferenceImage]
        List of images to be included in the set.
    """
    
    def __init__(self,
                 images: Union[List[ScienceImage], List[ReferenceImage]] = None):
        self.images = images if images is not None else []
        self._bkgmap = None
        self._bkgrms = None
        self._sourcemask = None
        self._catalog = None
        self._refcatalog = None
        self.target_images = self.images
        self._df = None
        self.helper = Helper()

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
        txt = f"ImageSet[n_selected/n_images= {len(self.target_images)}/{len(self.images)}] \n"
        txt += 'SELECTED FILTER ============\n'
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
    
    def clear(self):
        """Clear the image set."""
        for image in self.images:
            image.clear(clear_data=True, clear_header=False)
    
    def exclude_images(self,
                       file_key=None,
                       filter=None,
                       exptime=None,
                       objname=None,
                       obs_start=None,
                       obs_end=None,
                       seeing=None,
                       depth=None,
                       observatory=None,
                       telname=None,
                       ):
        """Exclude images based on given criteria. 
        
        One can access selected images via `.target_images` attribute.
        
        Parameters
        ----------
        file_key : str or list of str
            File key to exclude images.
        filter : str or list of str
            Filter to exclude images.
        exptime : float or list of float
            Exposure time to exclude images.
        objname : str or list of str
            Object name to exclude images.
        obs_start : str or list of str
            Observation start time to exclude images.
        obs_end : str or list of str
            Observation end time to exclude images.
        seeing : float or list of float
            Seeing to exclude images.
        depth : float or list of float
            Depth to exclude images.
        observatory : str or list of str
            Observatory to exclude images.
        telname : str or list of str
            Telescope name to exclude images.
            
        Returns
        -------
        None
        """
        df = self.df
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
            df = df[df['seeing'] > seeing]
        if depth is not None:
            df = df[df['depth'] < depth]
        if observatory is not None:
            observatory = np.atleast_1d(observatory)
            df = df[df['observatory'].isin(observatory)]
        if telname is not None:
            telname = np.atleast_1d(telname)
            df = df[df['telname'].isin(telname)]
        if df.empty:
            self.target_images = []
        else:
            self.target_images = [self.images[i] for i in df.index]
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
        self._last_mode = "exclude"  # <-- mark as exclude

    def select_images(self,
                      file_key=None,
                      filter=None,
                      exptime=None,
                      objname=None,
                      obs_start=None,
                      obs_end=None,
                      seeing=None,
                      depth=None,
                      observatory=None,
                      telname=None,
                      ):
        """Select images based on given criteria.
        
        One can access selected images via `.target_images` attribute.
        
        Parameters
        ----------
        file_key : str or list of str
            File key to select images.
        filter : str or list of str
            Filter to select images.
        exptime : float or list of float
            Exposure time to select images.
        objname : str or list of str
            Object name to select images.
        obs_start : str or list of str
            Observation start time to select images.
        obs_end : str or list of str
            Observation end time to select images.
        seeing : float or list of float
            Seeing to select images.
        depth : float or list of float
            Depth to select images.
        observatory : str or list of str
            Observatory to select images.
        telname : str or list of str
            Telescope name to select images.
            
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

        # Update target_images
        if df.empty:
            self.target_images = []
        else:
            self.target_images = [self.images[i] for i in df.index]
            
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
        
    def run_ds9(self):
        """Run DS9 on the image set.
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        all_imgpath = [img.path for img in self.target_images]
        self.helper.run_ds9(all_imgpath)
    
    @property
    def df(self):
        """Pandas DataFrame of the image set.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame of the image set.
        """
        if self._df is not None:
            return self._df
        if len(self.images) == 0:
            return pd.DataFrame()

        with ProcessPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(_extract_info, self.images), total=len(self.images), desc='Extracting info'))
        self._df = pd.DataFrame(results)
        return self._df
    
    @property
    def bkgrms(self):
        """Background RMS map of the image set.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        bkgrms : np.ndarray
            Background RMS maps of the image set.
        """
        if self._bkgrms is not None:
            return self._bkgrms
        if len(self.images) == 0:
            return None
        with ProcessPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(_load_bkgrms, self.target_images), total=len(self.target_images), desc='Loading bkgrms'))
        self._bkgrms = np.array(results)
        return self._bkgrms
    
    @property
    def bkgmap(self):
        """Background map of the image set.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        bkgmap : np.ndarray
            Background maps of the image set.
        """
        if self._bkgmap is not None:
            return self._bkgmap
        if len(self.images) == 0:
            return None
        with ProcessPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(_load_bkgmap, self.target_images), total=len(self.target_images), desc='Loading bkgmap'))
        self._bkgmap = np.array(results)
        return self._bkgmap
    
    @property
    def sourcemask(self):
        """Source mask of the image set.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        sourcemask : np.ndarray
            Source masks of the image set.
        """
        if self._sourcemask is not None:
            return self._sourcemask
        if len(self.images) == 0:
            return None
        with ProcessPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(_load_sourcemask, self.target_images), total=len(self.target_images), desc='Loading sourcemask'))
        self._sourcemask = np.array(results)
        return self._sourcemask
    
    @property
    def catalog(self):
        """Source catalog of the image set.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        catalog : np.ndarray
            Source catalogs of the image set.
        """
        if self._catalog is not None:
            return self._catalog
        if len(self.images) == 0:
            return None
        with ProcessPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(_load_catalog, self.target_images), total=len(self.target_images), desc='Loading catalog'))
        self._catalog = np.array(results)
        return self._catalog
    
    @property
    def refcatalog(self):
        """Reference catalog of the image set.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        refcatalog : np.ndarray
            Reference catalogs of the image set.
        """
        if self._refcatalog is not None:
            return self._refcatalog
        if len(self.images) == 0:
            return None
        with ProcessPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(_load_refcatalog, self.target_images), total=len(self.target_images), desc='Loading refcatalog'))
        self._refcatalog = np.array(results)
        return self._refcatalog