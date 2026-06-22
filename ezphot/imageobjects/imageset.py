
#%%
import inspect
from typing import Union, List
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.time import Time
from astropy.table import Table
from ezphot.helper import Helper
from ezphot.imageobjects import ScienceImage, ReferenceImage

#%%
def _extract_info_indexed(args):
    i, image = args
    info = image.info
    return dict(
        image_id=i,
        path=info.SAVEPATH,
        filter=info.FILTER,
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
        self.images = list(images) if images is not None else []
        self.image_ids = list(range(len(self.images)))
        self.target_image_ids = list(self.image_ids)

        self._df = None
        self._bkgmap = None
        self._bkgrms = None
        self._sourcemask = None
        self._catalog = None
        self._refcatalog = None
        
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
            self.target_image_ids = []
        else:
            self.target_image_ids = df["image_id"].astype(int).tolist()

        self._bkgmap = None
        self._bkgrms = None
        self._sourcemask = None
        self._catalog = None
        self._refcatalog = None

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
            Object name to select images.e
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
            self.target_image_ids = []
        else:
            self.target_image_ids = df["image_id"].astype(int).tolist()
        
        self._bkgmap = None
        self._bkgrms = None
        self._sourcemask = None
        self._catalog = None
        self._refcatalog = None

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
    
    def divide_images(self,
                      by_filter: bool = True,
                      by_exptime: bool = False,
                      by_objname: bool = False,
                      by_telname: bool = False,
                      by_observatory: bool = True,
                      by_obsdate: bool = True,
                      obsdate_delta: int = 0.5,
                      obsdate_key: str = 'obsdate'):
        """Divide the image set into groups based on given criteria.
        Parameters
        ----------
        None
        Returns
        -------
        None
        
        by_filter: bool = True
        by_exptime: bool = False
        by_objname: bool = False
        by_telname: bool = False
        by_observatory: bool = True
        by_obsdate: bool = True
        obsdate_delta: int = 0.5
        obsdate_key: str = 'obsdate'
        """
        group_keys = ['filter', 'exptime', 'objname', 'observatory', 'telname', 'group']
        group_keys_bools = [by_filter, by_exptime, by_objname, by_observatory, by_telname, by_obsdate]
        group_keys_applied = [key for key, bool in zip(group_keys, group_keys_bools) if bool]
        if by_obsdate:
            target_tbl = Table().from_pandas(self.target_df)
            target_tbl['mjd'] = Time(target_tbl[obsdate_key].tolist()).mjd
            target_tbl = self.helper.group_table(target_tbl, key = 'mjd', tolerance = obsdate_delta)
        else:
            target_tbl = Table().from_pandas(self.target_df)
        target_tbl_groups = target_tbl.group_by(group_keys_applied).groups
        all_imgsets = []
        for tbl in target_tbl_groups:
            group_imglist = [self.images[int(row['image_id'])] for row in tbl]
            group_imgset = ImageSet(group_imglist)
            all_imgsets.append(group_imgset)
        return all_imgsets
    
    def select_quality_images(self, 
                              mode: str = 'both',
                              min_obsdate: Union[Time, str, float] = None,
                              max_obsdate: Union[Time, str, float] = None,
                              seeing_key: str = 'SEEING',
                              depth_key: str = 'UL5SKY_APER_3',
                              ellipticity_key: str = 'ELLIP',
                              obsdate_key: str = 'DATE-OBS',
                              # For absolute filtering
                              seeing_limit: float = 6.0,
                              depth_limit: float = 15.0,
                              ellipticity_limit: float = 0.3,
                              rotang_abs_limit: float = 5,
                              # For relative filtering
                              ellipticity_percentile: float = 90.0,
                              seeing_percentile: float = 75.0,
                              depth_percentile: float = 75.0,
                              rotang_percentile: float = 100.0,
                              max_numbers: int = None,
                              # Other parameters
                              visualize: bool = True,
                              verbose: bool = True):
        """Select the best images based on seeing, depth, and ellipticity.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        from ezphot.methods import Stack
        stacker = Stack()
        target_images = stacker.select_quality_images(
            target_imglist = self.images,
            mode = mode,
            min_obsdate = min_obsdate,
            max_obsdate = max_obsdate,
            seeing_key = seeing_key,
            depth_key = depth_key,
            ellipticity_key = ellipticity_key,
            obsdate_key = obsdate_key,
            seeing_limit = seeing_limit,
            depth_limit = depth_limit,
            ellipticity_limit = ellipticity_limit,
            rotang_abs_limit = rotang_abs_limit,
            ellipticity_percentile = ellipticity_percentile,
            seeing_percentile = seeing_percentile,
            depth_percentile = depth_percentile,
            rotang_percentile = rotang_percentile,
            max_numbers = max_numbers,
            visualize = visualize,
            verbose = verbose
        )
        self._set_target_images_from_existing(target_images)
        return target_images
    
    def prepare_stack(self,
                      n_proc: int = 4,                       

                      # Scale parameters
                      scale: bool = True,
                      zp_key: str = 'ZP_APER_3',
                    
                      # Convolution parameters
                      convolve: bool = False,
                      seeing_key: str = 'SEEING', 

                      # Reproject parameters
                      reproject: bool = True,
                      reproject_type: str = 'LANCZOS3',
                      center_ra: float = None,
                      center_dec: float = None,
                      pixel_scale: float = None,
                      x_size: int = None,
                      y_size: int = None,     
                    
                      # Other parameters
                      verbose: bool = True,
                      save: bool = True,
                      clear: bool = True,
                      ):
        """Prepare the image set for stacking.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        from ezphot.methods import Stack
        stacker = Stack()
        prepared_imglist, prepared_bkgrmslist = stacker.prepare_images(
            target_imglist = self.target_images,
            target_bkglist = self.bkgmap,
            target_bkgrmslist = self.bkgrms,
            n_proc = n_proc,
            scale = scale,
            zp_key = zp_key,
            convolve = convolve,
            seeing_key = seeing_key,
            reproject = reproject,
            reproject_type = reproject_type,
            center_ra = center_ra,
            center_dec = center_dec,
            pixel_scale = pixel_scale,
            x_size = x_size,
            y_size = y_size,
            verbose = verbose,
            save = save,
            clear = clear,
        )
        self._replace_images(prepared_imglist)
        self._bkgrms = np.array(prepared_bkgrmslist)
        return prepared_imglist, prepared_bkgrmslist
    
    def stack(self, 
              target_outpath: str = None,
              bkgrms_outpath: str = None,
              n_proc: int = 8,
              combine_type: str = 'median',
              clip_type: str = None,
              sigma: float = 3.0,
              nlow: int = 1,
              nhigh: int = 1,
              verbose: bool = True,
              save: bool = True,
              remove_intermediate: bool = False,
              ):
        """Stack the image set.
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        if len(self.target_images) == 0:
            return None
        if len(self.target_images) == 1:
            return self.target_images[0]
        
        from ezphot.methods import Stack
        stacker = Stack()
        stacked_img, stacked_bkgrms = stacker.stack_multiprocess(
            target_imglist = self.target_images,
            target_bkgrmslist = self.bkgrms,
            target_outpath = target_outpath,
            bkgrms_outpath = bkgrms_outpath,
            combine_type = combine_type,
            n_proc = n_proc,
            clip_type = clip_type,
            sigma = sigma,
            nlow = nlow,
            nhigh = nhigh,
            verbose = verbose,
            save = save,
            remove_intermediate = remove_intermediate
        )
        return stacked_img, stacked_bkgrms
        
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
        if self._df is not None:
            return self._df
        if len(self.images) == 0:
            return pd.DataFrame()

        with ProcessPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(
                executor.map(_extract_info_indexed, zip(self.image_ids, self.images)),
                total=len(self.images),
                desc='Extracting info'
            ))

        self._df = pd.DataFrame(results)
        return self._df
    
    @property
    def target_df(self) -> pd.DataFrame:
        if not self.target_image_ids:
            return self.df.iloc[0:0].copy()

        return (
            self.df[
                self.df["image_id"].isin(self.target_image_ids)
            ]
            .copy()
            .reset_index(drop=True)
        )
        
    @property
    def target_images(self):
        return [
            self.images[image_id]
            for image_id in self.target_image_ids
        ]
        
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
            results = list(executor.map(_load_bkgrms, self.target_images))
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
            results = list(executor.map(_load_bkgmap, self.target_images))
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
            results = list(executor.map(_load_sourcemask, self.target_images))
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
            results = list(executor.map(_load_catalog, self.target_images))
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
            results = list(executor.map(_load_refcatalog, self.target_images))
        self._refcatalog = np.array(results)
        return self._refcatalog

    def _reset_target_cache(self):
        self._bkgmap = None
        self._bkgrms = None
        self._sourcemask = None
        self._catalog = None
        self._refcatalog = None


    def _set_target_images_from_existing(self, target_images):
        """Set target_image_ids from images already contained in self.images."""
        if target_images is None:
            self.target_image_ids = []
            self._reset_target_cache()
            return

        id_to_index = {id(img): i for i, img in enumerate(self.images)}

        target_ids = []
        for img in target_images:
            if id(img) not in id_to_index:
                raise ValueError(
                    "target_images must be a subset of self.images. "
                    "Use _replace_images() if these are newly created images."
                )
            target_ids.append(id_to_index[id(img)])

        self.target_image_ids = target_ids
        self._reset_target_cache()


    def _replace_images(self, images):
        """Replace the whole image set."""
        self.images = list(images) if images is not None else []
        self.image_ids = list(range(len(self.images)))
        self.target_image_ids = list(self.image_ids)

        self._df = None
        self._reset_target_cache()
