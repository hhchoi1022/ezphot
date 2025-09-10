

#%%
import inspect
from pathlib import Path
from typing import List, Union
from functools import reduce
import gc

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.io import ascii
from tqdm import tqdm
import ccdproc
from ccdproc import CCDData

from ezphot.helper import Helper
from ezphot.imageobjects import ScienceImage, CalibrationImage
from ezphot.methods import Stack


#%%
class Preprocess:
    """
    Preprocess the image.
    
    This class provides methods 
    
    1. Get master frame from the image.
    
    2. Correct bias, dark, and flat.
    
    3. Correct bias and dark.
    
    4. Correct flat.
    
    """
    
    def __init__(self):
        self.helper = Helper()
        self._cached_masterframe_tbl = None

    def __repr__(self):
        return f"Method class: {self.__class__.__name__}\n For help, use 'help(self)' or `self.help()`."

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
        self.helper.print(f"Help for {self.__class__.__name__}\n{help_text}\nPublic methods:\n" + "\n".join(lines), True)

    def get_masterframe_from_image(self,
                                   target_img: ScienceImage or CalibrationImage,
                                   imagetyp: str = 'BIAS',
                                   max_days: float = 5):
        """
        Get master frame from the image.
        
        This method will search for the master frame in the master frame directory.
        
        Parameters
        ----------
        target_img : ScienceImage or CalibrationImage
            The target image to get the master frame from.
        imagetyp : str, optional
            The type of image to get the master frame from.
        max_days : float, optional
            The maximum number of days to search for the master frame.
            
        Returns
        -------
        master_img : CalibrationImage
            The master frame image.
        master_frames_tbl : Table
            Metadata of the master frame(s) found.
        """
        from ezphot.imageobjects import CalibrationImage
        if not target_img.is_header_loaded:
            target_img.header

        master_frames_tbl = self.get_masterframe(
            observatory= target_img.observatory,
            telkey = target_img.telkey,
            telname = target_img.telname,
            imagetyp = imagetyp,
            exptime = target_img.exptime,
            filter_ = target_img.filter,
            obsdate = target_img.obsdate,
            max_days = max_days)
        
        master_img = CalibrationImage(master_frames_tbl[0]['file']) if master_frames_tbl else None
        
        return master_img, master_frames_tbl
        
    def get_masterframe(self,
                        observatory: str = '7DT',
                        telkey: str = '7DT_C361K_HIGH_1x1',
                        telname: str = '7DT02',
                        imagetyp: str = 'BIAS',
                        exptime: float = None,
                        filter_: str = None,
                        obsdate: str = None,
                        max_days: float = 5,
                        verbose: bool = True):
        """
        Get master frame from the image.
        
        This method will search for the master frame in the master frame directory.
        
        Parameters
        ----------
        observatory : str, optional
            The observatory to get the master frame from.
        telkey : str, optional
            The telescope key to get the master frame from.
        telname : str, optional
            The telescope name to get the master frame from.
        imagetyp : str, optional
            The type of image to get the master frame from.
        exptime : float, optional
            The exposure time of the image to get the master frame from.
        filter_ : str, optional
            The filter of the image to get the master frame from.
        obsdate : str, optional
            The observation date of the image to get the master frame from.
        max_days : float, optional
            The maximum number of days to search for the master frame.
            
        Returns
        -------
        master_frames : Table
            Metadata of the master frame(s) found.
        """
        from astropy.time import Time
        import numpy as np

        # Load summary tables
        all_masterframe_info = {}
        masterframe_summary_path = Path(self.helper.config['CALIBDATA_MASTERDIR']) / 'summary.ascii_fixed_width'
        
        if self._cached_masterframe_tbl is None:
            if masterframe_summary_path.exists():
                all_masterframe_tbl = ascii.read(masterframe_summary_path, format='fixed_width')
            else:
                self.helper.print(f"[WARNING] Master frame summary file not found: {masterframe_summary_path}", verbose)
                return 
            self._cached_masterframe_tbl = all_masterframe_tbl

        all_masterframe_tbl = self._cached_masterframe_tbl

        if len(all_masterframe_tbl) == 0:
            raise FileNotFoundError("No calibration frame metadata found.")
        
        if obsdate is None:
            self.helper.print('No obsdate provided. Returning None', verbose)
            return

        # === Base mask ===
        mask = np.ones(len(all_masterframe_tbl), dtype=bool)

        # Apply filters
        if observatory is not None:
            mask &= (all_masterframe_tbl['observatory'] == observatory)

        if telkey is not None:
            mask &= (all_masterframe_tbl['telkey'] == telkey)

        if telname is not None:
            mask &= (all_masterframe_tbl['telname'] == telname)

        if imagetyp is not None:
            mask &= (all_masterframe_tbl['imagetyp'] == imagetyp.upper())

        if imagetyp.upper() == 'DARK' and exptime is not None and 'exptime' in all_masterframe_tbl.colnames:
            mask &= np.isclose(all_masterframe_tbl['exptime'], exptime)
            
        elif imagetyp.upper() == 'FLAT' and filter_ is not None and 'filtername' in all_masterframe_tbl.colnames:
            mask &= (all_masterframe_tbl['filtername'] == filter_)

        all_masterframe_tbl = all_masterframe_tbl[mask]
        # === group_id logic ===
        use_group_id = False
        valid_group_id = None

        obsdate_target = self.helper.flexible_time_parser(obsdate).mjd
        obs_times = Time(all_masterframe_tbl['obsdate'], format='isot').mjd
        delta_t = np.abs(obs_times - obsdate_target)
        all_masterframe_tbl['time_difference'] = delta_t

        if 'group_id' in all_masterframe_tbl.colnames:
            # Try to find the nearest row with a valid group_id
            sorted_indices = np.argsort(delta_t)
            for idx in sorted_indices:
                gid = all_masterframe_tbl['group_id'][idx]
                if gid >= 0:
                    valid_group_id = gid
                    use_group_id = True
                    break  # stop at the first valid group_id

        if use_group_id:
            sub_mask = (all_masterframe_tbl['group_id'] == valid_group_id)
        else:
            # Fallback to time-based filtering
            delta_t = all_masterframe_tbl['time_difference']
            within_range = delta_t < max_days
            if not np.any(within_range):
                self.helper.print("No calibration frame found within max_days.", verbose)
                return None
            sub_mask = within_range

        # === Final filtering ===
        filtered_tbl = all_masterframe_tbl[sub_mask]
        
        base_dir = Path(self.helper.config['CALIBDATA_MASTERDIR'])
        filtered_tbl['file'] = [str((base_dir / Path(fname))) for fname in filtered_tbl['file']]

        if len(filtered_tbl) == 0:
            self.helper.print("No calibration frame found with the specified criteria.", verbose)
            return None

        filtered_tbl.sort('time_difference')

        return filtered_tbl

    def correct_bdf(self, target_img: ScienceImage, 
                    bias_image: CalibrationImage, 
                    dark_image: CalibrationImage, flat_image: CalibrationImage,
                    save : bool = False,
                    verbose: bool = True,
                    **kwargs
                    ):
        """
        Correct bias, dark, and flat.
        
        Parameters
        ----------
        target_img : ScienceImage
            The target image to correct bias, dark, and flat.
        bias_image : CalibrationImage
            The bias image to correct bias, dark, and flat.
        dark_image : CalibrationImage
            The dark image to correct bias, dark, and flat.
        flat_image : CalibrationImage
            The flat image to correct bias, dark, and flat.
        save : bool, optional
            Whether to save the corrected image.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        calib_image : ScienceImage
            The corrected image.
        """
        # if target_img.status.BIASCOR['status']:
        #     target_img.logger.warning(f"BIAS correction already applied to {target_img.path}. BIAS correction is not applied.")
        #     raise RuntimeError(f"BIAS correction already applied to {target_img.path}")
        # if target_img.status.DARKCOR['status']:
        #     target_img.logger.warning(f"DARK correction already applied to {target_img.path}. BIAS correction is not applied.")
        #     raise RuntimeError(f"DARK correction already applied to {target_img.path}")
        # if target_img.status.FLATCOR['status']:
        #     target_img.logger.warning(f"FLAT correction already applied to {target_img.path}. BIAS correction is not applied.")
        #     raise RuntimeError(f"FLAT correction already applied to {target_img.path}")

        # Convert input images to CCDData
        if target_img.data is None:
            target_img.load_data()
        if bias_image.data is None:
            bias_image.load_data()
        if dark_image.data is None:
            dark_image.load_data()
        if flat_image.data is None:
            flat_image.load_data()
        sci_ccddata = ccdproc.CCDData(data = target_img.data, meta = target_img.header, unit = 'adu')
        bias_ccddata = ccdproc.CCDData(data = bias_image.data, meta = bias_image.header, unit = 'adu')
        dark_ccddata = ccdproc.CCDData(data = dark_image.data, meta = dark_image.header, unit = 'adu')
        flat_ccddata = ccdproc.CCDData(data = flat_image.data, meta = flat_image.header, unit = 'adu')

        # Perform bias, dark, flat correction
        calib_data = self._correct_bdf(tgt_data = sci_ccddata, bias_data = bias_ccddata, dark_data = dark_ccddata, flat_data = flat_ccddata)        

        # Determine data types and convert to selected data type
        tgt_dtype = target_img.data.dtype
        bias_dtype = bias_image.data.dtype
        dark_dtype = dark_image.data.dtype
        flat_dtype = flat_image.data.dtype
        selected_dtype = reduce(np.promote_types, [tgt_dtype, bias_dtype, dark_dtype, flat_dtype])
        calib_data.data = calib_data.data.astype(selected_dtype)
        
        # Add metadata
        update_header_kwargs = dict(
            BIASCOR = True,
            BCORTIME = Time.now().isot,
            BIASPATH = str(bias_image.savepath.savepath),
            DARKCOR = True, 
            DCORTIME = Time.now().isot,
            DARKPATH = str(dark_image.savepath.savepath),
            FLATCOR = True,
            FCORTIME = Time.now().isot,
            FLATPATH = str(flat_image.savepath.savepath),
        )
        
        
        calib_data.header.update(update_header_kwargs)

        # Create new image object
        calib_image = type(target_img)(path  = target_img.path, telinfo = target_img.telinfo, status = target_img.status, load = False)
        
        calib_image.data = calib_data.data
        calib_image.header = calib_data.header
        calib_image.update_status(process_name= 'BIASCOR')
        calib_image.update_status(process_name = 'DARKCOR')
        calib_image.update_status(process_name = 'FLATCOR')
        # calib_image.logger.info(f"BIAS correction applied with {bias_image.savepath.savepath}")
        # calib_image.logger.info(f"DARK correction applied with {dark_image.savepath.savepath}")
        # calib_image.logger.info(f"FLAT correction applied with {flat_image.savepath.savepath}")
        # bias_image.logger.info(f"Used for BIAS correction: FILEPATH = {calib_image.savepath.savepath}")
        # dark_image.logger.info(f"Used for DARK correction: FILEPATH = {calib_image.savepath.savepath}")
        # flat_image.logger.info(f"Used for FLAT correction: FILEPATH = {calib_image.savepath.savepath}")
        if save:
            calib_image.write(verbose = verbose)
        return calib_image
    
    def _correct_bdf(self, tgt_data : CCDData, bias_data : CCDData, dark_data : CCDData, flat_data : CCDData):
        bcalib_data = self._correct_bias(tgt_data = tgt_data, bias_data = bias_data)
        dbcalib_data = self._correct_dark(tgt_data = bcalib_data, dark_data = dark_data)
        fdbcalib_data = self._correct_flat(tgt_data = dbcalib_data, flat_data = flat_data)
        return fdbcalib_data
    
    def correct_bd(self, target_img: ScienceImage, 
                   bias_image: CalibrationImage, 
                   dark_image: CalibrationImage, 
                   save : bool = False,
                   verbose: bool = True,
                   **kwargs
                   ):
        """
        Correct bias and dark.
        
        Parameters
        ----------
        target_img : ScienceImage   
            The target image to correct bias and dark.
        bias_image : CalibrationImage
            The bias image to correct bias and dark.
        dark_image : CalibrationImage
            The dark image to correct bias and dark.
        save : bool, optional
            Whether to save the corrected image.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        calib_image : ScienceImage
            The corrected image.
        """

        # Convert input images to CCDData
        if target_img.data is None:
            target_img.load_data()
        if bias_image.data is None:
            bias_image.load_data()
        if dark_image.data is None:
            dark_image.load_data()
        sci_ccddata = ccdproc.CCDData(data = target_img.data, meta = target_img.header, unit = 'adu')
        bias_ccddata = ccdproc.CCDData(data = bias_image.data, meta = bias_image.header, unit = 'adu')
        dark_ccddata = ccdproc.CCDData(data = dark_image.data, meta = dark_image.header, unit = 'adu')

        # Perform bias, dark correction
        calib_data = self._correct_bd(tgt_data = sci_ccddata, bias_data = bias_ccddata, dark_data = dark_ccddata)

        # Determine data types and convert to selected data type
        tgt_dtype = target_img.data.dtype
        bias_dtype = bias_image.data.dtype
        dark_dtype = dark_image.data.dtype
        selected_dtype = reduce(np.promote_types, [tgt_dtype, bias_dtype, dark_dtype])
        #selected_dtype = np.promote_types(tgt_dtype, bias_dtype, dark_dtype)
        calib_data.data = calib_data.data.astype(selected_dtype)
        
        # Add metadata
        update_header_kwargs = dict(
            BIASCOR = True,
            BCORTIME = Time.now().isot,
            BIASPATH = str(bias_image.savepath.savepath),
            DARKCOR = True, 
            DCORTIME = Time.now().isot,
            DARKPATH = str(dark_image.savepath.savepath),
        )
        
        calib_data.header.update(update_header_kwargs)

        # Create new image object
        calib_image = type(target_img)(path  = target_img.path, telinfo = target_img.telinfo, status = target_img.status, load = False)
        calib_image.data = calib_data.data
        calib_image.header = calib_data.header
        calib_image.update_status(process_name= 'BIASCOR')
        calib_image.update_status(process_name = 'DARKCOR')
        # calib_image.logger.info(f"BIAS correction applied with {bias_image.savepath.savepath}")
        # calib_image.logger.info(f"DARK correction applied with {dark_image.savepath.savepath}")
        # bias_image.logger.info(f"Used for BIAS correction: FILEPATH = {calib_image.savepath.savepath}")
        # dark_image.logger.info(f"Used for DARK correction: FILEPATH = {calib_image.savepath.savepath}")
        if save:
            calib_image.write(verbose = verbose)
        
        return calib_image
    
    def _correct_bd(self, tgt_data : CCDData, bias_data : CCDData, dark_data : CCDData):
        bcalib_data = self._correct_bias(tgt_data = tgt_data, bias_data = bias_data)
        dbcalib_data = self._correct_dark(tgt_data = bcalib_data, dark_data = dark_data)
        return dbcalib_data
        
    def correct_bias(self, target_img: ScienceImage or CalibrationImage, 
                     bias_image: CalibrationImage,
                     save : bool = False,
                     verbose: bool = True,
                     **kwargs
                     ):
        """
        Correct bias in the image.
        
        Parameters
        ----------
        target_img : ScienceImage or CalibrationImage
            The target image to correct bias.
        bias_image : CalibrationImage
            The bias image to correct bias.
        save : bool, optional
            Whether to save the corrected image.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        calib_image : ScienceImage
            The corrected image.
        """
        # if target_img.status.BIASCOR['status']:
        #     target_img.logger.warning(f"BIAS correction already applied to {target_img.path}. BIAS correction is not applied.")
        #     raise RuntimeError(f"BIAS correction already applied to {target_img.path}")
        
        # Convert input images to CCDData
        if target_img.data is None:
            target_img.load_data()
        if bias_image.data is None:
            bias_image.load_data()
        sci_ccddata = ccdproc.CCDData(data = target_img.data, meta = target_img.header, unit = 'adu')
        bias_ccddata = ccdproc.CCDData(data = bias_image.data, meta = bias_image.header, unit = 'adu')
        
        # Perform bias correction
        calib_data = self._correct_bias(tgt_data = sci_ccddata, bias_data = bias_ccddata)
        
        # Determine data types and convert to selected data type
        tgt_dtype = target_img.data.dtype
        bias_dtype = bias_image.data.dtype
        selected_dtype = np.promote_types(tgt_dtype, bias_dtype)
        calib_data.data = calib_data.data.astype(selected_dtype)

        # Update header
        update_header_kwargs = dict(
            BIASCOR = True,
            BCORTIME = Time.now().isot,
            BIASPATH = str(bias_image.savepath.savepath),
        )
        
        calib_data.header.update(update_header_kwargs)
        
        # Create new image object
        calib_image = type(target_img)(path  = target_img.path, telinfo = target_img.telinfo, status = target_img.status, load = False)
        calib_image.data = calib_data.data
        calib_image.header = calib_data.header
        calib_image.update_status(process_name= 'BIASCOR')
        # calib_image.logger.info(f"BIAS correction applied with {bias_image.savepath.savepath}")
        # bias_image.logger.info(f"Used for BIAS correction: FILEPATH = {calib_image.savepath.savepath}")
        if save:
            calib_image.write(verbose = verbose)
        return calib_image

    
    def _correct_bias(self, tgt_data : CCDData, bias_data : CCDData):
        calib_data = ccdproc.subtract_bias(tgt_data, bias_data)
        return calib_data
    
    def correct_dark(self, target_img: ScienceImage or CalibrationImage, 
                     dark_image: CalibrationImage, 
                     save : bool = False,
                     verbose: bool = True,
                     **kwargs
                     ):
        """
        Correct dark in the image.
        
        Parameters
        ----------
        target_img : ScienceImage or CalibrationImage
            The target image to correct dark.
        dark_image : CalibrationImage
            The dark image to correct dark.
        save : bool, optional
            Whether to save the corrected image.
        **kwargs : dict, optional
            Additional keyword arguments.
        
        Returns
        -------
        calib_image : ScienceImage
            The corrected image.
        """

        # Convert input images to CCDData
        if target_img.data is None:
            target_img.load_data()
        if dark_image.data is None:
            dark_image.load_data()
        sci_ccddata = ccdproc.CCDData(data = target_img.data, meta = target_img.header, unit = 'adu')
        dark_ccddata = ccdproc.CCDData(data = dark_image.data, meta = dark_image.header, unit = 'adu')
        
        # Perform dark correction
        calib_data = self._correct_dark(tgt_data = sci_ccddata, dark_data = dark_ccddata)
        
        # Determine data types and convert to selected data type
        tgt_dtype = target_img.data.dtype
        dark_dtype = dark_image.data.dtype
        selected_dtype = np.promote_types(tgt_dtype, dark_dtype)
        calib_data.data = calib_data.data.astype(selected_dtype)
        
        # Add metadata
        update_header_kwargs = dict(
            DARKCOR = True,
            DCORTIME = Time.now().isot,
            DARKPATH = str(dark_image.savepath.savepath),
        )
        
        calib_data.header.update(update_header_kwargs)
        
        # Create new image object
        calib_image = type(target_img)(path  = target_img.path, telinfo = target_img.telinfo, status = target_img.status, load = False)
        calib_image.data = calib_data.data
        calib_image.header = calib_data.header
        calib_image.update_status(process_name= 'DARKCOR')
        # calib_image.logger.info(f"DARK correction applied with {dark_image.savepath.savepath}")
        # dark_image.logger.info(f"Used for DARK correction: FILEPATH = {calib_image.savepath.savepath}")
        if save:
            calib_image.write(verbose = verbose)
        return calib_image

    def _correct_dark(self, tgt_data : CCDData, dark_data : CCDData):
        calib_data = ccdproc.subtract_dark(tgt_data, dark_data, scale = True, exposure_time = 'EXPTIME', exposure_unit = u.second)
        return calib_data
    
    def correct_flat(self, target_img: ScienceImage, flat_image: CalibrationImage,
                     save : bool = False,
                     verbose: bool = True,
                     **kwargs
                     ):
        """
        Correct flat in the image.
        
        Parameters
        ----------
        target_img : ScienceImage
            The target image to correct flat.
        flat_image : CalibrationImage
            The flat image to correct flat.
        save : bool, optional
            Whether to save the corrected image.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        calib_image : ScienceImage
            The corrected image.
        """
        # Convert input images to CCDData
        if target_img.data is None:
            target_img.load_data()
        if flat_image.data is None:
            flat_image.load_data()
        sci_ccddata = ccdproc.CCDData(data = target_img.data, meta = target_img.header, unit = 'adu')
        flat_ccddata = ccdproc.CCDData(data = flat_image.data, meta = flat_image.header, unit = 'adu')
        
        # Perform flat correction
        calib_data = self._correct_flat(tgt_data = sci_ccddata, flat_data = flat_ccddata)
        
        # Determine data types and convert to selected data type
        tgt_dtype = target_img.data.dtype
        flat_dtype = flat_image.data.dtype
        selected_dtype = np.promote_types(tgt_dtype, flat_dtype)
        calib_data.data = calib_data.data.astype(selected_dtype)
        
        # Update header
        update_header_kwargs = dict(
            FLATCOR = True,
            FCORTIME = Time.now().isot,
            FLATPATH = str(flat_image.savepath.savepath),
        )
        
        calib_data.header.update(update_header_kwargs)
        
        # Create new image object
        calib_image = type(target_img)(path  = target_img.path, telinfo = target_img.telinfo, status = target_img.status, load = False)
        calib_image.data = calib_data.data
        calib_image.header = calib_data.header
        calib_image.update_status(process_name= 'FLATCOR')
        # calib_image.logger.info(f"FLAT correction applied with {flat_image.savepath.savepath}")
        # flat_image.logger.info(f"Used for FLAT correction: FILEPATH = {calib_image.savepath.savepath}")
        if save:
            calib_image.write(verbose = verbose)
        return calib_image
        
    def _correct_flat(self, tgt_data : CCDData, flat_data : CCDData):
        calib_data = ccdproc.flat_correct(tgt_data, flat_data)
        return calib_data
    
    def generate_masterframe(self, calib_imagelist : List[CalibrationImage], 
                             mbias : Union[CalibrationImage or List[CalibrationImage]] = None,
                             mdark : Union[CalibrationImage or List[CalibrationImage]] = None,
                             
                             # Combine parameters
                             combine_type: str = 'median',
                             n_proc: int = 4,
                             clip_type: str = 'extrema',
                             sigma: float = 3.0,
                             nlow: int = 1,
                             nhigh: int = 1,
                            
                             # Other parameters
                             verbose: bool = True,
                             save: bool = True,
                             **kwargs
                             ):
        """
        Generate master bias, dark, flat frames.
        
        Parameters
        ----------
        calib_imagelist : List[CalibrationImage]
            The list of calibration images to generate master frames from.
        mbias : Union[CalibrationImage or List[CalibrationImage]], optional
            The master bias image to use for bias correction.
        mdark : Union[CalibrationImage or List[CalibrationImage]], optional
            The master dark image to use for dark correction.
        combine_type : str, optional
            The type of combination to use for the master frames.
        n_proc : int, optional
            The number of processors to use for the master frames.
        clip_type : str, optional
            The type of clipping to use for the master frames.
        sigma : float, optional
            The sigma for the master frames.
        nlow : int, optional
            The number of low values to clip for the master frames.
        nhigh : int, optional
            The number of high values to clip for the master frames.
        verbose : bool, optional
            Whether to print verbose output.
        save : bool, optional
            Whether to save the master frames.
        **kwargs : dict, optional   
            Additional keyword arguments.

        Returns
        -------
        master_files : dict
            The master frames.
        """
        
        def empty_memory(imagelist : Union[CalibrationImage or List[CalibrationImage]]):
            if isinstance(imagelist, CalibrationImage):
                imagelist = [imagelist]
            for image in imagelist:
                image.data = None

        combiner = Stack()
        all_filelist = [image.path for image in calib_imagelist]
        all_fileinfo = self.helper.get_imginfo(all_filelist, normalize_key = True)
        all_fileinfo['image'] = calib_imagelist
        all_fileinfo_by_group = all_fileinfo.group_by(['binning', 'gain']).groups
        master_files = dict()
        for group in all_fileinfo_by_group:
            key = (group['binning'][0], group['gain'][0])
            master_files[key] = dict(BIAS = None, DARK = dict(), FLAT = dict())
        
        if mbias:
            if isinstance(mbias, CalibrationImage):
                mbias = [mbias]
            for bias in mbias:
                bias_key = (str(bias.binning), str(bias.gain))
                master_files[bias_key]['BIAS'] = bias
        if mdark:
            if isinstance(mdark, CalibrationImage):
                mdark = [mdark]
            for dark in mdark:
                header = dark.header
                dark_key = (str(dark.binning), str(dark.gain))
                master_files[dark_key]['DARK'][str(dark.exptime)] = dark
        
        # Run the calibration
        for group in all_fileinfo_by_group:
            # Separate the images by type
            key = (group['binning'][0], group['gain'][0])
            if not master_files[key]['BIAS']:
                bias_key = ['BIAS', 'ZERO']
                bias_mask  = np.isin(group['imgtype'], bias_key)
                bias_fileinfo = group[bias_mask]
                new_bias = None
                if bias_fileinfo:
                    bias_rep = bias_fileinfo[0]
                    bias_imagelist = bias_fileinfo['image']
                    new_bias, _ = combiner.stack_multiprocess(
                        target_imglist = bias_imagelist,
                        target_errormaplist= None,
                        target_outpath = None,
                        weight_outpath = None,
                        combine_type = combine_type,
                        n_proc = n_proc,
                        clip_type = clip_type,
                        sigma = sigma, 
                        nlow = nlow,
                        nhigh = nhigh,
                        verbose = verbose)
                    empty_memory(bias_imagelist)
                    
                    master_files[key]['BIAS'] = new_bias
                else:
                    new_bias = self.get_masterframe_from_image(target_img = group[0]['image'], imagetyp = 'BIAS')
                    master_files[key]['BIAS'] = new_bias
                                
                if save:
                    new_bias.write(verbose = verbose)
                
            if not master_files[key]['DARK']:
                if not master_files[key]['BIAS']:
                    raise ValueError("Master BIAS is not found.")
                
                dark_key = ['DARK']
                dark_mask  = np.isin(group['imgtype'], dark_key)
                dark_fileinfo = group[dark_mask]
                if dark_fileinfo:
                    dark_fileinfo_by_exptime = dark_fileinfo.group_by('exptime').groups
                    for dark_group in dark_fileinfo_by_exptime:
                        dark_rep = dark_group[0]
                        exptime_name = dark_rep['exptime']
                        b_darkimagelist = []
                        dark_imagelist = dark_group['image']
                        for dark in tqdm(dark_imagelist, desc = 'BIAS correction on DARK frames...'):
                            b_dark_image = self.correct_bias(
                                target_img = dark, 
                                bias_image = master_files[key]['BIAS'],
                                save = False)
                            b_darkimagelist.append(b_dark_image)
                        
                        empty_memory(dark_imagelist)
                        
                        del dark_imagelist
                        gc.collect()
                        
                        new_dark, _ = combiner.stack_multiprocess(
                            target_imglist = b_darkimagelist,
                            target_errormaplist= None,
                            target_outpath = None,
                            weight_outpath = None,
                            combine_type = combine_type,
                            n_proc = n_proc,
                            clip_type = clip_type,
                            sigma = sigma, 
                            nlow = nlow,
                            nhigh = nhigh,
                            verbose = verbose)   
                        master_files[key]['DARK'][exptime_name] = new_dark

                        if save:
                            new_dark.write(verbose = verbose)
                        
                        del b_darkimagelist
                        gc.collect()       
                  

                if '100.0' not in master_files[key]['DARK'].keys():
                    new_dark = self.get_masterframe(target_img = group[0]['image'],
                                                    imagetyp = 'DARK',
                                                    exptime = 100)
                    master_files[key]['DARK']['100.0'] = new_dark
                
            
            if not master_files[key]['FLAT']:    
                if (not master_files[key]['DARK']['100.0']) or (not master_files[key]['BIAS']):
                    raise ValueError("Master BIAS or DARK frame not found.")

                flat_key = ['FLAT']
                flat_mask = np.isin(group['imgtype'], flat_key)
                flat_fileinfo = group[flat_mask]
                if flat_fileinfo:
                    flat_fileinfo_by_filter = flat_fileinfo.group_by('filter').groups
                    for flat_group in flat_fileinfo_by_filter:
                        flat_rep = flat_group[0]
                        filter_name = flat_rep['filter']
                        db_flatimagelist = []
                        flat_imagelist = flat_group['image']
                        for flat in tqdm(flat_imagelist, desc = 'BIAS, DARK correction on FLAT frames...'):
                            mbias = master_files[key]['BIAS']
                            # Convert list of exposure time strings to floats
                            mdark_exptime_key_float = [float(x) for x in master_files[key]['DARK'].keys()]

                            # Find the closest exposure time to 20.0 seconds
                            closest_exptime = min(mdark_exptime_key_float, key=lambda x: abs(x - 20.0))

                            # Use the string version as key to access master dark
                            closest_exptime_key = str(closest_exptime)       
                            mdark = master_files[key]['DARK'][closest_exptime_key]  
                                               
                            db_flat_image = self.correct_bd(
                                target_img = flat, 
                                bias_image = mbias, 
                                dark_image = mdark,
                                save = False)
                            db_flatimagelist.append(db_flat_image)
                        empty_memory(flat_imagelist)
                        
                        new_flat, _ = combiner.stack_multiprocess(
                            target_imglist = db_flatimagelist,
                            target_errormaplist= None,
                            target_outpath = None,
                            weight_outpath = None,
                            combine_type = combine_type,
                            n_proc = n_proc,
                            clip_type = clip_type,
                            sigma = sigma, 
                            nlow = nlow,
                            nhigh = nhigh,
                            verbose = verbose)    
                        master_files[key]['FLAT'][filter_name] = new_flat
                        if save:
                            new_flat.write(verbose = verbose)
                        
                        del db_flatimagelist
                        gc.collect()    
            
        return master_files
