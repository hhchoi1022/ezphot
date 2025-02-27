

#%%
from astropy.time import Time
from typing import List, Dict, Union
import ccdproc
from ccdproc import CCDData
import astropy.units as u
import numpy as np
import glob, os
from tqdm import tqdm
from functools import reduce

from tippy.helper import Helper
from tippy.image import ScienceImage
from tippy.image import CalibrationImage
#%%
class TIPPreprocess(Helper):
    
    def __init__(self):
        super().__init__()

    def get_masterframe(self, tgt_image : ScienceImage or CalibrationImage, 
                        imagetyp: str = 'BIAS',
                        exptime : float = None,
                        binning : int = None,
                        filter_ : str = None,
                        gain : str = None):
        """
        Searches for the nearest matching calibration frame.

        Parameters
        ----------
        search_dir : str
            Directory to search for calibration frames.
        date_obs : str, optional
            Observation date in ISO format (YYYY-MM-DDTHH:MM:SS).
            If None, uses the current time.
        imagetyp : str, optional
            Type of calibration frame (default is 'BIAS').
        exptime : float, optional
            Exposure time for filtering (only used if relevant).
        binning : int, optional
            Binning factor (default is 1).
        filter_ : str, optional
            Filter name (only used for FLAT frames).
        gain : str, optional
            Gain setting (if applicable).

        Returns
        -------
        closest_frame : str
            Path of the nearest matching calibration frame.

        Raises
        ------
        FileNotFoundError
            If no calibration frames are found.
        """
        # Locate calibration frames
        calibframe_key = os.path.join(tgt_image.config['CALIBDATA_MASTERDIR'], tgt_image.observatory, tgt_image.telkey, imagetyp, tgt_image.telname, '*.fits')
        calibframe_files = glob.glob(calibframe_key)
        if not calibframe_files:
            raise FileNotFoundError(f"No calibration frames found in {os.path.join(tgt_image.config['CALIBDATA_MASTERDIR'], tgt_image.observatory, tgt_image.telkey, imagetyp, tgt_image.telname)}")

        # Retrieve FITS metadata
        calibframe_info = self.get_imginfo(calibframe_files, normalize_key=True)

        # Initialize filtering mask (all True initially)
        output_idx = np.ones(len(calibframe_info), dtype=bool)

        # Apply optional filters (only if the column exists in `calibframe_info`)
        tgt_gain = tgt_image.gain
        tgt_binning = tgt_image.binning
        if imagetyp.upper() == 'BIAS':
            tgt_exptime = None
            tgt_filter_= None
        elif imagetyp.upper() == 'DARK':
            tgt_exptime = tgt_image.exptime
            tgt_filter_ = None
        elif imagetyp.upper() == 'FLAT':
            tgt_exptime = None
            tgt_filter_ = tgt_image.filter
        else:
            raise ValueError(f"Invalid imagetyp: {imagetyp}")
        
        # Update filter values if specified
        if exptime:
            tgt_exptime = exptime
        if binning:
            tgt_binning = binning
        if filter_:
            tgt_filter_ = filter_
        if gain:
            tgt_gain = gain
        
        float_keys = {'exptime': tgt_exptime, 'gain': tgt_gain, 'binning': tgt_binning}
        for key, value in float_keys.items():
            if value is not None and key in calibframe_info.colnames:
                output_idx &= (np.array(calibframe_info[key]).astype(float) == float(value))

        # Apply optional filters (only if the column exists in `calibframe_info`)
        str_keys = {'imgtype': imagetyp, 'filter': tgt_filter_}
        for key, value in str_keys.items():
            if value is not None and key in calibframe_info.colnames:
                output_idx &= (np.array(calibframe_info[key]).astype(str) == str(value))

        # Check if any matching calibration frames remain
        if not np.any(output_idx):
            print("No matching calibration frame found after filtering.")
            return 

        # Select filtered calibration frames
        calibframe_filtered = calibframe_info[output_idx]

        # Convert date_obs to Astropy Time
        obsdate = tgt_image.obsdate
        # Find the calibration frame closest to date_obs
        if 'obsdate' in calibframe_filtered.colnames:
            calibframe_filtered['obsdate'] = Time(calibframe_filtered['obsdate'])  # Convert column to Time format
            time_diffs = np.abs(calibframe_filtered['obsdate'] - Time(obsdate))  # Compute time difference
            closest_idx = np.argmin(time_diffs)  # Find the index with the smallest time difference
            best_match = calibframe_filtered[closest_idx]
        else:
            # If no date-obs column, return the first match
            best_match = calibframe_filtered[0]

        return CalibrationImage(best_match['file'], telinfo = tgt_image.telinfo)  # Return the file path of the best matching calibration frame

    def correct_bdf(self, tgt_image: ScienceImage, bias_image: CalibrationImage, dark_image: CalibrationImage, flat_image: CalibrationImage,
                    output_dir : str = None
                    ):
        if tgt_image.status.status['biascor']['status']:
            tgt_image.logger.warning(f"BIAS correction already applied to {tgt_image.path}. BIAS correction is not applied.")
            raise RuntimeError(f"BIAS correction already applied to {tgt_image.path}")
        if tgt_image.status.status['darkcor']['status']:
            tgt_image.logger.warning(f"DARK correction already applied to {tgt_image.path}. DARK correction is not applied.")
            raise RuntimeError(f"DARK correction already applied to {tgt_image.path}")
        if tgt_image.status.status['flatcor']['status']:
            tgt_image.logger.warning(f"FLAT correction already applied to {tgt_image.path}. FLAT correction is not applied.")
            raise RuntimeError(f"FLAT correction already applied to {tgt_image.path}")
        
        # Convert input images to CCDData
        sci_ccddata = ccdproc.CCDData(data = tgt_image.data, meta = tgt_image.header, unit = 'adu')
        bias_ccddata = ccdproc.CCDData(data = bias_image.data, meta = bias_image.header, unit = 'adu')
        dark_ccddata = ccdproc.CCDData(data = dark_image.data, meta = dark_image.header, unit = 'adu')
        flat_ccddata = ccdproc.CCDData(data = flat_image.data, meta = flat_image.header, unit = 'adu')

        # Perform bias, dark, flat correction
        calib_data = self._correct_bdf(tgt_data = sci_ccddata, bias_data = bias_ccddata, dark_data = dark_ccddata, flat_data = flat_ccddata)        

        # Determine data types and convert to selected data type
        tgt_dtype = tgt_image.data.dtype
        bias_dtype = bias_image.data.dtype
        dark_dtype = dark_image.data.dtype
        flat_dtype = flat_image.data.dtype
        selected_dtype = reduce(np.promote_types, [tgt_dtype, bias_dtype, dark_dtype, flat_dtype])
        calib_data.data = calib_data.data.astype(selected_dtype)
        
        # Add metadata
        calib_data.meta['BIASCOR'] = True
        calib_data.meta['BCORTIME'] = Time.now().isot
        calib_data.meta['BIASPATH'] = bias_image.path
        calib_data.meta['DARKCOR'] = True
        calib_data.meta['DCORTIME'] = Time.now().isot
        calib_data.meta['DARKPATH'] = dark_image.path
        calib_data.meta['FLATCOR'] = True
        calib_data.meta['FCORTIME'] = Time.now().isot
        calib_data.meta['FLATPATH'] = flat_image.path
        
        # Determine output filename
        if not output_dir:
            output_dir = tgt_image.savedir
        os.makedirs(output_dir, exist_ok = True)
        
        filename = os.path.basename(tgt_image.path)
        if 'cor_' in filename:
            filepath = os.path.join(output_dir, 'fdb' + filename)
        else:
            filepath = os.path.join(output_dir, 'fdbcor_' + filename)
        calib_data.write(filepath, overwrite = True)
        
        # Create new image object
        calib_image = type(tgt_image)(path  = filepath, telinfo = tgt_image.telinfo)
        calib_image.logger.info(f"BIAS, DARK, FLAT correction applied with {bias_image.path}, {dark_image.path}, {flat_image.path}")
        calib_image.status = tgt_image.status
        calib_image.update_status(process_name = 'biascor')
        calib_image.update_status(process_name = 'darkcor')
        calib_image.update_status(process_name = 'flatcor')
        
        # Log information
        tgt_image.logger.info(f"BIAS, DARK, FLAT correction applied: FILEPATH = {calib_image.path}")
        bias_image.logger.info(f"Used for BIAS correction: FILEPATH = {calib_image.path}")
        dark_image.logger.info(f"Used for DARK correction: FILEPATH = {calib_image.path}")
        flat_image.logger.info(f"Used for FLAT correction: FILEPATH = {calib_image.path}")
        return calib_image
    
    def _correct_bdf(self, tgt_data : CCDData, bias_data : CCDData, dark_data : CCDData, flat_data : CCDData):
        bcalib_data = self._correct_bias(tgt_data = tgt_data, bias_data = bias_data)
        dbcalib_data = self._correct_dark(tgt_data = bcalib_data, dark_data = dark_data)
        fdbcalib_data = self._correct_flat(tgt_data = dbcalib_data, flat_data = flat_data)
        return fdbcalib_data
    
    def correct_db(self, tgt_image: ScienceImage, bias_image: CalibrationImage, dark_image: CalibrationImage, 
                    output_dir : str = None
                    ):
        if tgt_image.status.status['biascor']['status']:
            tgt_image.logger.warning(f"BIAS correction already applied to {tgt_image.path}. BIAS correction is not applied.")
            raise RuntimeError(f"BIAS correction already applied to {tgt_image.path}")
        if tgt_image.status.status['darkcor']['status']:
            tgt_image.logger.warning(f"DARK correction already applied to {tgt_image.path}. DARK correction is not applied.")
            raise RuntimeError(f"DARK correction already applied to {tgt_image.path}")
        
        # Convert input images to CCDData
        sci_ccddata = ccdproc.CCDData(data = tgt_image.data, meta = tgt_image.header, unit = 'adu')
        bias_ccddata = ccdproc.CCDData(data = bias_image.data, meta = bias_image.header, unit = 'adu')
        dark_ccddata = ccdproc.CCDData(data = dark_image.data, meta = dark_image.header, unit = 'adu')

        # Perform bias, dark correction
        calib_data = self._correct_bd(tgt_data = sci_ccddata, bias_data = bias_ccddata, dark_data = dark_ccddata)

        # Determine data types and convert to selected data type
        tgt_dtype = tgt_image.data.dtype
        bias_dtype = bias_image.data.dtype
        dark_dtype = dark_image.data.dtype
        selected_dtype = reduce(np.promote_types, [tgt_dtype, bias_dtype, dark_dtype])
        #selected_dtype = np.promote_types(tgt_dtype, bias_dtype, dark_dtype)
        calib_data.data = calib_data.data.astype(selected_dtype)
        
        # Add metadata
        calib_data.meta['BIASCOR'] = True
        calib_data.meta['BCORTIME'] = Time.now().isot
        calib_data.meta['BIASPATH'] = bias_image.path
        calib_data.meta['DARKCOR'] = True
        calib_data.meta['DCORTIME'] = Time.now().isot
        calib_data.meta['DARKPATH'] = dark_image.path
        
        # Determine output filename
        if not output_dir:
            output_dir = tgt_image.savedir
        os.makedirs(output_dir, exist_ok = True)
        
        filename = os.path.basename(tgt_image.path)
        if 'cor_' in filename:
            filepath = os.path.join(output_dir, 'db' + filename)
        else:
            filepath = os.path.join(output_dir, 'dbcor_' + filename)
        calib_data.write(filepath, overwrite = True)
        
        # Create new image object
        calib_image = type(tgt_image)(path  = filepath, telinfo = tgt_image.telinfo)
        calib_image.logger.info(f"BIAS, DARK correction applied with {bias_image.path}, {dark_image.path}")
        calib_image.status = tgt_image.status
        calib_image.update_status(process_name = 'biascor')
        calib_image.update_status(process_name = 'darkcor')
        
        # Log information
        tgt_image.logger.info(f"BIAS, DARK correction applied: FILEPATH = {calib_image.path}")
        bias_image.logger.info(f"Used for BIAS correction: FILEPATH = {calib_image.path}")
        dark_image.logger.info(f"Used for DARK correction: FILEPATH = {calib_image.path}")
        return calib_image
    
    def _correct_bd(self, tgt_data : CCDData, bias_data : CCDData, dark_data : CCDData):
        bcalib_data = self._correct_bias(tgt_data = tgt_data, bias_data = bias_data)
        dbcalib_data = self._correct_dark(tgt_data = bcalib_data, dark_data = dark_data)
        return dbcalib_data
        
    def correct_bias(self, tgt_image: ScienceImage or CalibrationImage, bias_image: CalibrationImage,
                     output_dir : str = None
                     ):
        """ Corrects bias in the image """
        if tgt_image.status.status['biascor']['status']:
            tgt_image.logger.warning(f"BIAS correction already applied to {tgt_image.path}. BIAS correction is not applied.")
            raise RuntimeError(f"BIAS correction already applied to {tgt_image.path}")
        
        # Convert input images to CCDData
        sci_ccddata = ccdproc.CCDData(data = tgt_image.data, meta = tgt_image.header, unit = 'adu')
        bias_ccddata = ccdproc.CCDData(data = bias_image.data, meta = bias_image.header, unit = 'adu')
        
        # Perform bias correction
        calib_data = self._correct_bias(tgt_data = sci_ccddata, bias_data = bias_ccddata)
        
        # Determine data types and convert to selected data type
        tgt_dtype = tgt_image.data.dtype
        bias_dtype = bias_image.data.dtype
        selected_dtype = np.promote_types(tgt_dtype, bias_dtype)
        calib_data.data = calib_data.data.astype(selected_dtype)

        # Add metadata
        calib_data.meta['BIASCOR'] = True
        calib_data.meta['BCORTIME'] = Time.now().isot
        calib_data.meta['BIASPATH'] = bias_image.path
        
        # Determine output filename
        if not output_dir:
            output_dir = tgt_image.savedir
        os.makedirs(output_dir, exist_ok = True)
        
        filename = os.path.basename(tgt_image.path)
        if 'cor_' in filename:
            filepath = os.path.join(output_dir, 'b' + filename)
        else:
            filepath = os.path.join(output_dir, 'bcor_' + filename)          
        calib_data.write(filepath, overwrite = True)
        
        # Create new image object
        calib_image = type(tgt_image)(path  = filepath, telinfo = tgt_image.telinfo)
        calib_image.logger.info(f"BIAS correction applied with {bias_image.path}")
        calib_image.status = tgt_image.status
        calib_image.update_status(process_name = 'biascor')
        
        # Log information
        tgt_image.logger.info(f"BIAS correction applied: FILEPATH = {calib_image.path}")
        bias_image.logger.info(f"Used for BIAS correction: FILEPATH = {calib_image.path}")
        return calib_image

    
    def _correct_bias(self, tgt_data : CCDData, bias_data : CCDData):
        calib_data = ccdproc.subtract_bias(tgt_data, bias_data)
        return calib_data
    
    def correct_dark(self, tgt_image: ScienceImage or CalibrationImage, dark_image: CalibrationImage, 
                     output_dir : str = None
                     ):
        """ Corrects dark in the image """
        if tgt_image.status.status['darkcor']['status']:
            tgt_image.logger.warning(f"DARK correction already applied to {tgt_image.path}. DARK correction is not applied.")
            raise RuntimeError(f"DARK correction already applied to {tgt_image.path}")
        
        # Convert input images to CCDData
        sci_ccddata = ccdproc.CCDData(data = tgt_image.data, meta = tgt_image.header, unit = 'adu')
        dark_ccddata = ccdproc.CCDData(data = dark_image.data, meta = dark_image.header, unit = 'adu')
        
        # Perform dark correction
        calib_data = self._correct_dark(tgt_data = sci_ccddata, dark_data = dark_ccddata)
        
        # Determine data types and convert to selected data type
        tgt_dtype = tgt_image.data.dtype
        dark_dtype = dark_image.data.dtype
        selected_dtype = np.promote_types(tgt_dtype, dark_dtype)
        calib_data.data = calib_data.data.astype(selected_dtype)
        
        # Add metadata
        calib_data.meta['DARKCOR'] = True
        calib_data.meta['DCORTIME'] = Time.now().isot
        calib_data.meta['DARKPATH'] = dark_image.path
        
        # Determine output filename
        if not output_dir:
            output_dir = tgt_image.savedir
        os.makedirs(output_dir, exist_ok = True)
        
        filename = os.path.basename(tgt_image.path)
        if 'cor_' in filename:
            filepath = os.path.join(output_dir, 'd' + filename)
        else:
            filepath = os.path.join(output_dir, 'dcor_' + filename)           
        calib_data.write(filepath, overwrite = True)
        
        # Create new image object
        calib_image = type(tgt_image)(path  = filepath, telinfo = tgt_image.telinfo)
        calib_image.logger.info(f"DARK correction applied with {dark_image.path}")
        calib_image.status = tgt_image.status
        calib_image.update_status(process_name = 'darkcor')
        
        # Log information
        tgt_image.logger.info(f"DARK correction applied: FILEPATH = {calib_image.path}")
        dark_image.logger.info(f"Used for DARK correction: FILEPATH = {calib_image.path}")
        return calib_image

    def _correct_dark(self, tgt_data : CCDData, dark_data : CCDData):
        calib_data = ccdproc.subtract_dark(tgt_data, dark_data, scale = True, exposure_time = 'EXPTIME', exposure_unit = u.second)
        return calib_data
    
    def correct_flat(self, tgt_image: ScienceImage, flat_image: CalibrationImage,
                     output_dir : str = None
                     ):
        if tgt_image.status.status['flatcor']['status']:
            tgt_image.logger.warning(f"FLAT correction already applied to {tgt_image.path}. FLAT correction is not applied.")
            raise RuntimeError(f"FLAT correction already applied to {tgt_image.path}")
        
        # Convert input images to CCDData
        sci_ccddata = ccdproc.CCDData(data = tgt_image.data, meta = tgt_image.header, unit = 'adu')
        flat_ccddata = ccdproc.CCDData(data = flat_image.data, meta = flat_image.header, unit = 'adu')
        
        # Perform flat correction
        calib_data = self._correct_flat(tgt_data = sci_ccddata, flat_data = flat_ccddata)
        
        # Determine data types and convert to selected data type
        tgt_dtype = tgt_image.data.dtype
        flat_dtype = flat_image.data.dtype
        selected_dtype = np.promote_types(tgt_dtype, flat_dtype)
        calib_data.data = calib_data.data.astype(selected_dtype)
        
        # Add metadata
        calib_data.meta['FLATCOR'] = True
        calib_data.meta['FCORTIME'] = Time.now().isot
        calib_data.meta['FLATPATH'] = flat_image.path
        
        # Determine output filename
        if not output_dir:
            output_dir = tgt_image.savedir
        os.makedirs(output_dir, exist_ok = True)
        
        filename = os.path.basename(tgt_image.path)
        if 'cor_' in filename:
            filepath = os.path.join(output_dir, 'f' + filename)
        else:
            filepath = os.path.join(output_dir, 'fcor_' + filename)
        calib_data.write(filepath, overwrite = True)
        
        # Create new image object
        calib_image = type(tgt_image)(path  = filepath, telinfo = tgt_image.telinfo)
        calib_image.logger.info(f"FLAT correction applied with {flat_image.path}")
        calib_image.status = tgt_image.status
        calib_image.update_status(process_name = 'flatcor')
        
        # Log information
        tgt_image.logger.info(f"FLAT correction applied: FILEPATH = {calib_image.path}")
        flat_image.logger.info(f"Used for FLAT correction: FILEPATH = {calib_image.path}")
        return calib_image
        
    def _correct_flat(self, tgt_data : CCDData, flat_data : CCDData):
        calib_data = ccdproc.flat_correct(tgt_data, flat_data)
        return calib_data
    
    def generate_masterframe(self, calib_imagelist : List[CalibrationImage], 
                             mbias : Union[CalibrationImage or List[CalibrationImage],None],
                             mdark : Union[CalibrationImage or List[CalibrationImage],None]):
        """ Generate master bias, dark, flat frames """
        all_filelist = [image.path for image in calib_imagelist]
        all_fileinfo = self.get_imginfo(all_filelist, normalize_key = True)
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
                    date_str = Time(np.mean(Time(bias_fileinfo['obsdate']).jd), format = 'jd').datetime.strftime('%Y%m%d_%H%M%S')
                    output_name = f'{date_str}-bias.fits'
                    combined_path = os.path.join(bias_rep['image'].config['CALIBDATA_MASTERDIR'], bias_rep['image'].observatory, bias_rep['image'].telkey, 'BIAS', bias_rep['image'].telname, output_name)
                    combined_path = self.combine_img(filelist = bias_fileinfo['file'], 
                                                     output_path = combined_path,
                                                     combine_method = 'median', 
                                                     scale = None, 
                                                     print_output = True,
                                                     clip = 'extrema',
                                                     clip_extrema_nlow=1,
                                                     clip_extrema_nhigh=1)
                    new_bias = CalibrationImage(path = combined_path, telinfo = bias_rep['image'].telinfo, status = bias_rep['image'].status)
                    master_files[key]['BIAS'] = new_bias
                else:
                    new_bias = self.get_masterframe(tgt_image = group[0]['image'],
                                                    imagetyp = 'BIAS')
                    master_files[key]['BIAS'] = new_bias
                                
            if not master_files[key]['DARK']:
                dark_key = ['DARK']
                dark_mask  = np.isin(group['imgtype'], dark_key)
                dark_fileinfo = group[dark_mask]
                if dark_fileinfo:
                    dark_fileinfo_by_exptime = dark_fileinfo.group_by('exptime').groups
                    for dark_group in dark_fileinfo_by_exptime:
                        dark_rep = dark_group[0]
                        exptime_name = dark_rep['exptime']
                        date_str = Time(np.mean(Time(dark_fileinfo['obsdate']).jd), format = 'jd').datetime.strftime('%Y%m%d_%H%M%S')
                        output_name = '%s-dark_%.1fs.fits' % (date_str, float(exptime_name))
                        combined_path = os.path.join(dark_rep['image'].config['CALIBDATA_MASTERDIR'], dark_rep['image'].observatory, dark_rep['image'].telkey, 'DARK', dark_rep['image'].telname, output_name)
                        b_darkimagelist = []
                        for dark in tqdm(dark_group['image'], desc = 'BIAS correction on DARK frames...'):
                            b_dark_image = self.correct_bias(tgt_image = dark, bias_image = master_files[key]['BIAS'])
                            b_darkimagelist.append(b_dark_image)
                        b_darkfilelist = [image.path for image in b_darkimagelist]
                            
                        combined_path = self.combine_img(filelist = b_darkfilelist, 
                                                         output_path = combined_path,
                                                         combine_method = 'median', 
                                                         scale = None, 
                                                         print_output = True,
                                                         clip = 'extrema',
                                                         clip_extrema_nlow=1,
                                                         clip_extrema_nhigh=1)
                        new_dark = CalibrationImage(path = combined_path, telinfo = dark_rep['image'].telinfo, status = dark_rep['image'].status)
                        master_files[key]['DARK'][exptime_name] = new_dark

                if '100.0' not in master_files[key]['DARK'].keys():
                    new_dark = self.get_masterframe(tgt_image = group[0]['image'],
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
                        date_str = Time(np.mean(Time(flat_fileinfo['obsdate']).jd), format = 'jd').datetime.strftime('%Y%m%d_%H%M%S')
                        output_name = f'{date_str}-flat-{filter_name}.fits'
                        combined_path = os.path.join(flat_rep['image'].config['CALIBDATA_MASTERDIR'], flat_rep['image'].observatory, flat_rep['image'].telkey, 'FLAT', flat_rep['image'].telname, output_name)
                        db_flatimagelist = []
                        for flat in tqdm(flat_group['image'], desc = 'BIAS, DARK correction on FLAT frames...'):
                            db_flat_image = self.correct_db(tgt_image = flat, bias_image = master_files[key]['BIAS'], dark_image = master_files[key]['DARK']['100.0'])
                            db_flatimagelist.append(db_flat_image)
                        db_flatfilelist = [image.path for image in db_flatimagelist]
                        
                        combined_path = self.combine_img(filelist = db_flatfilelist, 
                                                         output_path = combined_path,
                                                         combine_method = 'median', 
                                                         scale = None, 
                                                         print_output = True,
                                                         clip = 'extrema',
                                                         clip_extrema_nlow=1,
                                                         clip_extrema_nhigh=1)
                        new_flat = CalibrationImage(path = combined_path, telinfo = flat_rep['image'].telinfo, status = flat_rep['image'].status)
                        master_files[key]['FLAT'][filter_name] = new_flat
            
        return master_files
#%%
#%%
if __name__ == '__main__':
    import glob
    from tippy.routine import SDTData
    S = SDTData()
    obsdata = S.show_obssourcedata(foldername = '2025-02-02_gain2750')

    filelist = obsdata['7DT02']
    self = TIPPreprocess()
    fileinfo = self.get_imginfo(filelist)
    calib_info = fileinfo[fileinfo['imgtype'] != 'LIGHT']
    telinfo = self.get_telinfo('7DT', 'C361K', 'HIGH', 1)
    
    run_calib = True
    if run_calib:
        if calib_info:
            calib_images = [CalibrationImage(file, telinfo = telinfo) for file in calib_info['file']]
            self.generate_masterframe(calib_images, mbias = None, mdark = None)

    
    light_info = fileinfo[fileinfo['imgtype'] == 'LIGHT']
    light_groups = light_info.group_by(['binning', 'gain', 'filter']).groups
    
    for light_group in tqdm(light_groups):
        light_images = [ScienceImage(file, telinfo = telinfo) for file in light_group['file']]
        mbias = self.get_masterframe(light_images[0], imagetyp = 'BIAS')
        mdark = self.get_masterframe(light_images[0], imagetyp = 'DARK')        
        mflat = self.get_masterframe(light_images[0], imagetyp = 'FLAT')
        for tgt_image in light_images:    
            calib_image = self.correct_bdf(tgt_image = tgt_image, bias_image = mbias, dark_image = mdark, flat_image = mflat)