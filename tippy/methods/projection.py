
#%%
import os
from typing import Union, Optional, Tuple, List
import numpy as np
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.wcs import WCS
from astropy.io.fits import Header
from tippy.methods import TIPPlateSolve
from tippy.imageobjects import Mask
from tippy.imageobjects import ScienceImage, ReferenceImage, Errormap  # Adjust import path if needed
from tippy.helper import Helper  # Adjust import path if needed
#%%
class TIPProjection(Helper):
    
    def __init__(self):
        self.platesolve = TIPPlateSolve()
        super().__init__()
        
    def flip_image(self, 
                   data: np.ndarray, 
                   header: Header, 
                   flip: str = None) -> Tuple[np.ndarray, Header]:
        """
        Flip image data and WCS in a consistent way.
        
        Parameters
        ----------
        data : np.ndarray
            2D image array
        wcs : astropy.wcs.WCS
            WCS object to be updated
        flip : str
            'fliplr' for left-right, 'flipud' for up-down
        
        Returns
        -------
        data_flipped : np.ndarray
            Flipped image data
        wcs_flipped : WCS
            Flipped WCS
        """
        wcs = WCS(header)
        ny, nx = data.shape
        if flip == 'fliplr':
            data_flipped = np.fliplr(data)
            header['CRPIX1'] = nx + 1 - wcs.wcs.crpix[0]
            
            # Flip relevant coefficients
            for key in ['CD1_1', 'CD2_1']:
                if key in header.keys():
                    header[key] *= -1
            for key in ['A_0_2', 'A_2_0', 'B_1_1']:
                if key in header.keys():
                    header[key] *= -1
            for key in ['AP_0_0', 'AP_0_1', 'AP_0_2', 'AP_2_0']:
                if key in header.keys():
                    header[key] *= -1
            for key in ['BP_1_0', 'BP_1_1']:
                if key in header.keys():
                    header[key] *= -1
                
        elif flip == 'flipud':
            data_flipped = np.flipud(data)
            header['CRPIX2'] = ny + 1 - wcs.wcs.crpix[1]
            
            # Flip relevant coefficients
            for key in ['CD1_2', 'CD2_2']:
                if key in header.keys():
                    header[key] *= -1
            for key in ['A_1_1', 'B_0_2', 'B_2_0']:
                if key in header.keys():
                    header[key] *= -1
            for key in ['AP_0_1', 'AP_1_1']:
                if key in header.keys():
                    header[key] *= -1
            for key in ['BP_0_0', 'BP_0_2', 'BP_1_0', 'BP_2_0']:
                if key in header.keys():
                    header[key] *= -1

        elif flip is None:
            data_flipped = data
            header = header
        else:
            raise ValueError("flip must be 'fliplr' or 'flipud' or None")
        # Update the WCS header
        return data_flipped, header
    
    def align(self,
              target_img: Union[ScienceImage, ReferenceImage],
              reference_img: Union[ScienceImage, ReferenceImage],
              detection_sigma: float = 5.0,
              verbose: bool = True,
              overwrite: bool = True,
              save: bool = True,
              
              # platesolve parameters
              platesolve: bool = False,
              **kwargs
              ):
        # Path                
        target_data, target_header = target_img.data, target_img.header
        reference_data, reference_header = reference_img.data, reference_img.header

        # If sign on determinant of pixel scale matrix is different, flip the target image horizontally.
        if target_img.wcs is not None and reference_img.wcs is not None:
            target_det = np.linalg.det(target_img.wcs.pixel_scale_matrix)
            reference_det = np.linalg.det(reference_img.wcs.pixel_scale_matrix)
            if np.sign(target_det) != np.sign(reference_det):
                if verbose:
                    print("[WARNING] Target and reference images have opposite handedness (flip/mirror). Flipping the target image.")
                target_data, target_header = self.flip_image(target_data, target_header, flip='fliplr')
                
        flip_modes = dict(original = None, 
                          flip_horizon = 'fliplr',
                          flip_vertical = 'flipud')
        
        success = False
        for label, flip_mode in flip_modes.items():
            try:
                if verbose:
                    print(f"[INFO] Trying astroalign with {label} image...")

                flipped_data, flipped_header = self.flip_image(
                    data = target_data, 
                    header = target_header, 
                    flip = flip_mode)

                aligned_data, aligned_header, footprint = self.img_astroalign(
                    target_img=flipped_data,
                    reference_img=reference_data,
                    target_header=flipped_header,
                    reference_header=reference_header,
                    target_outpath=None,
                    detection_sigma=detection_sigma,
                    verbose=verbose
                )
                success = True
                if verbose:
                    print(f"[SUCCESS] Alignment succeeded with {label} image.")
                break
            except Exception as e:
                if verbose:
                    print(f"[FAILURE] Astroalign failed with {label} image: {e}")
            
        if not overwrite:      
            aligned_path = target_img.savepath.alignpath
            target_img = type(target_img)(path = aligned_path, telinfo = target_img.telinfo, status = target_img.status, load = False)
        
        target_img.data = aligned_data
        target_img.header = aligned_header
        update_header_kwargs = dict(
            ALIGNREF = str(reference_img.path),
            ALIGNSIG = detection_sigma,
        )
            
        target_img.header.update(update_header_kwargs)
            
        target_img.update_status(process_name = 'ASTROALIGN')
        
        if platesolve:
            target_img = self.platesolve.solve_scamp(
                target_img = target_img,
                scamp_sexparams = None,
                scamp_params = None,
                # Other parameters
                overwrite = True,
                verbose = verbose)[0]

        if save:
            target_img.write()

        return target_img

    def reproject(self,
                  target_img: Union[ScienceImage, ReferenceImage],
                  target_errormap: Optional[Errormap] = None,
                  swarp_params: Optional[dict] = None,
                  
                  resample_type: str = 'LANCZOS3',
                  center_ra: Optional[float] = None,
                  center_dec: Optional[float] = None,
                  x_size: Optional[int] = None,
                  y_size: Optional[int] = None,
                  pixelscale: Optional[float] = None,
                  verbose: bool = True,
                  overwrite: bool = False,
                  save: bool = True,
                  return_ivpmask: bool = False,
                  fill_zero_tonan: bool = True,
                  **kwargs
                  ):
        # If target_img is not saved, save it to the savepath
        target_img = target_img.copy()
        if target_img.is_exists is False:
            target_img.write()
        # If target_errormap is not saved, save it to the savepath

        if target_errormap is not None:
            if target_errormap.emaptype == 'bkgrms':
                target_errormap.to_weight()
        if target_errormap is not None and target_errormap.is_exists is False:
            target_errormap.write()

        original_header = target_img.header
        target_path = target_img.path
        # If overwrite, set the output path to the savepath
        if overwrite:
            target_outpath = target_img.savepath.savepath
            errormap_outpath = target_errormap.savepath.savepath if target_errormap is not None else None
        else:
            target_outpath = target_img.savepath.coaddpath
            errormap_outpath = target_errormap.savepath.coaddpath if target_errormap is not None else None
        # Temporary output paths
        target_outpath_tmp = str(target_outpath) + '.tmp'
        errormap_outpath_tmp = str(errormap_outpath) + '.tmp' if target_errormap is not None else None

        swarp_configfile = target_img.config['SWARP_CONFIG']
        
        target_outpath, errormap_outpath_tmp = self.run_swarp(
            target_path = target_path,
            swarp_configfile = swarp_configfile,
            swarp_params = swarp_params,
            target_outpath = target_outpath,
            weight_inpath = target_errormap.path if target_errormap else None,            
            weight_outpath = errormap_outpath_tmp,
            weight_type = 'MAP_WEIGHT' if target_errormap else None,
            resample = True,
            resample_type = resample_type,
            center_ra = center_ra,
            center_dec = center_dec,
            x_size = x_size,
            y_size = y_size,
            pixelscale = pixelscale,
            combine = True,
            subbkg = False,
            verbose = verbose,
            fill_zero_tonan = fill_zero_tonan,
            ) 
        
        os.remove(errormap_outpath_tmp) if errormap_outpath_tmp is not None else None
        
        if target_errormap is not None:
            target_outpath_tmp, errormap_outpath = self.run_swarp(
                target_path = target_path,
                swarp_configfile = swarp_configfile,
                swarp_params = swarp_params,
                target_outpath = target_outpath_tmp,
                weight_inpath = target_errormap.path if target_errormap else None,            
                weight_outpath = errormap_outpath,
                weight_type = 'MAP_WEIGHT' if target_errormap else None,
                resample = True,
                resample_type = 'NEAREST',
                center_ra = center_ra,
                center_dec = center_dec,
                x_size = x_size,
                y_size = y_size,
                pixelscale = pixelscale,
                combine = True,
                subbkg = False,
                verbose = verbose,
                ) 
            os.remove(target_outpath_tmp)
        
        reprojected_img = type(target_img)(path = target_outpath, telinfo = target_img.telinfo, status = target_img.status, load = False)
        reprojected_img.savedir = target_img.savedir
        reprojected_img.header = self.merge_header(reprojected_img.header, original_header, exclude_keys = ['PV*', '*SEC'])
        # reprojected_img.header = merged_header
        # original_header.update(reprojected_img.header)  # Update with new header
        # non_copy_header_keywords = ['DATASEC', 'BIASSEC', 'TRIMSEC', 'CCDSEC']
        # for key in non_copy_header_keywords:
        #     if key in original_header:
        #         original_header.remove(key)
        # reprojected_img.header = original_header
        reprojected_img.update_status(process_name = 'REPROJECT')

        if target_errormap is not None:
            target_errormap = Errormap(path = errormap_outpath, emaptype = 'bkgweight', status = target_errormap.status, load = True)
            target_errormap.header = original_header
            target_errormap.data
            target_errormap.remove(
                remove_main = True, 
                remove_connected_files = True,
                skip_exts = [],
                verbose = False)
            target_errormap.to_rms()

        else:
            target_errormap = None       
        
        if not save:
            reprojected_img.data
            reprojected_img.remove()
        else:
            reprojected_img.write()
            if target_errormap is not None:
                target_errormap.write()
        
        target_ivpmask = None
        if return_ivpmask:
            from tippy.methods import TIPMasking
            T = TIPMasking()
            reprojected_img.data
            target_ivpmask = T.mask_invalidpixel(
                target_img = reprojected_img,
                target_mask = None,
                save = save,
                verbose = verbose,
                visualize = False,
                save_fig = False
            )
            if save:
                target_ivpmask.write()
            
        return reprojected_img, target_errormap, target_ivpmask
        