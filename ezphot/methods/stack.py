#%%
from typing import List,Union,Optional,Tuple
import numpy as np
from multiprocessing import Pool, cpu_count
import bottleneck as bn
from tqdm import tqdm
import re
from astropy.io import fits
from astropy.time import Time
import os
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.contrib.concurrent import process_map

from ezphot.helper import Helper
from ezphot.imageobjects import ScienceImage, CalibrationImage, MasterImage, ReferenceImage, Errormap, Background
from ezphot.methods import Reproject
from ezphot.methods import BackgroundGenerator
from ezphot.methods import PSFPhotometry


def combine_patch(patch_tuple, combine_method='mean', clip_method='sigma', sigma=3.0, nlow=1, nhigh=1):
    (i_start, i_end, j_start, j_end, tile_stack, bkgrms_stack) = patch_tuple

    if combine_method.lower() in ['weight', 'weighted'] and bkgrms_stack is None:
        raise ValueError("combine_method='weight' requires bkgrms_stack to be provided.")

    # --- Clipping ---
    if clip_method is None or clip_method.lower() == 'none':
        clipped = tile_stack
        clipped_rms = bkgrms_stack
    elif clip_method.lower() == 'sigma':
        mean = np.nanmean(tile_stack, axis=0)
        std = np.nanstd(tile_stack, axis=0)
        mask = np.abs(tile_stack - mean) < sigma * std
        clipped = np.where(mask, tile_stack, np.nan)
        clipped_rms = np.where(mask, bkgrms_stack, np.nan)
    elif clip_method.lower() == 'extrema':
        sorted_idx = np.argsort(tile_stack, axis=0)
        valid_idx = sorted_idx[nlow:len(tile_stack) - nhigh]

        # Apply indices to both image and RMS
        clipped = np.take_along_axis(tile_stack, valid_idx, axis=0)
        if bkgrms_stack is not None:
            clipped_rms = np.take_along_axis(bkgrms_stack, valid_idx, axis=0)
        else:
            clipped_rms = None
    else:
        raise ValueError(f"Unknown clip_method: {clip_method}")

    # -- Combine image ---
    if combine_method.lower() in ['weight', 'weighted']:
        if clipped_rms is None:
            raise ValueError("Weighted combination requires background RMS (clipped_rms).")

        # Compute weights as 1 / (rms^2), safely avoiding divide-by-zero
        weights = 1.0 / np.where(clipped_rms > 0, clipped_rms**2, np.nan)
        
        # Weighted average
        weighted_sum = np.nansum(clipped * weights, axis=0)
        weight_total = np.nansum(weights, axis=0)
        combined = np.divide(weighted_sum, weight_total, out=np.zeros_like(weight_total), where=weight_total > 0)

    elif combine_method.lower() == 'mean':
        combined = bn.nanmean(clipped, axis=0)

    elif combine_method.lower() == 'median':
        combined = bn.median(clipped, axis=0)

    elif combine_method.lower() == 'sum':
        combined = np.nansum(clipped, axis=0)

    else:
        raise ValueError(f"Unknown combine_method: {combine_method}")
    
    # --- Combine RMS ---
    if clipped_rms is not None:
        N = clipped_rms.shape[0]

        if combine_method.lower() == 'mean':
            # Combine assuming uncorrelated noise: ?_combined = sqrt(sum ?_i^2) / N
            combined_rms = np.sqrt(np.nansum(clipped_rms**2, axis=0)) / N

        elif combine_method.lower() in ['weight', 'weighted']:
            # Already computed weights in image combination block
            # combined_rms = sqrt(1 / sum w_i)
            combined_rms = np.sqrt(1.0 / np.where(weight_total > 0, weight_total, np.nan))

        elif combine_method.lower() == 'median':
            # Approximation for standard error of median: ?_combined ? 1.253 / sqrt(N) * median(?_i)
            combined_rms = 1 / np.sqrt(N) * np.nanmedian(clipped_rms, axis=0)

        else:
            combined_rms = None
    else:
        combined_rms = None

    return i_start, i_end, j_start, j_end, combined, combined_rms

# Backgroud subtraction worker function
bkg_handler = BackgroundGenerator()
def _subtract_background_worker(args):
    target_img, target_bkg = args
    result = bkg_handler.subtract_background(
        target_img=target_img,
        target_bkg=target_bkg,
        save=True, 
        overwrite=False,
        visualize=False
    )
    return result

def _scale_worker(args) -> Tuple:
    target_img, target_errormap, ref_zp, zp_key, save, overwrite = args
    import numpy as np

    zp = float(target_img.header[zp_key])
    delta_zp = ref_zp - zp
    scale_factor = 10 ** (0.4 * (delta_zp))

    if not overwrite:
        target_img_path = target_img.savepath.savedir / f"scaled_{target_img.savepath.savepath.name}"
        target_errormap_path = target_errormap.savepath.savedir / f"scaled_{target_errormap.savepath.savepath.name}" if target_errormap else None
    else:
        target_img_path = target_img.savepath.savepath
        target_errormap_path = target_errormap.savepath.savepath if target_errormap else None

    # Scale image
    scaled_img = type(target_img)(path=target_img_path, telinfo=target_img.telinfo, status=target_img.status, load=False)
    scaled_img.data = target_img.data * scale_factor
    scaled_img.header = target_img.header.copy()
    scaled_img.header[zp_key] = ref_zp
    scaled_img.header.update({
        'SCLEKEY': zp_key,
        'SCLEREF': ref_zp,
        'SCLEZP': delta_zp,
        'SCLEFACT': scale_factor,
    })
    for key in target_img.header.keys():
        if key.startswith('ZP_'):
            scaled_img.header[key] = target_img.header[key] + delta_zp
    scaled_img.update_status('ZPSCALE')

    # Scale error map
    scaled_errormap = None
    if target_errormap:
        emaptype = target_errormap.emaptype.lower()
        if emaptype == 'bkgrms':
            factor = scale_factor
        elif emaptype == 'bkgweight':
            factor = 1.0 / scale_factor**2
        else:
            raise ValueError(f"Unsupported emaptype '{emaptype}'")

        scaled_errormap = Errormap(path=target_errormap_path, emaptype=target_errormap.emaptype, status=target_errormap.status, load=False)
        scaled_errormap.data = target_errormap.data * factor
        scaled_errormap.header.update({
            'SCLEKEY': zp_key,
            'SCLEREF': ref_zp,
            'SCLEZP': delta_zp,
            'SCLEFACT': scale_factor,
        })
        scaled_errormap.add_status('zpscale', key=zp_key, ref_zp=ref_zp, scale_zp=delta_zp, scale_factor=scale_factor)

    if save:
        scaled_img.write()
        if scaled_errormap:
            scaled_errormap.write()

    return scaled_img, scaled_errormap

projection_handler = Reproject()
def _reproject_worker(args):
    target_img, target_bkgrms, resample_type, center_ra, center_dec, x_size, y_size, pixel_scale = args
    reprojected_img, reprojected_bkgrms, _ = projection_handler.reproject(
        target_img=target_img,
        target_errormap=target_bkgrms,
        swarp_params=None,
        resample_type=resample_type,
        center_ra=center_ra,
        center_dec=center_dec,
        x_size=x_size,
        y_size=y_size,
        pixelscale=pixel_scale,
        verbose=False,
        overwrite=False,
        save=False,
        return_ivpmask=False,
    )
    return reprojected_img, reprojected_bkgrms

# Wrapper to allow map-style multiprocessing with multiple args
def worker_wrapper(args):
    return combine_patch(*args)


class Combiner:
    
    def __init__(self,
                 n_proc: int = 8):
        self.n_proc = cpu_count() if n_proc is None else n_proc
        self.pool = Pool(processes=self.n_proc)
    
    def close_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
            self.n_proc = 0
    
    @staticmethod
    def split_image_stack_by_nproc(stack, n_proc):
        N, H, W = stack.shape
        nx = int(np.sqrt(n_proc))
        ny = (n_proc + nx - 1) // nx
        x_splits = np.linspace(0, H, ny + 1, dtype=int)
        y_splits = np.linspace(0, W, nx + 1, dtype=int)

        patches = []
        for i in range(len(x_splits) - 1):
            for j in range(len(y_splits) - 1):
                i_start, i_end = x_splits[i], x_splits[i + 1]
                j_start, j_end = y_splits[j], y_splits[j + 1]
                tile_stack = stack[:, i_start:i_end, j_start:j_end]
                patches.append((i_start, i_end, j_start, j_end, tile_stack))

        return patches

    def combine_images_parallel(self, 
                                image_list, 
                                bkgrms_list=None,
                                combine_method='mean',
                                clip_method='sigma',
                                sigma=3.0, 
                                nlow=1,
                                nhigh=1,
                                verbose=True,
                                **kwargs):
        if verbose:
            print(f"[Combiner] Combining {len(image_list)} images with combine='{combine_method}', clip='{clip_method}', using {self.n_proc} processes")

        stack = np.stack(image_list)
        bkgrms_stack = np.stack(bkgrms_list) if bkgrms_list is not None else None
        H, W = stack.shape[1:]

        combined = np.zeros((H, W), dtype=np.float32)
        bkgrms_out = np.zeros((H, W), dtype=np.float32) if bkgrms_stack is not None else None

        image_patches = self.split_image_stack_by_nproc(stack, self.n_proc)
        bkgrms_patches = self.split_image_stack_by_nproc(bkgrms_stack, self.n_proc) if bkgrms_stack is not None else [None] * len(image_patches)

        patch_args = []
        for (img_patch, bkgrms_patch) in zip(image_patches, bkgrms_patches):
            i, j, k, l, tile = img_patch
            bkgrms = bkgrms_patch[4] if bkgrms_patch is not None else None
            patch_args.append(((i, j, k, l, tile, bkgrms), combine_method, clip_method, sigma, nlow, nhigh))

        # Use persistent pool if available
        #if self.pool is not None:
        pool = self.pool
        results = list(
            tqdm(pool.imap_unordered(worker_wrapper, patch_args),
                    total=len(patch_args),
                    desc="Combining...",
                    ncols=80,
                    bar_format="{l_bar}{bar}| {elapsed}") if verbose else pool.imap_unordered(worker_wrapper, patch_args)
        )
        # else:
        #     # Fallback to temporary pool
        #     with Pool(processes=self.n_proc) as pool:
        #         results = list(
        #             tqdm(pool.imap_unordered(worker_wrapper, patch_args),
        #                  total=len(patch_args),
        #                  desc="Combining...",
        #                  ncols=80,
        #                  bar_format="{l_bar}{bar}| {elapsed}") if verbose else pool.imap_unordered(worker_wrapper, patch_args)
        #         )

        for i_start, i_end, j_start, j_end, patch_result, patch_bkgrms in results:
            combined[i_start:i_end, j_start:j_end] = patch_result
            if bkgrms_out is not None and patch_bkgrms is not None:
                bkgrms_out[i_start:i_end, j_start:j_end] = patch_bkgrms

        return combined, bkgrms_out


class Stack:
    
    def __init__(self):        
        self.helper = Helper()
        self.combiner = Combiner()
        self.background = BackgroundGenerator()
        self.psfphot    = PSFPhotometry()
    
    def stack_multiprocess(self,
                           target_imglist: Union[List[ScienceImage], List[CalibrationImage]],
                           target_bkglist: Optional[List[Background]] = None,
                           target_bkgrmslist: Optional[List[Errormap]] = None,
                           target_outpath: str = None,
                           bkgrms_outpath: str = None,
                           combine_type: str = 'median',
                           n_proc=4,
                           
                           # Clip parameters
                           clip_type: str = None,
                           sigma: float = 3.0,
                           nlow: int = 1,
                           nhigh: int = 1,
                           
                           # Resample parameters
                           resample: bool = True,
                           resample_type: str = 'LANCZOS3',
                           center_ra: float = None,
                           center_dec: float = None,
                           pixel_scale: float = None,
                           x_size: int = None,
                           y_size: int = None,
                           
                           # Scale parameters
                           scale: bool = True,
                           scale_type: str = 'min',
                           zp_key : str = 'ZP_APER_1',
                            
                           # Convolution parameters
                           convolve: bool = False,
                           seeing_key: str = 'SEEING',
                           kernel: str = 'gaussian',
                           
                           # Other parameters
                           verbose: bool = True,
                           save: bool = True,
                           **kwargs):
        """
        Stack a list of images.
        
        Parameters
        ----------
        target_imglist : List[ScienceImage] or List[CalibrationImage]
            The list of images to stack.
        target_bkglist : List[Background], optional
            The list of backgrounds to subtract from the images.
        target_bkgrmslist : List[Errormap], optional
            The list of background RMS maps to use for the stacking.
        target_outpath : str, optional
            The path to save the stacked image.
        bkgrms_outpath : str, optional
            The path to save the background RMS map.
        combine_type : str, optional
            The type of combination to use for the stacking.
        n_proc : int, optional
            The number of processes to use for the stacking.
        clip_type : str, optional
            The type of clipping to use for the stacking. [sigma, extrema]
        sigma : float, optional
            The sigma for the clipping.
        nlow : int, optional
            The number of low values to clip.
        nhigh : int, optional
            The number of high values to clip.
        resample : bool, optional
            Whether to resample the images.
        resample_type : str, optional
            The type of resampling to use for the stacking in SWArp configuration. ['NEAREST', 'BILINEAR', 'BICUBIC', 'LANCZOS3', etc.]
        center_ra : float, optional
            The RA of the center of the stacked image.
        center_dec : float, optional
            The Dec of the center of the stacked image.
        pixel_scale : float, optional
            The pixel scale of the stacked image.
        x_size : int, optional
            The size of the stacked image in the x-direction.
        y_size : int, optional
            The size of the stacked image in the y-direction.
        scale : bool, optional
            Whether to scale the images.
        scale_type : str, optional
            The type of scaling to use for the stacking. ['min', 'mean', 'median', 'max']
        zp_key : str, optional
            The key to use for the zero point.
        convolve : bool, optional
            Whether to convolve the images.
        seeing_key : str, optional
            The key to use for the seeing.
        kernel : str, optional
            The kernel to use for the convolution. ['gaussian', 'image']
        verbose : bool, optional
            Whether to print verbose output.
        save : bool, optional
            Whether to save the stacked image.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        target_img : ScienceImage
            The stacked image.
        target_bkgrms : Errormap    
            The stacked background RMS map.
        """
        
        if self.combiner.n_proc != n_proc:
            self.helper.print('[Combiner] Re-initializing Combiner with new n_proc', verbose)
            self.combiner.close_pool()
            self.combiner = Combiner(n_proc=n_proc)
        
        # Define output paths if not provided
        if target_outpath is None:
            target_outpath = target_imglist[0].savepath.savepath.with_suffix('.com.fits')
        if (target_bkgrmslist is not None) & (bkgrms_outpath is None):
            suffix = '.com.fits' + target_bkgrmslist[0].savepath.savepath.suffix 
            bkgrms_outpath = target_imglist[0].savepath.savepath.with_suffix(suffix) 
        
        subbkg_imglist = []
        if target_bkglist is not None:
            if len(target_imglist) != len(target_bkglist):
                raise ValueError("Length of target_imglist and target_bkglist must be the same.")
        
            input_list = [(img, bkg) for img, bkg in zip(target_imglist, target_bkglist)]

            if verbose:
                target_imglist = process_map(_subtract_background_worker, input_list, max_workers=n_proc, desc="Subtracting background...")
            else:
                with Pool(processes=n_proc) as pool:
                    target_imglist = pool.map(_subtract_background_worker, input_list)

            # Clean up memory
            for target_img in target_imglist:
                target_img.data = None
            
            for target_bkg in target_bkglist:
                target_bkg.data = None
                
            subbkg_imglist = target_imglist.copy()
                         
        # Zero-point scaling
        scaled_imglist = []
        scaled_bkgrmslist = []
        if scale:
            target_imglist, target_bkgrmslist = self.match_zeropoints(
                target_imglist = target_imglist,
                target_errormaplist = target_bkgrmslist,
                method = scale_type,
                zp_key = zp_key,
                save = True, 
                overwrite = False,
                verbose = verbose,
                n_proc = n_proc
            )
            
            # Clean up memory
            for target_img in target_imglist:
                target_img.data = None
            if target_bkgrmslist is not None:
                for target_bkgrms in target_bkgrmslist:
                    target_bkgrms.data = None
            
            scaled_imglist = target_imglist.copy()
            scaled_bkgrmslist = target_bkgrmslist.copy() if target_bkgrmslist is not None else None
            
        # Convolution
        convolved_imglist = []
        convolved_bkgrmslist = []
        if convolve:
            target_imglist, target_bkgrmslist = self.match_seeing(
                target_imglist = target_imglist,
                target_errormaplist = target_bkgrmslist,
                seeing_key = seeing_key,
                kernel = kernel,
                save = False, 
                overwrite = False,
                verbose = verbose
            )
            
            for target_img in target_imglist:
                target_img.data = None
            if target_bkgrmslist is not None:
                for target_bkgrms in target_bkgrmslist:
                    target_bkgrms.data = None   
            
            convolved_imglist = target_imglist.copy()
            convolved_bkgrmslist = target_bkgrmslist.copy() if target_bkgrmslist is not None else None
            
        coadd_imglist = None
        coadd_bkgrmslist = None
        if resample:
            task_args = [
                (img, bkgrms, resample_type, center_ra, center_dec, x_size, y_size, pixel_scale)
                for img, bkgrms in zip(target_imglist, target_bkgrmslist or [None] * len(target_imglist))
            ]
            
            if verbose:
                results = process_map(_reproject_worker, task_args, max_workers=n_proc, desc="Performing image reprojection...")
            else:
                with Pool(processes=n_proc) as pool:
                    results = pool.map(_reproject_worker, task_args)
            
            coadd_imglist, coadd_bkgrmslist =  zip(*results)
            if target_bkgrmslist is None:
                coadd_bkgrmslist = None

        else:
            coadd_imglist = target_imglist
            coadd_bkgrmslist = target_bkgrmslist
              
        # Load target images and error maps ---
        image_datalist = []
        image_hdrlist = []
        
        iterator = tqdm(coadd_imglist, desc="Loading target images...", ncols=80, bar_format="{l_bar}{bar}| {elapsed}") if verbose else coadd_imglist
        for img in iterator:
            image_datalist.append(img.data)
            image_hdrlist.append(img.header)

        bkgrms_datalist = None
        if coadd_bkgrmslist is not None:
            bkgrms_datalist = []
            iterator = tqdm(coadd_bkgrmslist, desc="Loading target bkgrms maps...", ncols=80, bar_format="{l_bar}{bar}| {elapsed}") if verbose else coadd_bkgrmslist
            for target_bkgrms in iterator:
                bkgrms_datalist.append(target_bkgrms.data)
        
        # Remove original target images and error maps
        for target_imglist in [subbkg_imglist, scaled_imglist, convolved_imglist, coadd_imglist]:
            if target_imglist:
                for target_img in target_imglist:
                    target_img.remove()
                    
        for target_bkgrmslist in [scaled_bkgrmslist, convolved_bkgrmslist, coadd_bkgrmslist]:
            if target_bkgrmslist:
                for target_bkgrms in target_bkgrmslist:
                    target_bkgrms.remove()
                
        # Combine the image stack
        if clip_type is 'extrema':
            if len(image_datalist) - nlow - nhigh < 3:
                self.helper.print(f"[Combiner] Not enough images to clip: ({len(image_datalist)}). Clip type is set as None", verbose)
                clip_type = None
                
        combined_data, combined_bkgrms = self.combiner.combine_images_parallel(
            image_list=image_datalist,
            bkgrms_list = bkgrms_datalist,
            combine_method=combine_type,
            clip_method=clip_type,
            sigma=sigma,
            nlow=nlow,
            nhigh=nhigh,
            verbose=verbose
        )

        # Initialize combined header 
        combined_header = image_hdrlist[0].copy()

        # Update header keywords with mean
        update_header_keywords_mean = ['ALTITUDE', 'AZIMUTH', 'CENTALT', 'CENTAZ', 'RA', 'DEC', 'AIRMASS', 'SEEING', 'PEEING', 'ELLIP', 'SKYVAL', 'JD', 'MJD', 'MJD-OBS']
        for key in update_header_keywords_mean:
            values = [hdr.get(key) for hdr in image_hdrlist if hdr.get(key) not in [None, '']]
            try:
                if values:
                    combined_header[key] = float(np.nanmean(values))
            except Exception:
                pass  # Handle non-numeric or incompatible values
        for i, target_img in enumerate(target_imglist):
            combined_header[f'COMBIM{i+1}'] = target_img.path.name
        
        values = [Time(hdr.get('DATE-OBS')).jd for hdr in image_hdrlist if hdr.get('DATE-OBS') not in [None, '']]
        combined_header['DATE-OBS'] = Time(np.nanmean(values), format='jd').isot if values else None
        values = [Time(hdr.get('DATE-LOC')).jd for hdr in image_hdrlist if hdr.get('DATE-LOC') not in [None, '']]
        combined_header['DATE-LOC'] = Time(np.nanmean(values), format='jd').iso if values else None

        update_header_keywords_sum = ['EXPTIME', 'EXPOSURE']
        for key in update_header_keywords_sum:
            values = [hdr.get(key) for hdr in image_hdrlist if hdr.get(key) not in [None, '']]
            try:
                if values:
                    combined_header[key] = float(np.nansum(values))
            except Exception:
                pass
            
        # Remove unwanted header keywords
        update_header_keywords_remove = ['IMAGEID', 'NOTE', 'MAG_*', 'ZP*', 'UL*', 'EZP*', 'APER*', 'SKYSIG']
        for pattern in update_header_keywords_remove:
            if '*' in pattern:
                regex = re.compile('^' + pattern.replace('*', '.*') + '$')
                keys_to_remove = [k for k in combined_header if regex.match(k)]
            else:
                keys_to_remove = [k for k in combined_header if k == pattern]
            for k in keys_to_remove:
                del combined_header[k]

        # Save combined image
        # If CalibrationImage is input, Save it as MasterImage. This will be saved in the master_frame directory.
        # Else, save it in the target_outpath.
        stack_instance =  type(target_imglist[0])(path = target_outpath, telinfo = target_imglist[0].telinfo, status = target_imglist[0].status, load = False)
        stack_instance.data = combined_data
        stack_instance.header = combined_header
        stack_instance.update_status(process_name = 'STACK')
        
        stack_bkgrms_instance = None
        if target_bkgrmslist is not None:
            stack_bkgrms_instance = Errormap(path=bkgrms_outpath, emaptype = 'bkgrms', status = target_bkgrmslist[0].status, load=False)
            stack_bkgrms_instance.data = combined_bkgrms
            stack_bkgrms_instance.header = combined_header
        
        if save:
            stack_instance.write()
            stack_bkgrms_instance.write() if stack_bkgrms_instance is not None else None

        self.combiner.close_pool()
        return stack_instance, stack_bkgrms_instance
    
    def stack_swarp(self,
                    target_imglist : Union[List[ScienceImage], List[CalibrationImage]],
                    target_bkglist: Optional[List[Background]] = None,
                    target_errormaplist: Optional[List[Errormap]] = None,
                    target_outpath: str = None,
                    errormap_outpath: str = None,
                    combine_type: str = 'median', # median, weighted, mean, sum, min, max
                    
                    # Resample parameters
                    resample: bool = False,
                    resample_type: str = 'LANCZOS3',
                    center_ra: float = None,
                    center_dec: float = None,
                    pixel_scale: float = None,
                    x_size: int = None,
                    y_size: int = None,
                    
                    # Scale parameters
                    scale: bool = False,
                    scale_type: str = 'min',
                    zp_key : str = 'ZP_APER_1',
                    
                    # Convolution parameters
                    convolve: bool = False,
                    seeing_key: str = 'SEEING',
                    kernel: str = 'gaussian',
                    
                    # Other parameters
                    save: bool = True,
                    verbose: bool = True,
                    **kwargs
                    ):
        """
        Stack multiple images using SWArp.
        
        Parameters
        ----------
        target_imglist : List[Union[ScienceImage, CalibrationImage]]
            List of images to stack
        target_bkglist : Optional[List[Background]]
            Optional list of background maps to stack
        target_errormaplist : Optional[List[Errormap]]
            Optional list of error maps to stack
        target_outpath : str
            Path to save the stacked image
        errormap_outpath : str
            Path to save the stacked error map
        combine_type : str
            Method to combine images ('median', 'weighted', 'mean', 'sum', 'min', 'max')
        resample : bool
            Whether to resample the images
        resample_type : str
            Type of resampling to use ('NEAREST', 'BILINEAR', 'BICUBIC', 'LANCZOS3', etc.)
        center_ra : float
            RA of the center of the stacked image
        center_dec : float
            Dec of the center of the stacked image
        pixel_scale : float
            Pixel scale of the stacked image
        x_size : int
            Size of the stacked image in pixels
        y_size : int
            Size of the stacked image in pixels
        scale : bool
            Whether to scale the images
        scale_type : str
            Method to scale images ('min', 'mean', 'median', 'max')
        zp_key : str
            Header keyword for zero point
        convolve : bool
            Whether to convolve the images
        seeing_key : str
            Header keyword for seeing/FWHM in pixel units
        kernel : str
            Convolution kernel type ('gaussian')
        save : bool
            Whether to save the stacked image and error map
        verbose : bool
            Print progress messages
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        (stack_instance, stack_weight_instance) : Tuple[Union[ScienceImage, CalibrationImage], Optional[Errormap]]
            Stacked image and optionally its weight map
        """ 
        # Set default output paths if not provided
        if target_outpath is None:
            target_outpath = target_imglist[0].savepath.combinepath
        
        errormap_outpath = target_outpath + '.weight'
        
        # Set temporary output paths
        target_outpath_tmp = str(target_outpath) + '.tmp'

        # Subtract background 
        if target_bkglist is not None:
            if len(target_imglist) != len(target_bkglist):
                raise ValueError("Length of target_imglist and target_bkglist must be the same.")
            iterator = enumerate(tqdm(zip(target_imglist, target_bkglist), desc="Subtracting background...")) if verbose else enumerate(zip(target_imglist, target_bkglist))
            for i, (target_img, target_bkg) in iterator:
                target_imglist[i] = self.background.subtract_background(
                    target_img=target_img,
                    target_bkg=target_bkg,
                    save=False,
                    overwrite=False,
                    visualize=False)

        # Image scaling
        if scale:
            target_imglist, target_errormaplist = self.match_zeropoints(
                target_imglist = target_imglist,
                target_errormaplist = target_errormaplist,
                method = scale_type,
                zp_key = zp_key,
                overwrite = False,
                save = False,
                verbose = verbose
            )
            
        # Image convolution
        if convolve:
            target_imglist, target_errormaplist = self.match_seeing(
                target_imglist = target_imglist,
                target_errormaplist = target_errormaplist,
                seeing_key = seeing_key,
                kernel = kernel,
                save = False,
                overwrite = False,
                verbose = verbose
            )

        # Loading the images
        image_pathlist = []
        image_hdrlist = []
        remove_image = []
        iterator = tqdm(target_imglist, desc="Loading target images...", ncols=80, bar_format="{l_bar}{bar}| {elapsed}") if verbose else target_imglist
        for target_img in iterator:
            if not target_img.is_exists:
                target_img.write()
                remove_image.append(True)
            else:
                remove_image.append(False)
            image_pathlist.append(target_img.path)
            image_hdrlist.append(target_img.header)
            
        weight_pathlist = None
        remove_errormap = []
        if target_errormaplist is not None:
            weight_pathlist = []
            iterator = tqdm(target_errormaplist, desc="Loading target weight maps...", ncols=80, bar_format="{l_bar}{bar}| {elapsed}") if verbose else target_errormaplist
            for target_errormap in iterator:
                if target_errormap.emaptype.lower() != 'weight':
                    target_errormap.to_weight()
                    if not target_errormap.is_exists:
                        target_errormap.write()
                        remove_errormap.append(True)
                    else:
                        remove_errormap.append(False)
                else:
                    if not target_errormap.is_exists:
                        target_errormap.write()
                        remove_errormap.append(True)
                    else:
                        remove_errormap.append(False)
                    
                weight_pathlist.append(target_errormap.path)
                
        # Header modification
        combined_header = image_hdrlist[0].copy()

        # --- Update header keywords with mean ---
        update_header_keywords_mean = ['ALTITUDE', 'AZIMUTH', 'RA', 'DEC', 'AIRMASS', 'SEEING', 'PEEING', 'ELLIP', 'ELONG', 'SKYVAL', 'JD', 'MJD', 'MJD-OBS']
        for key in update_header_keywords_mean:
            values = [hdr.get(key) for hdr in image_hdrlist if hdr.get(key) not in [None, '']]
            try:
                if values:
                    combined_header[key] = float(np.nanmean(values))
            except Exception:
                pass  # Handle non-numeric or incompatible values
        for i, target_img in enumerate(target_imglist):
            combined_header[f'COMBIM{i+1}'] = target_img.path.name
            
        values = [Time(hdr.get('DATE-OBS')).jd for hdr in image_hdrlist if hdr.get('DATE-OBS') not in [None, '']]
        combined_header['DATE-OBS'] = Time(np.nanmean(values), format='jd').isot if values else None
        values = [Time(hdr.get('DATE-LOC')).jd for hdr in image_hdrlist if hdr.get('DATE-LOC') not in [None, '']]
        combined_header['DATE-LOC'] = Time(np.nanmean(values), format='jd').iso if values else None

        # --- Remove unwanted header keywords ---
        update_header_keywords_remove = ['IMAGEID', 'NOTE', 'MAG_*', 'ZP*', 'UL*', 'EZP*', 'APER*', 'SKYSIG']
        for pattern in update_header_keywords_remove:
            if '*' in pattern:
                regex = re.compile('^' + pattern.replace('*', '.*') + '$')
                keys_to_remove = [k for k in combined_header if regex.match(k)]
            else:
                keys_to_remove = [k for k in combined_header if k == pattern]
            for k in keys_to_remove:
                del combined_header[k]

        # Image combine
        self.helper.print(f"Start image combining...", verbose)
        imagestack_path = None
        weightstack_path = None
        # Run swarp 
        stack_pathlist = self.helper.run_swarp(
            target_path = image_pathlist,
            swarp_configfile = target_imglist[0].config['SWARP_CONFIG'],
            swarp_params = None,
            target_outpath = target_outpath,
            weight_inpath = weight_pathlist,
            weight_outpath = errormap_outpath,
            weight_type = 'MAP_WEIGHT',
            resample = resample,
            resample_type = resample_type,
            center_ra = center_ra,
            center_dec = center_dec,
            x_size = x_size,
            y_size = y_size,
            pixelscale = np.mean(target_imglist[0].pixelscale),
            combine = True,
            combine_type = combine_type,
            subbkg = False
        )
        imagestack_path, weightstack_path = stack_pathlist
            
        # If errormaplist is provided, run swarp for error maps (NEAREST resampling)
        if target_errormaplist is not None:
            self.helper.print(f"Start weight combining...", verbose)
            stack_pathlist = self.helper.run_swarp(
                target_path = image_pathlist,
                swarp_configfile = target_imglist[0].config['SWARP_CONFIG'],
                swarp_params = None,
                target_outpath = target_outpath_tmp,
                weight_inpath = weight_pathlist,
                weight_outpath = errormap_outpath,
                weight_type = 'MAP_WEIGHT',
                resample = resample,
                resample_type = 'NEAREST',
                center_ra = center_ra,
                center_dec = center_dec,
                x_size = x_size,
                y_size = y_size,
                pixelscale = np.mean(target_imglist[0].pixelscale),
                combine = True,
                combine_type = combine_type,
                subbkg = False
            )
            imagestack_tmppath, weightstack_path = stack_pathlist
            os.remove(imagestack_tmppath)
        
        if type(target_imglist[0]) == CalibrationImage:
            stack_instance = MasterImage(path = target_outpath, telinfo = target_imglist[0].telinfo, status = target_imglist[0].status, load = True)
            stack_instance.header = self.helper.merge_header(stack_instance.header, combined_header, exclude_keys = ['PV*'])
        else:
            stack_instance = type(target_imglist[0])(path = imagestack_path, telinfo = target_imglist[0].telinfo, status = target_imglist[0].status, load = True)
            stack_instance.header = self.helper.merge_header(stack_instance.header, combined_header, exclude_keys = ['PV*'])
            stack_instance.update_status(process_name = 'STACK')

        stack_weight_instance = None
        stack_weight_instance = Errormap(path = weightstack_path, emaptype = 'bkgweight', status = None, load = True)
        stack_weight_instance.header = self.helper.merge_header(stack_weight_instance.header, combined_header, exclude_keys = ['PV*'])
        event_details_kwargs = dict(
            stack_type = 'SWARP',
            combine_type = combine_type,
            resample = resample,
            resample_type = resample_type,
            ncombine = len(target_imglist)
        )
        stack_weight_instance.add_status('stack_swarp', **event_details_kwargs)
        
        if save:
            stack_instance.write()
            stack_weight_instance.write() if stack_weight_instance is not None else None
            self.helper.print(f"Stacked image saved to {stack_instance.path}", verbose)
            self.helper.print(f"Stacked weight map saved to {stack_weight_instance.path}", verbose)
        else:
            stack_instance.load()
            stack_weight_instance.load()
            stack_instance.remove()
            stack_weight_instance.remove() if stack_weight_instance is not None else None
        
        if any(remove_errormap):
            for remove_key, target_errormap in zip(remove_errormap, target_errormaplist):
                if remove_key:
                    target_errormap.remove()
        if any(remove_image):
            for remove_key, target_img in zip(remove_image, target_imglist):
                if remove_key:
                    target_img.remove()
                
        return stack_instance, stack_weight_instance
    
    def select_quality_images(self, 
                              target_imglist: Union[List[ScienceImage], List[ReferenceImage]],
                              min_obsdate: Union[Time, str, float] = None,
                              max_obsdate: Union[Time, str, float] = None,
                              seeing_key: str = 'SEEING',
                              depth_key: str = 'UL5_APER_1',
                              ellipticity_key: str = 'ELLIP',
                              obsdate_key: str = 'DATE-OBS',
                              weight_ellipticity: float = 3.0,
                              weight_seeing: float = 1.0,
                              weight_depth: float = 2.0,
                              max_numbers: int = None,
                              seeing_limit: float = 6.0,
                              depth_limit: float = 18.0,
                              ellipticity_limit: float = 0.3,
                              visualize: bool = False,
                              verbose: bool = True):
        """
        Select the best images based on seeing, depth, and ellipticity.
        
        Parameters
        ----------
        target_imglist : List[Union[ScienceImage, ReferenceImage]]
            List of images to select from.
        min_obsdate : Union[Time, str, float]
            Minimum observation date.
        max_obsdate : Union[Time, str, float]
            Maximum observation date.
        seeing_key : str
            Header keyword for seeing/FWHM in pixel units
        depth_key : str
            Header keyword for depth in AB magnitude
        ellipticity_key : str
            Header keyword for ellipticity
        obsdate_key : str
            Header keyword for observation date
        weight_ellipticity : float
            Weight for ellipticity
        weight_seeing : float
            Weight for seeing
        weight_depth : float
            Weight for depth
        max_numbers : int, optional
            Maximum number of images to select.
        seeing_limit : float
            Maximum seeing limit in arcseconds
        depth_limit : float
            Minimum depth limit in AB magnitude
        ellipticity_limit : float
            Maximum ellipticity limit
        visualize : bool
            Whether to visualize the selected images
        verbose : bool
            Whether to print verbose output.

        Returns
        -------
        (selected_imglist, selected_errormaplist) : Tuple[List[Union[ScienceImage, ReferenceImage]], Optional[List[Errormap]]]
            List of selected images and optionally their error maps
        """
        
        seeinglist = []
        depthlist = []
        ellipticitylist = []
        obsdatelist = []
        iterator = tqdm(target_imglist, desc="Querying images...", ncols=80, bar_format="{l_bar}{bar}| {elapsed}") if verbose else target_imglist
        for target_img in iterator:
            seeinglist.append(target_img.header.get(seeing_key, None))
            depthlist.append(target_img.header.get(depth_key, None))
            ellipticitylist.append(target_img.header.get(ellipticity_key, None))
            obsdatelist.append(target_img.header.get(obsdate_key, None))
        
        try:
            obsdate_time = Time(obsdatelist)
            min_obs_time = self.helper.flexible_time_parser(min_obsdate) if min_obsdate is not None else Time('1990-01-01')
            max_obs_time = self.helper.flexible_time_parser(max_obsdate) if max_obsdate is not None else Time.now()
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")          
        
        # Mask for images before max_obsdate
        valid_obs_mask = (obsdate_time < max_obs_time) & (obsdate_time > min_obs_time)
        
        # Also apply validity mask for seeing, depth, ellipticity
        seeinglist = np.array([v if v is not None else np.nan for v in seeinglist], dtype=float)
        depthlist = np.array([v if v is not None else np.nan for v in depthlist], dtype=float)
        ellipticitylist = np.array([v if v is not None else np.nan for v in ellipticitylist], dtype=float)
        valid_value_mask = (~np.isnan(seeinglist)) & (~np.isnan(depthlist)) & (~np.isnan(ellipticitylist))
        if not np.any(valid_value_mask):
            return []
                    
        # Apply limits mask
        valid_seeing_mask = seeinglist < seeing_limit
        valid_ellipticity_mask = ellipticitylist < ellipticity_limit
        valid_depth_mask = depthlist > depth_limit
        
        # Final combined mask (same length as target_imglist)
        combined_mask = (
            valid_obs_mask &
            valid_value_mask &
            valid_seeing_mask &
            valid_ellipticity_mask &
            valid_depth_mask
        )
        if not np.any(combined_mask):
            return []
        
        # Apply final mask
        ell_all = np.array(ellipticitylist)[valid_value_mask]
        see_all = np.array(seeinglist)[valid_value_mask]
        dep_all = np.array(depthlist)[valid_value_mask]
        obsdate_all = np.array(obsdatelist)[valid_value_mask]
        
        ell_filtered = np.array(ellipticitylist)[combined_mask]
        see_filtered = np.array(seeinglist)[combined_mask]
        dep_filtered = np.array(depthlist)[combined_mask]
        imgs_filtered = np.array(target_imglist)[combined_mask]
        obsdate_filtered = np.array(obsdate_time)[combined_mask]
        from sklearn.preprocessing import MinMaxScaler
        from matplotlib.gridspec import GridSpec

        # Normalize
        scaler = MinMaxScaler()
        ell_norm = scaler.fit_transform(ell_filtered.reshape(-1, 1)).flatten()
        see_norm = scaler.fit_transform(see_filtered.reshape(-1, 1)).flatten()
        dep_norm = scaler.fit_transform(dep_filtered.reshape(-1, 1)).flatten()

        # Compute combined score
        # You can adjust weights if needed
        score = (1 - ell_norm) * weight_ellipticity + (1 - see_norm) * weight_seeing + dep_norm * weight_depth

        # Rank and select best images
        sorted_idx = np.argsort(score)[::-1]  # descending
        best_images = imgs_filtered[sorted_idx]
        if max_numbers is None:
            num_select = max(1, int(len(sorted_idx)))  # select top 90%
        else:
            num_select = max_numbers
        selected_idx = sorted_idx[:num_select]

        # Top N or just best
        best_image = best_images[0]
        
        # Data for plotting
        x_all = np.array(see_all)
        y_all = np.array(dep_all)
        c_all = np.array(ell_all)
        x_valid = np.array(see_filtered)
        y_valid = np.array(dep_filtered)
        c_valid = np.array(ell_filtered)
        x_selected = x_valid[selected_idx]
        y_selected = y_valid[selected_idx]
        c_selected = c_valid[selected_idx]
        idx_best = sorted_idx[0]
        x_best = x_valid[idx_best]
        y_best = y_valid[idx_best]
        c_best = c_valid[idx_best]

        # Create marker masks with full length
        marker_sizes_full = np.where(combined_mask, 50, 10)
        marker_alphas_full = np.where(combined_mask, 0.8, 0.2)

        # Apply valid_value_mask to match x_all, y_all
        marker_sizes = marker_sizes_full[valid_value_mask]
        marker_alphas = marker_alphas_full[valid_value_mask]

        # Calculate percentiles (90%, 75%, and 50%)
        p90_x, p75_x, p50_x, p25_x, p10_x = np.percentile(x_all, [10, 25, 50, 75, 90])
        p90_y, p75_y, p50_y, p25_y, p10_y = np.percentile(y_all, [90, 75, 50, 25, 10])

        # Calculate the number of images for each percentile
        num_images_p90 = np.sum((x_all <= p90_x) & (y_all >= p90_y))  # Number of images below or equal to the 10th percentile
        num_images_p75 = np.sum((x_all <= p75_x) & (y_all >= p75_y))  # Number of images below or equal to the 25th percentile
        num_images_p50 = np.sum((x_all <= p50_x) & (y_all >= p50_y))  # Number of images below or equal to the 50th percentile
        num_images_p25 = np.sum((x_all <= p25_x) & (y_all >= p25_y))  # Number of images below or equal to the 75th percentile

        # Create figure with GridSpec layout
        if visualize:
            fig = plt.figure(figsize=(6, 6), dpi=300)
            gs = GridSpec(4, 4, fig, wspace=1.5, hspace=0.5)

            # Create scatter plot
            ax_main = fig.add_subplot(gs[1:, :-1])
            sc = ax_main.scatter(x_all, y_all,
                                c=c_all,
                                s=marker_sizes,
                                alpha=marker_alphas,
                                cmap='viridis', edgecolors='k', linewidths=0.5,
                                label = f'All images ({len(x_all)})')        
            ax_main.scatter(0,0, s = 10, alpha = 0.2, label = f'Filtered out images ({len(x_all) - len(x_selected)})')
            cbar = fig.colorbar(sc, ax=ax_main, pad=0.01)
            cbar.set_label('Ellipticity')
            ax_main.axvline(p90_x, color='r', linestyle='--')
            ax_main.axvline(p75_x, color='b', linestyle='--')
            ax_main.axvline(p50_x, color='g', linestyle='--')
            ax_main.axvline(p25_x, color='k', linestyle='--')
            ax_main.axhline(p90_y, color='r', linestyle='--')
            ax_main.axhline(p75_y, color='b', linestyle='--')
            ax_main.axhline(p50_y, color='g', linestyle='--')
            ax_main.axhline(p25_y, color='k', linestyle='--')
            ax_main.set_xlim(p90_x - 0.5, p10_x + 0.5)
            ax_main.set_ylim(p10_y - 1, p90_y + 1)
            ax_main.set_xlabel('Seeing [arcsec]')
            ax_main.set_ylabel('Depth [AB]')
            ax_main.scatter(x_selected, y_selected, marker='*', s=200, c='red', edgecolors='black', label=f'Selected ({len(selected_idx)}) images')
            ax_main.scatter(x_best, y_best, marker='*', s=200, c='red', edgecolors='black')
            ax_main.text(x_best, y_best + 0.3,
                        f"Best\nSeeing = {x_best:.2f} arcsec\nDepth = {y_best:.2f} AB\nEllipticity = {c_best:.2f}",
                        color='red', fontsize=8, ha='center', va='bottom',
                        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
            ax_main.legend(loc='upper right', fontsize=8, frameon=True)


            # Create top histogram
            ax_histx = fig.add_subplot(gs[0, :-1], sharex=ax_main)
            ax_histx.hist(x_valid, bins=30, color='black', edgecolor='black', alpha=0.7)
            ax_histx.spines['top'].set_visible(False)  # Hide top spine
            ax_histx.spines['right'].set_visible(False)  # Hide right spine

            # Create right histogram
            ax_histy = fig.add_subplot(gs[1:, -1], sharey=ax_main)
            ax_histy.hist(y_valid, bins=30, color='black', edgecolor='black', alpha=0.7, orientation='horizontal')
            ax_histy.spines['top'].set_visible(False)  # Hide top spine
            ax_histy.spines['right'].set_visible(False)  # Hide right spine

            # Set limits for histograms to fit within the black box
            ax_histx.set_xlim(ax_main.get_xlim())
            ax_histy.set_ylim(ax_main.get_ylim())

            # Plot vertical regions for percentiles in histograms
            ax_histx.axvline(p90_x, color='r', linestyle='--', label='90%')
            ax_histx.axvline(p75_x, color='b', linestyle='--', label='75%')
            ax_histx.axvline(p50_x, color='g', linestyle='--', label='50%')
            ax_histx.axvline(p25_x, color='k', linestyle='--', label='25%')

            ax_histy.axhline(p90_y, color='r', linestyle='--', label='90%')
            ax_histy.axhline(p75_y, color='b', linestyle='--', label='75%')
            ax_histy.axhline(p50_y, color='g', linestyle='--', label='50%')
            ax_histy.axhline(p25_y, color='k', linestyle='--', label='25%')

            # Add text annotation in the upper right region of the scatter plot
            text = f'Percentile (# of images, Seeing, Depth):\n'
            text += f'90% ({num_images_p90}, {p90_x:.2f}, {p90_y:.2f})\n'
            text += f'75% ({num_images_p75}, {p75_x:.2f}, {p75_y:.2f})\n'
            text += f'50% ({num_images_p50}, {p50_x:.2f}, {p50_y:.2f})\n'
            text += f'25% ({num_images_p25}, {p25_x:.2f}, {p25_y:.2f})'
            ax_main.text(0.5, 0.15, text,
                        ha='center', va='center',
                        transform=ax_main.transAxes,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            
            from matplotlib.lines import Line2D

            dashed_lines = [
                Line2D([0], [0], color='red', linestyle='--', label='90%'),
                Line2D([0], [0], color='blue', linestyle='--', label='75%'),
                Line2D([0], [0], color='green', linestyle='--', label='50%'),
                Line2D([0], [0], color='black', linestyle='--', label='25%')
            ]

            fig.legend(handles=dashed_lines,
                    loc='upper right',
                    bbox_to_anchor=(0.95, 0.95),
                    fontsize=10, frameon=True)
            plt.tight_layout()
            plt.show()
        
        selected_images = imgs_filtered[selected_idx]
        
        return selected_images

    def match_zeropoints(self,
                         target_imglist: List[Union[ScienceImage, CalibrationImage]],
                         target_errormaplist: Optional[List[Errormap]] = None,
                         method: str = 'median',
                         zp_key: str = 'ZP_APER_1',
                         
                         # Other parameters
                         save: bool = False,
                         overwrite: bool = False,
                         verbose: bool = True,
                         n_proc: int = 8,
                         **kwargs):
        """
        Match the zero points of multiple images.
        
        Parameters
        ----------
        target_imglist : List[Union[ScienceImage, CalibrationImage]]
            List of images to match zero points.
        target_errormaplist : Optional[List[Errormap]]
            Optional list of error maps to match zero points.
        method : str
            Method to determine reference ZP ('min', 'max', or 'median').
        zp_key : str
            Header keyword for zero point.
        save : bool
            Whether to save scaled images and error maps.
        overwrite : bool
            Whether to overwrite existing files.
        verbose : bool
            Print progress messages.
        n_proc : int, optional
            The number of processes to use for the stacking.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        (scaled_imglist, scaled_errormaplist) : Tuple[List[Union[ScienceImage, CalibrationImage]], Optional[List[Errormap]]]
            List of zero-matched images and optionally zero-matched error maps.
        """
        
        # Extract ZPs
        zp_values = []
        for img in target_imglist:
            if img.header is None:
                img.load_header()
            if zp_key not in img.header:
                raise ValueError(f"Missing ZP key '{zp_key}' in {img.path}")
            zp_values.append(float(img.header[zp_key]))

        # Determine reference ZP
        if method == 'min':
            ref_zp = np.min(zp_values)
        elif method == 'max':
            ref_zp = np.max(zp_values)
        elif method == 'median':
            ref_zp = np.median(zp_values)
        else:
            raise ValueError(f"Invalid method: {method}")

        if verbose:
            print(f"[match_zeropoints] Reference ZP: {ref_zp:.3f} using '{method}'")

        # Prepare tasks
        tasks = [
            (img, errormap, ref_zp, zp_key, save, overwrite)
            for img, errormap in zip(target_imglist, target_errormaplist or [None] * len(target_imglist))
        ]
        # Run with process_map
        if verbose:
            desc = "Matching ZP..."
        else:
            desc = None
        results = process_map(_scale_worker, tasks, desc=desc, max_workers=n_proc)
        scaled_imglist, scaled_errormaplist = zip(*results)

        return list(scaled_imglist), list(scaled_errormaplist) if target_errormaplist else None
    
    def match_seeing(self,
                     target_imglist: List[Union[ScienceImage, CalibrationImage]],
                     target_errormaplist: Optional[List[Errormap]] = None,
                     target_bkglist: Optional[List[Background]] = None,
                     seeing_key: str = 'SEEING',
                     kernel: str = 'gaussian', # gaussian or image
                     
                     # Other parameters
                     save: bool = False,
                     overwrite: bool = False,
                     visualize: bool = True,
                     verbose: bool = True,
                     **kwargs) -> List[Union[ScienceImage, CalibrationImage]]:
        """
        Match the seeing (PSF FWHM) of multiple images using convolution.

        Parameters
        ----------
        target_imglist : List[Union[ScienceImage, CalibrationImage]]
            List of images to match seeing
        target_errormaplist : Optional[List[Errormap]]
            Corresponding error maps to also convolve
        target_bkglist : Optional[List[Background]]
            Corresponding background maps to also convolve
        seeing_key : str
            Header keyword for seeing/FWHM in pixel units
        kernel : str
            Convolution kernel type ('gaussian')
        save : bool
            Whether to save the matched images and error maps
        overwrite : bool
            Whether to overwrite existing files
        visualize : bool
            Whether to visualize the matched images and error maps
        verbose : bool
            Print progress messages
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        (matched_imglist, matched_errormaplist) : Tuple[List[Union[ScienceImage, CalibrationImage]], Optional[List[Errormap]]]
            List of seeing-matched images and optionally their convolved error maps
        """
        from photutils.psf.matching import create_matching_kernel
        from astropy.convolution import convolve_fft

        # Get all seeing values
        seeing_values = []
        for target_img in target_imglist:
            if seeing_key not in target_img.header:
                raise ValueError(f"Seeing key '{seeing_key}' not found in header of {target_img.path}")
            seeing_values.append(float(target_img.header[seeing_key]/np.mean(target_img.pixelscale)))  # Convert to pixel units
        
        # Determine reference seeing based on method
        ref_idx = np.argmax(seeing_values)
        ref_seeing = seeing_values[ref_idx]  # Get the actual reference seeing value
        ref_target_img = target_imglist[ref_idx]
        ref_target_bkg = target_bkglist[ref_idx] if target_bkglist is not None else None
        ref_psf_size = int(np.ceil(ref_seeing * 6)) | 1
        
        # If kernel is 'image', build the PSF model from the reference image
        if kernel.lower() == 'image':
            if ref_target_img.status.BKGSUB['status']:
                ref_target_bkg = None
            ref_psf_model = self.psfphot.build_epsf_model_psfex(
                target_img=ref_target_img,
                target_bkg=ref_target_bkg,
                fwhm_estimate_pixel=ref_seeing,
                num_grids=1,
                oversampling=1,
                psf_size = ref_psf_size,
                verbose=verbose,
                visualize=True
            )[(0, 0)].data
        elif kernel.lower() == 'gaussian':
            pass
        else:
            raise ValueError(f"Unsupported kernel type '{kernel}'. Use 'image' or 'gaussian'")
        
        self.helper.print(f"Matching seeing to reference FWHM = {ref_seeing:.3f} using {kernel} method", verbose)
        
        # Convolve each image to match reference seeing
        matched_imglist = []
        matched_errormaplist = [] if target_errormaplist is not None else None
        iterator = tqdm(zip(target_imglist, target_errormaplist or [None] * len(target_imglist)), desc='Matching seeing...') if verbose else zip(target_imglist, target_errormaplist or [None] * len(target_imglist))
        
        for idx, (target_img, target_errormap) in enumerate(iterator):
            current_seeing = float(target_img.header[seeing_key] / np.mean(target_img.pixelscale))  # Convert to pixel units

            # Define output path
            if not overwrite:
                target_img_path = target_img.savepath.savedir / f"convolved_{target_img.savepath.savepath.name}"
                if target_errormap is not None:
                    target_errormap_path = target_errormap.savepath.savedir / f"convolved_{target_errormap.savepath.savepath.name}"
            else:
                target_img_path = target_img.savepath.savepath
                target_errormap_path = target_errormap.savepath.savepath if target_errormap is not None else None
                        
            # Skip convolution if seeing already matches (within small tolerance)
            seeing_diff = abs(current_seeing - ref_seeing) * np.mean(target_img.pixelscale)  # Convert to arcseconds
            tolerance = 3e-1  # 0.3arcsec
            if seeing_diff < tolerance:
                matched_img = target_img.copy()
                matched_img.path = target_img_path
                matched_imglist.append(matched_img)

                if matched_errormaplist is not None and target_errormap is not None:
                    matched_errormap = target_errormap.copy()
                    matched_errormap.path = target_errormap_path
                    matched_errormaplist.append(matched_errormap)

                if save:
                    matched_img.write()
                    if matched_errormaplist is not None:
                        matched_errormap.write()
                
                self.helper.print(f"Skipping convolution for {target_img.path.name} (already matches reference seeing with the tolerenace {tolerance}arcsec) [diff = {seeing_diff}arcsec]", verbose)
                continue
            
            # Convolve the target image
            target_bkg = target_bkglist[idx] if target_bkglist is not None else None
            if kernel.lower() == 'gaussian':
                convolved_data, updated_header = self.helper.img_convolve(
                    target_img=target_img.data,
                    input_type='image',
                    kernel=kernel,
                    target_header=target_img.header,
                    fwhm_target=current_seeing,
                    fwhm_reference=ref_seeing,
                    fwhm_key=seeing_key,
                    verbose=verbose
                )
            
            else:
                # Build the PSF model for the target image
                target_psf_model = self.psfphot.build_epsf_model_psfex(
                    target_img=target_img,
                    target_bkg=target_bkg,
                    fwhm_estimate_pixel=current_seeing,
                    num_grids=1,
                    oversampling=1,
                    psf_size = ref_psf_size,
                    verbose=verbose,
                    visualize=False
                )[(0, 0)].data

                psf_input = target_psf_model / target_psf_model.sum()
                psf_ref = ref_psf_model / ref_psf_model.sum()
                from photutils.psf.matching import CosineBellWindow
                from photutils.psf.matching import TopHatWindow
                window = CosineBellWindow(alpha=1)
                #window = TopHatWindow(0.35)
                conv_kernel = create_matching_kernel(psf_input, psf_ref, window = window)
                if visualize:
                    fig, axes = plt.subplots(1,3, figsize=(12, 4), dpi = 300)
                    axes[0].imshow(psf_input, cmap='gray', origin='lower')
                    axes[0].set_title('Input PSF')
                    axes[1].imshow(psf_ref, cmap='gray', origin='lower')
                    axes[1].set_title('Reference PSF')
                    axes[2].imshow(conv_kernel, cmap='gray', origin='lower')
                    axes[2].set_title('Convolution Kernel')

                convolved_data = convolve_fft(target_img.data, conv_kernel, normalize_kernel=True)
                updated_header = target_img.header.copy()
                updated_header[seeing_key] = ref_seeing * np.mean(target_img.pixelscale)
                
            matched_img = type(target_img)(path=target_img_path, telinfo=target_img.telinfo, status=target_img.status, load=False)
            matched_img.data = convolved_data
            matched_img.header = updated_header
            matched_imglist.append(matched_img)
            
            if matched_errormaplist is not None and target_errormap is not None:
                if kernel.lower() == 'image':
                    convolved_error = convolve_fft(target_errormap.data, conv_kernel, normalize_kernel=True) 
                else:
                    convolved_error, _ = self.img_convolve(
                    target_img=target_errormap.data,
                    input_type='error',
                    kernel=kernel,
                    target_header=target_errormap.header,
                    fwhm_target=current_seeing,
                    fwhm_reference=ref_seeing,
                    fwhm_key=seeing_key,
                    verbose=verbose
                    )

                matched_errormap = Errormap(path=target_errormap_path, emaptype=target_errormap.emaptype, status=target_errormap.status, load=False)
                matched_errormap.data = convolved_error
                matched_errormap.header = target_errormap.header.copy()
                matched_errormaplist.append(matched_errormap)
            
            if save:
                matched_img.write()
                if matched_errormaplist is not None and target_errormap is not None:
                    matched_errormap.write()
        
        self.helper.print(f"Successfully matched seeing for {len(matched_imglist)} images", verbose)
        return matched_imglist, matched_errormaplist
# %%
