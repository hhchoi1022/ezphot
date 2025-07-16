

#%%
import numpy as np
import glob
import sep
from astropy.io.fits import Header
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.background import Background2D, MedianBackground, MeanBackground, SExtractorBackground
from pathlib import Path
from typing import Union, Optional
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numexpr as ne

from tippy.helper import Helper
from tippy.photometry import TIPBackground
from tippy.photometry import TIPMasking
from tippy.image import ScienceImage, ReferenceImage, CalibrationImage 
from tippy.image import Mask, Errormap, Background
#%%
class TIPErrormap(Helper): ############## CHECKED ##############
    def __init__(self):
        """
        Initialize the BackgroundEstimator class with default parameters.
        """
        self.background = TIPBackground()
        self.mask = TIPMasking()
        super().__init__()

    def calculate_from_image(self,
                             target_img: Union[ScienceImage, ReferenceImage],
                             mbias_img: CalibrationImage,
                             mdark_img: CalibrationImage,                             
                             mflat_img: CalibrationImage,
 
                             # Calibration inputs
                             mflaterr_img: Optional[Errormap] = None,
                             
                             # Others
                             save: bool = True,
                             verbose: bool = True,
                             visualize: bool = True,
                             save_fig: bool = False,
                             **kwargs
                             ):
        # --- Inputs ---
        data = target_img.data                   # assumed to be calibrated science image in ADU
        ncombine = target_img.ncombine or 1      # number of science images combined to make master science image
        mbias = mbias_img.data                   # bias image in ADU
        ncombine_bias = mbias_img.ncombine or 9  # number of bias images combined to make master bias
        mdark = mdark_img.data                   # dark image in ADU
        ncombine_dark = mdark_img.ncombine or 9  # number of dark images combined to make master dark
        mflat = mflat_img.data                   # normalized flat image (unitless, ~1.0)
        egain = target_img.egain                 # electrons/ADU
        if target_img.ncombine is None:
            self.print('Warning: target_img.ncombine is None. Using 1 as default value.', verbose)
        if mbias_img.ncombine is None:
            self.print('Warning: mbias_img.ncombine is None. Using 9 as default value.', verbose)
        if mdark_img.ncombine is None:
            self.print('Warning: mdark_img.ncombine is None. Using 9 as default value.', verbose)
        
        # --- Readout noise from master bias ---
        ny, nx = mbias.shape
        y0 = ny // 3
        y1 = 2 * ny // 3
        x0 = nx // 3
        x1 = 2 * nx // 3
        central_bias = mbias[y0:y1, x0:x1] # Central region of the bias image
        mbias_var = np.var(central_bias)          # in ADU
        sbias_var = mbias_var * ncombine_bias  # in ADU^2
        readout_noise = np.sqrt(sbias_var)  # Readout noise in ADU
        
        # --- Readout noise from master dark ---
        mdark_var = sbias_var / ncombine_dark + mbias_var

        # 
        if mflaterr_img is not None:
            mflat_err = mflaterr_img.data
            mflat_var = mflat_err**2
            mflaterr_path = str(mflaterr_img.path)
        else:
            mflat_err = 0
            mflat_var = 0
            mflaterr_path = None

        signal = np.abs(data + mdark)
        error_map = ne.evaluate("sqrt((signal / egain / mflat + sbias_var / mflat**2 + signal**2 * mflat_var / mflat**2) + mbias_var + mdark_var)")
        # HERE, mflat**2? or mflat? with signal / egain /
        target_errormap = Errormap(target_img.savepath.srcrmspath, emaptype = 'sourcerms' ,load = False)
        target_errormap.data = error_map
        target_errormap.header = target_img.header

        # Update header
        update_header_kwargs = dict(
            TGTPATH = str(target_img.path),
            BIASPATH = str(mbias_img.path),
            DARKPATH = str(mdark_img.path),
            FLATPATH = str(mflat_img.path),
            EFLTPATH = mflaterr_path,
            )
        target_errormap.header.update(update_header_kwargs)
        
        # Update header of the target image
        update_header_kwargs_image = dict(
            EMAPPATH = str(target_errormap.path),
            )
        target_img.header.update(update_header_kwargs_image)
        
        ## Update status          
        event_details = dict(type = 'sourcerms', readnoise = float(readout_noise), mbias = str(mbias_img.path), mdark = str(mdark_img.path), mflat =str(mflat_img.path), mflaterr = mflaterr_path)
        target_errormap.add_status("error_propagation", **event_details)
        
        if save:
            target_errormap.write()
        
        if visualize:
            save_path = None
            if save_fig:
                save_fig = str(target_errormap.savepath.savepath) + '.png'
            self._visualize(
                target_img = target_img,
                target_errormap = target_errormap,
                target_bkg = None,
                save_path = save_path,
                show = visualize
            )
        
        return target_errormap
    

    def calculate_from_bkg(self,
                           target_bkg: Background,
                           mbias_img: CalibrationImage,
                           mdark_img: CalibrationImage,                             
                           mflat_img: CalibrationImage,

                           mflaterr_img: Errormap = None,
                           ncombine: Optional[int] = None,
                           
                           # Other parameters
                           save: bool = False,
                           verbose: bool = True,
                           visualize: bool = True,
                           save_fig: bool = False,
                           **kwargs
                           ):  
        # --- Inputs ---
        data = target_bkg.data                   # assumed to be calibrated science image in ADU
        if ncombine is None:      # number of science images combined to make master science image
            self.print('Warning: ncombine is None. Using 1 as default value.', verbose)

        mbias = mbias_img.data                   # bias image in ADU
        ncombine_bias = mbias_img.ncombine or 9  # number of bias images combined to make master bias
        mdark = mdark_img.data                   # dark image in ADU
        ncombine_dark = mdark_img.ncombine or 9  # number of dark images combined to make master dark
        mflat = mflat_img.data                   # normalized flat image (unitless, ~1.0)
        egain = target_bkg.egain                 # electrons/ADU
        if mbias_img.ncombine is None:
            self.print('Warning: mbias_img.ncombine is None. Using 9 as default value.', verbose)
        if mdark_img.ncombine is None:
            self.print('Warning: mdark_img.ncombine is None. Using 9 as default value.', verbose)
        
        # --- Readout noise from master bias ---
        ny, nx = mbias.shape
        y0 = ny // 3
        y1 = 2 * ny // 3
        x0 = nx // 3
        x1 = 2 * nx // 3
        central_bias = mbias[y0:y1, x0:x1] # Central region of the bias image
        mbias_var = np.var(central_bias)          # in ADU
        sbias_var = mbias_var * ncombine_bias  # in ADU^2
        readout_noise = np.sqrt(sbias_var)  # Readout noise in ADU
        
        # --- Readout noise from master dark ---
        mdark_var = sbias_var / ncombine_dark + mbias_var

        # 
        if mflaterr_img is not None:
            mflat_err = mflaterr_img.data
            mflat_var = mflat_err**2
            mflaterr_path = str(mflaterr_img.path)
        else:
            mflat_err = 0
            mflat_var = 0
            mflaterr_path = None

        signal = np.abs(data + mdark)
        error_map = ne.evaluate("sqrt(signal / egain / mflat + sbias_var / mflat**2 + signal**2 * mflat_var / mflat**2 + mbias_var / mflat**2 + mdark_var / mflat**2)")

        target_errormap = Errormap(str(target_bkg.path).replace('bkgmap','bkgrms'), emaptype = 'bkgrms', load = False)
        target_errormap.data = error_map
        target_errormap.header = target_bkg.header

        # Update header
        update_header_kwargs = dict(
            BKGPATH = str(target_bkg.path),
            BIASPATH = str(mbias_img.path),
            DARKPATH = str(mdark_img.path),
            FLATPATH = str(mflat_img.path),
            EFLTPATH = mflaterr_path,
            )
        target_errormap.header.update(update_header_kwargs)
        
        ## Update status          
        event_details = dict(type = 'sourcerms', readnoise = float(readout_noise), mbias = str(mbias_img.path), mdark = str(mdark_img.path), mflat =str(mflat_img.path), mflaterr = mflaterr_path)
        target_errormap.add_status("error_propagation", **event_details)
        
        if save:
            target_errormap.write()

        if visualize:
            save_path = None
            if save_fig:
                save_fig = str(target_errormap.savepath.savepath) + '.png'
            self._visualize(
                target_img = None,
                target_errormap = target_errormap,
                target_bkg = target_bkg,
                save_path = save_path,
                show = visualize
            )
        
        return target_errormap

    def calculate_from_sourcemask(self,
                                  # Input parameters
                                  target_img: Union[ScienceImage, ReferenceImage],
                                  target_mask: Optional[Mask] = None,
                                  box_size: int = 128,
                                  filter_size: int = 3,
                                  errormap_type: str = 'bkgrms', # bkgrms or sourcerms
                                  mode: str = 'sep',

                                  # Iterative background estimation. Set n_iterations to 0 to skip
                                  n_iterations: int = 1,
                                  mask_sigma: float = 3.0,
                                  mask_radius_factor: float = 3,
                                  mask_saturation_level: float = 50000,
                                  bkg_estimator: str = 'median',
                                
                                  # Others
                                  save: bool = True,
                                  verbose: bool = True,
                                  visualize: bool = True,
                                  save_fig: bool = False,
                                  **kwargs
                                  ):
        if mode.lower() == 'sep':
            target_bkg, bkg = self.background.calculate_sep(
                target_img = target_img,
                target_mask = target_mask,
                box_size = box_size,
                filter_size = filter_size,
                n_iterations = n_iterations,
                mask_sigma = mask_sigma,
                mask_radius_factor = mask_radius_factor,
                mask_saturation_level = mask_saturation_level,
                return_bkg_instance = True,
                save = False,
                verbose = verbose,
                visualize = visualize,
                save_fig = False
            )
            # Calculate error map
            bkg_rms_map = bkg.rms()
        else:
            target_bkg, bkg = self.background.calculate_photutils(
                target_img = target_img,
                target_mask = target_mask,
                box_size = box_size,
                filter_size = filter_size,
                bkg_estimator = bkg_estimator,
                n_iterations = n_iterations,
                mask_sigma = mask_sigma,
                mask_radius_factor = mask_radius_factor,
                mask_saturation_level = mask_saturation_level,
                return_bkg_instance = True,
                save = False,
                verbose = verbose,
                visualize = visualize,
                save_fig = False)
            # Calculate error map
            bkg_rms_map = bkg.background_rms
            
        if errormap_type.lower() == 'sourcerms':
            egain = target_img.egain
            bkg_map = target_bkg.data
            source_var_map = np.abs(self.operation.subtract(target_img.data.astype(np.float32), bkg_map)) / egain
            error_map = self.operation.sqrt(self.operation.power(bkg_rms_map,2) + source_var_map)
            target_errormap = Errormap(target_img.savepath.srcrmspath, emaptype = 'sourcerms', load = False)
        else:
            error_map = bkg_rms_map
            target_errormap = Errormap(target_img.savepath.bkgrmspath, emaptype = 'bkgrms', load = False)

        target_errormap.data = error_map
        target_errormap.header = target_img.header

        # Update header
        update_header_kwargs = dict(
            TGTPATH = str(target_img.path),
            BKGPATH = str(target_bkg.path),
            MASKPATH = str(target_bkg.info.MASKPATH),
            )
        target_errormap.header.update(update_header_kwargs)
        
        ## Update status          
        if errormap_type.lower() == 'sourcerms':
            event_details = dict(type = 'sourcerms', bkg_path = str(target_bkg.path), bkg_mask = str(target_bkg.info.MASKPATH), box_size = box_size, filter_size = filter_size, n_iterations = n_iterations, mask_sigma = mask_sigma, mask_radius_factor = mask_radius_factor)
        else:
            event_details = dict(type = 'bkgrms', bkg_path = str(target_bkg.path), bkg_mask = str(target_bkg.info.MASKPATH), box_size = box_size, filter_size = filter_size, n_iterations = n_iterations, mask_sigma = mask_sigma, mask_radius_factor = mask_radius_factor)

        target_errormap.add_status("sourcemask", **event_details)
        
        if save:
            target_errormap.write()
        
        if visualize:
            save_path = None
            if save_fig:
                save_fig = str(target_errormap.savepath.savepath) + '.png'
            self._visualize(
                target_img = target_img,
                target_errormap = target_errormap,
                target_bkg = target_bkg,
                save_path = save_path,
                show = visualize
            )
        return target_errormap, target_bkg, bkg
    
    def _visualize(self,
                   target_errormap: Union[Errormap],
                   target_img: Union[ScienceImage, ReferenceImage, CalibrationImage] = None,
                   target_bkg: Union[Background] = None,
                   save_path: str = None,
                   show: bool = False):
        from astropy.visualization import ZScaleInterval
        interval = ZScaleInterval()        

        """
        Visualize the image and mask.
        """
        panels = []
        titles = []
        
        def downsample(data, factor=4):
            return data[::factor, ::factor]
        
        if target_img is not None:
            image_data_small = downsample(target_img.data)
            vmin, vmax = interval.get_limits(image_data_small)
            panels.append((image_data_small, dict(cmap='Greys_r', vmin=vmin, vmax=vmax)))
            titles.append("Original Image")

        if target_bkg is not None:
            bkg_map_small = downsample(target_bkg.data)
            panels.append((bkg_map_small, dict(cmap='viridis')))
            titles.append("2D Background")

        error_map_small = downsample(target_errormap.data)
        vmin, vmax = interval.get_limits(error_map_small)
        panels.append((error_map_small, dict(cmap='Greys_r', vmin=vmin, vmax=vmax)))
        titles.append("Error map")
            
        n = len(panels)
        if n == 0:
            print("Nothing to visualize.")
            return

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]  # make iterable

        for i, (data, imshow_kwargs) in enumerate(panels):
            ax = axes[i]
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(data, origin='lower', **imshow_kwargs)
            ax.set_title(titles[i])
            fig.colorbar(im, cax=cax, orientation='vertical')

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close(fig)

        


#%% # From image (total)
if __name__ == '__main__':

    from tippy.photometry import TIPPreprocess

    target_path  =Path('/data/data1/factory_hhchoi/data/scidata/7DT/7DT_C361K_HIGH_1x1/T22956/7DT11/m425/calib_7DT11_T22956_20250329_044258_m425_100.fits')
    target_img = ScienceImage(path = target_path, telinfo=Helper().get_telinfo('7DT', 'C361K', 'HIGH', 1), load= True)
    target_bkg = Background(target_img.savepath.bkgpath, load = True)
    target_mask = Mask(str(target_img.savepath.srcmaskpath), masktype = 'source', load = True)
    self = TIPErrormap()    
    preprocess = TIPPreprocess()
    mbias_path  = preprocess.get_masterframe_from_image(target_img, imagetyp = 'bias')[0]['file']
    mdark_path  = preprocess.get_masterframe_from_image(target_img, imagetyp = 'dark')[0]['file']
    mflat_path  = preprocess.get_masterframe_from_image(target_img, imagetyp = 'flat')[0]['file']
    mbias_img = CalibrationImage(
        path  = mbias_path,
        telinfo = Helper().get_telinfo('7DT', 'C361K', 'HIGH', '1'),
        load = True)
    mdark_img = CalibrationImage(
        path = mdark_path,
        telinfo = Helper().get_telinfo('7DT', 'C361K', 'HIGH', '1'),
        load = True)
    mflat_img = CalibrationImage(
        path = mflat_path,
        telinfo = Helper().get_telinfo('7DT', 'C361K', 'HIGH', '1'),
        load = True)
    target_bkg = Background(
        path = target_img.savepath.bkgpath,
        load = True,
    )
    mflat_err = None
    
    readout_noise = 6.0
    dark_current = None
    
    box_size: int = 128
    filter_size: int = 3
    errormap_type = 'total'

    # Iterative background estimation
    n_iterations: int = 2
    mask_sigma: float = 3.0
    mask_radius_factor: float = 3
    mask_saturation_level: float = 50000
    
    # Others
    save: bool = True
    verbose: bool = True
    visualize: bool = True
    save_fig = True
#%%|
    target_bkgrms = self.calculate_from_sourcemask(target_img = target_img,
                                    target_mask = target_mask,
                                    box_size = box_size,
                                    filter_size = filter_size,
                                    errormap_type = 'bkgrms',
                                    n_iterations = n_iterations,
                                    mask_sigma = mask_sigma,
                                    mask_radius_factor = mask_radius_factor,
                                    mask_saturation_level = mask_saturation_level,
                                    verbose = verbose,
                                    visualize = visualize,
                                    save = save,
                                    save_fig = save_fig)
    
    target_bkgrms = self.calculate_from_image(
        target_img = target_img,
        mbias_img = mbias_img,
        mdark_img = mdark_img,
        mflat_img = mflat_img,
        mflaterr_img = mflat_err,
        save = save,
        verbose = verbose,
        visualize = visualize,
        save_fig = save_fig
    )
    
    target_bkgrms_sq = self.calculate_from_bkg(
        target_bkg = target_bkg,
        mbias_img = mbias_img,
        mdark_img = mdark_img,
        mflat_img = mflat_img,
        mflaterr_img = None,
        save = save,
        verbose = verbose,
        visualize = visualize,
        save_fig = save_fig)
#%%
if __name__ == '__main__':

    bkgrms_mask = target_bkgrms_mask[0].data
    bkgrms_flat = target_bkgrms.data
    bkgrms_flatsq = target_bkgrms_sq.data

    case1 = bkgrms_mask - bkgrms_flat
    case2 = bkgrms_mask - bkgrms_flatsq
    case3 = bkgrms_flat - bkgrms_flatsq
    case6_div = bkgrms_flat / bkgrms_flatsq
    case5_div = bkgrms_mask / bkgrms_flatsq
    case4_div = bkgrms_mask / bkgrms_flat
    plt.imshow(bkgrms_mask, origin='lower', cmap='Greys_r', vmin=6, vmax=18)
    plt.show()
    plt.imshow(bkgrms_flat, origin='lower', cmap='Greys_r', vmin=6, vmax=18)
    plt.show()
    plt.imshow(bkgrms_flatsq, origin='lower', cmap='Greys_r', vmin=6, vmax=18)

    plt.imshow(case4_div, origin='lower', cmap='Greys_r', vmin=0.9, vmax=1.1)
    plt.imshow(case6_div, origin='lower', cmap='Greys_r', vmin=0.9, vmax=1.1)

#%% # From image (background)
if __name__ == '__main__':
    self = TIPErrormap()
    dark_path = Path('/data/data1/factory/master_frame_1x1_gain2750/7DT02/dark/100-20250324-dark.fits')
    flat_path = Path('/data/data1/factory/master_frame_1x1_gain2750/7DT02/flat/20250324-nm450.fits')
    target_path = list(Path('/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T00176/7DT02/m450').glob('*.fits'))[0]
    target_img =  ScienceImage(
        path = target_path,
        telinfo = Helper().get_telinfo('7DT', 'C361K', 'HIGH', '1'),
        load = True)
    mdark_img = CalibrationImage(
        path = dark_path,
        telinfo = Helper().get_telinfo('7DT', 'C361K', 'HIGH', '1'),
        load = False)
    mdark_img.load()
    mdark_img.add_header(IMGTYPE = 'DARK')
    mdark_img.add_header(TELNAME = '7DT02')
    mflat_img = CalibrationImage(
        path = flat_path,
        telinfo = Helper().get_telinfo('7DT', 'C361K', 'HIGH', '1'),
        load = False)
    mflat_img.load()
    mflat_img.add_header(IMGTYPE = 'FLAT')
    mflat_img.add_header(TELNAME = '7DT02')
    
    readout_noise = None
    dark_current = None
    error_map = self.calculate_from_image(
        target_img = target_img,
        readout_noise = readout_noise,
        dark_current = dark_current,
        mdark_img = mdark_img,
        mflat_img = mflat_img,
        mflat_err = None,
        save = True,
        visualize = True
    )

#%%
if __name__ == '__main__':
    from tippy.utils import SDTData
    filelist_dict = SDTData().show_scisourcedata('T00176')
    target_path = Path(filelist_dict['g'][0])
    self = TIPErrormap()
    target_mask = None#Mask(str(target_path) + '.mask', load = True)
    target_img =  ScienceImage(
        path = target_path,
        telinfo = Helper().get_telinfo('7DT', 'C361K', 'HIGH', '1'),
        load = True)
    box_size: int = 128
    filter_size: int = 3
    errormap_type = 'total'

    # Iterative background estimation
    n_iterations: int = 0
    mask_sigma: float = 3.0
    mask_radius_factor: float = 3
    mask_saturation_level: float = 50000
    
    # Others
    verbose: bool = True
    visualize: bool = True
    save: bool = True
    self.calculate_from_sourcemask(target_img = target_img,
                                target_mask = target_mask,
                                box_size = box_size,
                                filter_size = filter_size,
                                errormap_type = 'total',
                                n_iterations = n_iterations,
                                mask_sigma = mask_sigma,
                                mask_radius_factor = mask_radius_factor,
                                mask_saturation_level = mask_saturation_level,
                                verbose = verbose,
                                visualize = visualize,
                                save = save)
#%%