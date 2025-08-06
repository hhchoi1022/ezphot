
#%%
from typing import Union, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np
from numba import jit
from photutils.aperture import CircularAperture
import astroscrappy as cr
import sep
from scipy.interpolate import griddata
import cv2

from tippy.methods import TIPMasking
from tippy.imageobjects import Mask, Background
from tippy.imageobjects import ScienceImage, ReferenceImage, CalibrationImage
from tippy.helper import Helper
#%%
class TIPBackground(Helper): ############## CHECKED ##############
    def __init__(self):
        """
        Initialize the BackgroundEstimator class with default parameters.
        """
        self.masking = TIPMasking()
        super().__init__()

    def mask_sources(self, 
                     # Input parameters
                     target_img: Union[ScienceImage, ReferenceImage, CalibrationImage],
                     target_mask: Mask = None,
                     sigma: float = 5.0, 
                     mask_radius_factor: float = 3,
                     saturation_level: float = 50000,
                     
                     # Others
                     save: bool = False,
                     verbose: bool = True,
                     visualize: bool = True,
                     save_fig: bool = False,
                     **kwargs):
        """
        Mask sources in the target image using the provided mask.
        """
        target_mask = self.masking.mask_sources(target_img = target_img,
                                                target_mask = target_mask,
                                                sigma = sigma,
                                                mask_radius_factor = mask_radius_factor,
                                                saturation_level = saturation_level,
                                                save = save,
                                                verbose = verbose,
                                                visualize = visualize,
                                                save_fig = save_fig
                                                )
        return target_mask
    
    def calculate_sep(self,
                      # Input parameters
                      target_img: Union[ScienceImage, ReferenceImage],
                      target_mask: Optional[Mask] = None,
                      is_2D_bkg: bool = True,
                      box_size: int = 32,
                      filter_size: int = 3,
                      
                      # Iterative background estimation
                      n_iterations: int = 1,
                      mask_sigma: float = 3.0,
                      mask_radius_factor: float = 3,
                      mask_saturation_level: float = 50000,
                     
                      # Others
                      save: bool = True,
                      verbose: bool = True,
                      visualize: bool = True,
                      save_fig: bool = False,
                      **kwargs
                      ):
        # Step 1: Load the image and mask
        mask_to_use = None
        if target_mask is None:
            target_mask = Mask(target_img.savepath.srcmaskpath, masktype = 'source', load=False)
        else:
            mask_to_use = target_mask.data.astype(bool)
            self.print("External mask is loaded.", verbose)
            
        image_data = target_img.data
        # If image_data is uint16, convert to float32
        if image_data.dtype == np.uint16:
            image_data = image_data.astype(np.float32)
        target_img.data = image_data

        # Create a mask of NaN values
        invalid_mask = ~np.isfinite(image_data)

        if np.any(invalid_mask):
            mask = invalid_mask.astype(np.uint8)
            image_filled = np.nan_to_num(image_data, nan=0, posinf=0.0, neginf=0.0).astype(np.float32)
            # Inpaint using the mask
            image_data = cv2.inpaint(image_filled, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Step 2: Background estimation
        image_data = self.to_native(image_data)
        
        bkg = sep.Background(image_data, 
                             mask=mask_to_use, 
                             bw=box_size, 
                             bh=box_size,
                             fw=filter_size, 
                             fh=filter_size)
        bkg_map = bkg.back()
        if is_2D_bkg:
            bkg_map = bkg_map
        else:
            bkg_val = np.mean(bkg_map)
            bkg_map = np.full_like(image_data, bkg_val, dtype=image_data.dtype)
        
        # Step 3: Iterative background estimation
        prev_n_mask = -1
        i = 0
        if n_iterations > 0:
            target_img.data = self.operation.subtract(image_data, bkg_map)
            for i in range(n_iterations):
                self.print(f"Iterative background estimation {i+1}/{n_iterations}...", verbose)
                previous_mask = target_mask.data
                mask_sigma_iter = max(3, mask_sigma - i * 0.5)
                target_mask = self.mask_sources(target_img = target_img,
                                                target_mask = target_mask,
                                                sigma = mask_sigma_iter,
                                                mask_radius_factor = mask_radius_factor,
                                                saturation_level = mask_saturation_level,
                                                save = False,
                                                verbose = verbose,
                                                visualize = visualize,
                                                save_fig = False)
                n_mask = np.sum(target_mask.data)
                if prev_n_mask > 0:
                    variation = abs(n_mask - prev_n_mask) / prev_n_mask
                    self.print(f" mask variation → {variation:.4f}", verbose)
                    if variation < 0.01:
                        self.print(f"Converged after {i+1} iterations (mask variation < 1%)", verbose)
                        break

                mask_to_use = target_mask.data.astype(bool)
                bkg = sep.Background(target_img.data, 
                                     mask=mask_to_use, 
                                     bw=box_size, 
                                     bh=box_size,
                                     fw=filter_size, 
                                     fh=filter_size)
                bkg_map = bkg.back()
                if visualize or save_fig:
                    save_path = None
                    if save_fig:
                        save_path = str(target_mask.savepath.savepath) + f'.iter_{i}.png'
                        
                print(save_path)
                self._visualize(target_img = target_img,
                                mask_data = previous_mask ,
                                bkg_map = mask_to_use, 
                                save_path = save_path,
                                subtitles = ['Target Image', 'Previous Mask', 'New Mask'],
                                show = visualize)
                #target_img.data -= bkg_map
                #target_img.data = self.operation.subtract(target_img.data, bkg_map)

                prev_n_mask = n_mask
                
            target_img.data = image_data # Restore the original image data
            bkg = sep.Background(target_img.data, 
                                 mask=mask_to_use, 
                                 bw=box_size, 
                                 bh=box_size,
                                 fw=filter_size, 
                                 fh=filter_size)
            bkg_map = bkg.back()


        target_bkg = Background(target_img.savepath.bkgpath, load=False)
        target_bkg.data = bkg_map
        target_bkg.header = target_img.header
        # Update header of the background image
        update_header_kwargs = dict(
            TGTPATH = str(target_img.path),
            MASKPATH = str(target_mask.path) if target_mask is not None else None,
            BKGTYPE = 'SEP',
            BKGIS2D = True,
            BKGVALU = float(np.mean(bkg_map)),
            BKGSIG = float(np.std(bkg_map)),
            BKGITER = int(i+1),
            BKGBOX = int(box_size),
            BKGFILT = int(filter_size),
        )
        target_bkg.header.update(update_header_kwargs)
        
        # Update header of the target image
        update_header_kwargs_image = dict(
            BKGPATH = str(target_bkg.path) if target_bkg.is_exists else None,
            BKGTYPE = 'SEP'
        )
        target_img.header.update(update_header_kwargs_image)

        ## Update status          
        event_details_kwargs = dict(
            type = 'SEP', 
            box_size = box_size, 
            filter_size = filter_size, 
            n_iterations = i, 
            mask_sigma = mask_sigma, 
            mask_radius_factor = mask_radius_factor)
        target_bkg.add_status("background_sep", **event_details_kwargs)

        if save:
            target_bkg.write()
        
        if visualize or save_fig:
            save_path = None
            if save_fig:
                save_path = str(target_img.savepath.bkgpath) + '.png'
            self._visualize(target_img = target_img,
                            mask_data = mask_to_use ,
                            bkg_map = bkg_map, 
                            save_path = save_path,
                            show = visualize)
        
        return target_bkg, bkg

    def calculate_photutils(self,
                            # Input parameters
                            target_img: Union[ScienceImage, ReferenceImage],
                            target_mask: Optional[Mask] = None,
                            box_size: int = 128,
                            filter_size: int = 3,
                            bkg_estimator: str = 'sextractor', # 'mean', 'median', 'sextractor'
                            
                            # Iterative background estimation
                            n_iterations: int = 1,
                            mask_sigma: float = 3.0,
                            mask_radius_factor: float = 3,
                            mask_saturation_level: float = 50000,
                            
                            # Others
                            save: bool = True,
                            verbose: bool = True,
                            visualize: bool = True,
                            save_fig: bool = False,
                            **kwargs
                            ):        
        from photutils.background import Background2D, MedianBackground, MeanBackground, SExtractorBackground

        # Step 1: Load the image and mask
        mask_to_use = None
        if target_mask is None:
            target_mask = Mask(target_img.savepath.srcmaskpath, masktype = 'source', load=False)
        else:
            mask_to_use = target_mask.data.astype(bool)
            self.print("External mask is loaded.", verbose)
            
        image_data = target_img.data.astype(np.float32).copy()

        # Step 2: Background estimation
        # Set the background estimation methods
        bkg_estimator_dict = {
            'mean': MeanBackground(),
            'median': MedianBackground(),
            'sextractor': SExtractorBackground()
        }
        if bkg_estimator.lower() not in bkg_estimator_dict:
            raise ValueError(f"Invalid background estimator '{bkg_estimator}'. Choose from 'mean', 'median', 'sextractor'.")
        bkgestimator = bkg_estimator_dict[bkg_estimator.lower()]

        self.print('Estimating 2D background...', verbose)
        bkg = Background2D(image_data, 
                           box_size = (box_size, box_size), 
                           mask=mask_to_use,
                           filter_size=(filter_size, filter_size),
                           sigma_clip=SigmaClip(sigma=3, maxiters = 10),
                           bkg_estimator=bkgestimator)
        bkg_map = bkg.background

        # Step 3: Iterative background estimation
        prev_n_mask = -1
        i = 0
        if n_iterations > 0:
            target_img.data = self.operation.subtract(image_data, bkg_map)
            self.print('Start iterative background estimation...', verbose)
            for i in range(n_iterations):
                self.print(f"Iterative background estimation {i+1}/{n_iterations}...", verbose)
                target_mask = self.mask_sources(target_img = target_img,
                                                target_mask = target_mask,
                                                sigma = mask_sigma,
                                                mask_radius_factor = mask_radius_factor,
                                                saturation_level = mask_saturation_level,
                                                save = False,
                                                verbose = verbose,
                                                visualize = visualize,
                                                save_fig = False)
                n_mask = np.sum(target_mask.data)
                if prev_n_mask > 0:
                    variation = abs(n_mask - prev_n_mask) / prev_n_mask
                    self.print(f" mask variation → {variation:.4f}", verbose)
                    if variation < 0.01:
                        self.print(f"Converged after {i+1} iterations (mask variation < 1%)", verbose)
                        break
                
            
                mask_to_use = target_mask.data.astype(bool)
                bkg = Background2D(target_img.data, 
                                box_size = (box_size, box_size), 
                                mask=mask_to_use,
                                filter_size=(filter_size, filter_size),
                                sigma_clip=SigmaClip(sigma=3, maxiters = 10),
                                bkg_estimator=bkgestimator)
                bkg_map = bkg.background
                if visualize:
                    self._visualize(target_img = target_img,
                                    mask_data = mask_to_use ,
                                    bkg_map = bkg_map, 
                                    save_path = None,
                                    show = visualize)

                #target_img.data -= bkg_map
                target_img.data = self.operation.subtract(target_img.data, bkg_map)
                prev_n_mask = n_mask
                
            target_img.data = image_data # Restore the original image data
            bkg = Background2D(image_data, 
                            box_size = (box_size, box_size), 
                            mask=mask_to_use,
                            filter_size=(filter_size, filter_size),
                            sigma_clip=SigmaClip(sigma=3, maxiters = 10),
                            bkg_estimator=bkgestimator)
            bkg_map = bkg.background
            
        target_bkg = Background(target_img.savepath.bkgpath, load=False)
        target_bkg.data = bkg_map
        target_bkg.header = target_img.header

        # Update header/status 
        update_header_kwargs = dict(
            TGTPATH = str(target_img.path),
            MASKPATH = str(target_mask.path) if target_mask is not None else None,
            BKGTYPE = 'Photutils',
            BKGIS2D = True,
            BKGVALU = float(np.mean(bkg_map)),
            BKGSIG = float(np.std(bkg_map)),
            BKGITER = int(i+1),
            BKGBOX = int(box_size),
            BKGFILT = int(filter_size),
        )
        target_bkg.header.update(update_header_kwargs)
        
        # Update header of the target image
        update_header_kwargs_image = dict(
            BKGPATH = str(target_bkg.path) if target_bkg.is_exists else None,
            BKGTYPE = 'Photutils'
        )
        target_img.header.update(update_header_kwargs_image)

        ## Update status          
        event_details_kwargs = dict(
            type = 'Photutils', 
            box_size = box_size, 
            filter_size = filter_size, 
            n_iterations = i)
        target_bkg.add_status("background_sep", **event_details_kwargs)

        if save:
            target_bkg.write()
        
        if save_fig or visualize:
            save_path = None
            if save_fig:
                save_path = str(target_img.savepath.bkgpath) + '.png'
            self._visualize(target_img = target_img, 
                            mask_data = mask_to_use , 
                            bkg_map = bkg_map, 
                            save_path = save_path,
                            show = visualize)
        
        return target_bkg, bkg

    def subtract_background(self, 
                            target_img: Union[ScienceImage, ReferenceImage],
                            target_bkg: Background,
                            
                            # Other parameters
                            save: bool = True,
                            overwrite: bool = False,
                            visualize: bool = True,
                            save_fig: bool = False,
                            **kwargs):
        
        # Step 1: Load the image and mask
        #target_img = target_img.copy()
        image_data = target_img.data
        image_data = image_data.astype(image_data.dtype.newbyteorder("="))
        image_header = target_img.header.copy()
        bkg_data = target_bkg.data
        bkg_data = bkg_data.astype(bkg_data.dtype.newbyteorder("="))
        
        # Step 2: Subtract the background
        if not overwrite:
            new_path = target_img.savepath.savepath.parent / Path('subbkg_' + target_img.savepath.savepath.name)
            target_img = type(target_img)(path = new_path, telinfo = target_img.telinfo, status = target_img.status, load = False)
            target_img.header = image_header
        target_img.data = image_data - bkg_data
        bkg_value = np.mean(bkg_data)
        # Step 3: Update the header
        # Update backgroundimg info
        target_img.header.update(target_bkg.info.to_dict())
        update_header_kwargs = dict(
            BKGPATH = str(target_bkg.path) if target_bkg.is_exists else None, 
            BKGVALU = 0.0,  # Background value is set to 0 after subtraction
        )
        # Update subbkg info
        target_img.header.update(update_header_kwargs)
        # Update target_img status
        target_img.update_status('BKGSUB')
        
        # Step 4: Save the image
        if save:
            target_img.write()
        
        if visualize or save_fig:
            save_path = None
            if save_fig:
                save_path = str(target_img.savepath.savepath) + '.subbkg.png'
            self._visualize(
                target_img = target_img,
                mask_data = None ,
                bkg_map = bkg_data, 
                save_path = save_path,
                show = visualize)
        
        return target_img
    
    def _visualize(self, 
                   target_img: Union[ScienceImage, ReferenceImage],
                   mask_data: Optional[np.ndarray] = None,
                   bkg_map: Optional[np.ndarray] = None,
                   subtitles: Optional[list] = None,
                   save_path: str = None,
                   show: bool = False):
        """
        Visualize available data: image, mask, and/or background map.
        """
        from astropy.visualization import ZScaleInterval
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt
        import numpy as np

        interval = ZScaleInterval()

        def downsample(data, factor=4):
            return data[::factor, ::factor]

        panels = []
        default_titles = []

        image_data = target_img.data
        image_data_small = downsample(image_data)
        vmin, vmax = interval.get_limits(image_data_small)
        panels.append((image_data_small, dict(cmap='Greys_r', vmin=vmin, vmax=vmax)))
        default_titles.append("Target Image")

        if mask_data is not None:
            mask_data_small = downsample(mask_data)
            panels.append((mask_data_small, dict(cmap='Greys_r', vmin=0, vmax=1)))
            default_titles.append("Mask")

        if bkg_map is not None:
            bkg_map_small = downsample(bkg_map)
            vmin, vmax = interval.get_limits(bkg_map_small)
            panels.append((bkg_map_small, dict(cmap='Greys_r', vmin=vmin, vmax=vmax)))
            default_titles.append("2D Background")

        n = len(panels)
        if n == 0:
            print("Nothing to visualize.")
            return

        if subtitles is None or len(subtitles) != n:
            subtitles = default_titles

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]

        for i, (data, imshow_kwargs) in enumerate(panels):
            ax = axes[i]
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(data, origin='lower', **imshow_kwargs)
            ax.set_title(subtitles[i])
            fig.colorbar(im, cax=cax, orientation='vertical')
            
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        
        plt.close(fig)
