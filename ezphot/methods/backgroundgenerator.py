
#%%
import inspect
from typing import Union, Optional
from pathlib import Path

import numpy as np
import sep
import cv2
from astropy.stats import SigmaClip

from ezphot.imageobjects import Mask, Background, ScienceImage, ReferenceImage
from ezphot.helper import Helper

#%%
class BackgroundGenerator:
    """
    Method class to estimate and subtract the background of astronomical images using 
    SEP (Source Extractor-like) or photutils-based methods.

    This class provides methods 
    
    1. Calculation of 2D background map from SEP or photutils-based methods.
    
    2. Calculation of 1D (constant) background level from SEP or photutils-based methods.
    
    3. Subtraction of background from the target image.
    """

    def __init__(self):
        """
        Initialize the BackgroundGenerator class.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self.helper = Helper()

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
        print(f"Help for {self.__class__.__name__}\n{help_text}\nPublic methods:\n" + "\n".join(lines))

    def estimate_with_sep(self,
                          # Input parameters
                          target_img: Union[ScienceImage, ReferenceImage],
                          target_srcmask: Optional[Mask] = None,
                          target_ivpmask: Optional[Mask] = None,
                          is_2D_bkg: bool = True,
                          box_size: int = 32,
                          filter_size: int = 3,
  
                          # Others
                          save: bool = False,
                          verbose: bool = True,
                          visualize: bool = True,
                          save_fig: bool = False,
                          ):
        """
        Estimate the background of an astronomical image using SEP.
        
        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage
            The target image to estimate the background from.
        target_srcmask : Mask, optional
            The source mask to use for the background estimation.
        target_ivpmask : Mask, optional
            The invalid pixel mask to use for the background estimation.
        is_2D_bkg : bool, optional
            Whether to estimate a 2D background.
        box_size : int, optional
            The size of the box for the background estimation.
        filter_size : int, optional
            The size of the filter for the background estimation.
        save : bool, optional
            Whether to save the background image.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the background image.
        save_fig : bool, optional
            Whether to save the background image as a figure.
            
        Returns
        -------
        target_bkg : Background
            The estimated background image instance from ezphot.
        bkg : sep.Background
            The background image instance from sep.
        """
        
        # Step 1: Load the image and mask
        mask_to_use = None
        if target_srcmask is None:
            pass
        else:
            mask_to_use = target_srcmask.data.astype(bool)
            self.helper.print("External mask is loaded.", verbose)
            
        image_data = target_img.data
        # If image_data is uint16, convert to float32
        if image_data.dtype == np.uint16:
            image_data = image_data.astype(np.float32)

        # Create a mask of NaN values
        # invalid_mask = ~np.isfinite(image_data)

        # if np.any(invalid_mask):
        #     mask = invalid_mask.astype(np.uint8)
        #     image_filled = np.nan_to_num(image_data, nan=0, posinf=0.0, neginf=0.0).astype(np.float32)
        #     # Inpaint using the mask
        #     image_data = cv2.inpaint(image_filled, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        if target_ivpmask is not None:
            image_data[target_ivpmask.data] = np.nan

        # Step 2: Background estimation
        image_data = self.helper.to_native(image_data)
        
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
        
        if target_ivpmask is not None:
            bkg_map[target_ivpmask.data] = np.nan
        
        target_bkg = Background(target_img.savepath.bkgpath, load=False)
        target_bkg.data = bkg_map
        target_bkg.header = target_img.header
        
        # Update header of the background image
        update_header_kwargs = dict(
            TGTPATH = str(target_img.path),
            MASKPATH = str(target_srcmask.path) if target_srcmask is not None else None,
            BKGTYPE = 'SEP',
            BKGIS2D = True,
            BKGVALU = float(np.nanmean(bkg_map)),
            BKGSIG = float(np.nanstd(bkg_map)),
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
            filter_size = filter_size)
        target_bkg.add_status("background_sep", **event_details_kwargs)

        if save:
            target_bkg.write(verbose = verbose)
        
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

    def estimate_with_photutils(self,
                                # Input parameters
                                target_img: Union[ScienceImage, ReferenceImage],
                                target_mask: Optional[Mask] = None,
                                is_2D_bkg: bool = True,
                                box_size: int = 128,
                                filter_size: int = 3,
                                bkg_estimator: str = 'sextractor', # 'mean', 'median', 'sextractor'
                                
                                # Others
                                save: bool = False,
                                verbose: bool = True,
                                visualize: bool = True,
                                save_fig: bool = False):     
        """
        Estimate the background of an astronomical image using photutils.
        OBSOLETE: Use scienceimage.calculate_background() instead.
        
        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage
            The target image to estimate the background from.
        target_mask : Mask, optional
            The mask to use for the background estimation.
        is_2D_bkg : bool, optional
            Whether to estimate a 2D background.
        box_size : int, optional
            The size of the box for the background estimation.
        filter_size : int, optional
            The size of the filter for the background estimation.
        bkg_estimator : str, optional
            The background estimator to use.
        save : bool, optional
            Whether to save the background image.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the background image.
        save_fig : bool, optional
            Whether to save the background image as a figure.
            
        Returns
        -------
        target_bkg : Background
            The estimated background image instance from ezphot.
        bkg : photutils.background.Background2D
            The background image instance from photutils.
        """
        from photutils.background import Background2D, MedianBackground, MeanBackground, SExtractorBackground

        # Step 1: Load the image and mask
        mask_to_use = None
        if target_mask is None:
            pass
        else:
            mask_to_use = target_mask.data.astype(bool)
            self.helper.print("External mask is loaded.", verbose)
            
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

        self.helper.print('Estimating 2D background...', verbose)
        bkg = Background2D(image_data, 
                           box_size = (box_size, box_size), 
                           mask=mask_to_use,
                           filter_size=(filter_size, filter_size),
                           sigma_clip=SigmaClip(sigma=3, maxiters = 10),
                           bkg_estimator=bkgestimator)
        if is_2D_bkg:
            bkg_map = bkg.background
        else:
            bkg_val = np.mean(bkg.background)
            bkg_map = np.full_like(image_data, bkg_val, dtype=image_data.dtype)

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
            filter_size = filter_size)
        target_bkg.add_status("background_sep", **event_details_kwargs)

        if save:
            target_bkg.write(verbose = verbose)

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
                            overwrite: bool = False,
                            save: bool = False,
                            verbose: bool = True,
                            visualize: bool = True,
                            save_fig: bool = False):
        """
        Subtract the background from the target image.
        
        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage
            The target image to subtract the background from.
        target_bkg : ezphot.imageobjects.Background
            The background image to subtract from the target image.
        overwrite : bool, optional
            Whether to overwrite the target image.
        save : bool, optional
            Whether to save the subtracted image.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the subtracted image.
        save_fig : bool, optional
            Whether to save the subtracted image as a figure.
            
        Returns
        -------
        target_img : ScienceImage or ReferenceImage
            The subtracted image.
        """
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
            target_img.write(verbose = verbose)
        
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
            self.helper.print("Nothing to visualize.", True)
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

