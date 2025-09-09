#%%
import inspect
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.stats import SigmaClip
from skimage.draw import disk
import astroscrappy as cr
from photutils.aperture import CircularAperture
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog

from ezphot.imageobjects import Mask, ScienceImage, ReferenceImage, CalibrationImage
from ezphot.helper import Helper

#%%
class MaskGenerator():
    """
    Method class to generate masks for astronomical images.
    
    This class provides methods 
    
    1. Generation of invalid pixel mask with NaN and large connected regions of zero value.
    
    2. Generation of source mask with SExtractor-like detection.
    
    3. Generation of circular mask for given position and radius.
    
    4. Generation of cosmic ray mask with Astroscrappy.
    """
    
    def __init__(self):
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
     
    def mask_invalidpixel(self,
                         # Input parameters
                          target_img: Union[ScienceImage, ReferenceImage, CalibrationImage],
                          threshold_invalid_connection: int = 100000,
                          # Others
                          save: bool = False,
                          verbose: bool = True,
                          visualize: bool = True,
                          save_fig: bool = False):
        """
        Generate invalid pixel mask.
        
        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage or CalibrationImage
            The target image to generate the invalid pixel mask from.
        threshold_invalid_connection : int, optional
            The threshold for the invalid pixel mask generation.
        save : bool, optional
            Whether to save the invalid pixel mask.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the invalid pixel mask.
        save_fig : bool, optional
            Whether to save the invalid pixel mask as a figure.
            
        Returns
        -------
        target_mask : Mask
            The invalid pixel mask.
        """
        
        import numpy as np
        from scipy.ndimage import label

        image_data = target_img.data

        # Mask NaNs
        nan_mask = np.isnan(image_data)
        if np.any(nan_mask):
            self.helper.print(f"Masked {np.sum(nan_mask)} NaN pixels.", verbose)

        # Mask large connected regions of 0s
        zero_mask = (np.abs(image_data) == 0)
        labeled_array, num_features = label(zero_mask)
        self.helper.print(f"Found {num_features} connected regions", verbose)

        # Efficient region size filtering using np.bincount
        label_sizes = np.bincount(labeled_array.ravel())
        large_labels = np.where(label_sizes > threshold_invalid_connection)[0]
        large_labels = large_labels[large_labels != 0]  # exclude background

        large_zero_mask = np.isin(labeled_array, large_labels)
        self.helper.print(f"{len(large_labels)} regions larger than {threshold_invalid_connection} pixels", verbose)

        # Combine all invalid pixel masks
        invalidmask = nan_mask | large_zero_mask

        target_mask = Mask(target_img.savepath.invalidmaskpath, masktype='invalid', load=False)
        target_mask.data = invalidmask
        target_mask.header = target_img.header

        # Update header/status
        update_header_kwargs = dict(
            TGTPATH=str(target_img.path),
            MASKTYPE='InvalidPixel'
        )
        target_mask.header.update(update_header_kwargs)

        event_details = dict(
            nan_masked=str(np.sum(nan_mask)),
            num_zero_regions=str(num_features),
            threshold_invalid_connection=str(threshold_invalid_connection),
            zero_masked=str(np.sum(large_zero_mask))
        )
        target_mask.add_status("invalid_mask", **event_details)

        if save:
            target_mask.write(verbose = verbose)

        # Visualize the mask
        if visualize or save_fig:
            save_path = None
            if save_fig:
                save_path = str(target_mask.savepath.savepath) + '.png'
            self._visualize(
                target_img=target_img,
                final_mask=target_mask,
                previous_mask=None,
                save_path=save_path,
                show=visualize
            )

        return target_mask

    def mask_sources(self, 
                     # Input parameters
                     target_img: Union[ScienceImage, ReferenceImage, CalibrationImage],
                     target_mask: Optional[Mask] = None,
                     sigma: float = 5.0, 
                     mask_radius_factor: float = 3,
                     saturation_level: float = 50000,
                     
                     # Others
                     save: bool = False,
                     verbose: bool = True,
                     visualize: bool = True,
                     save_fig: bool = False,
                     ): 
        """
        Generate source mask.
        
        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage or CalibrationImage
            The target image to generate the source mask from.
        target_mask : Mask, optional
            The mask to use for the source mask generation.
        sigma : float, optional
            The sigma for the source mask generation.
        mask_radius_factor : float, optional
            The mask radius factor for the source mask generation.
        saturation_level : float, optional
            The saturation level for the source mask generation.
        save : bool, optional
            Whether to save the source mask.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the source mask.
        save_fig : bool, optional
            Whether to save the source mask as a figure.
            
        Returns
        -------
        target_mask : Mask
            The source mask.
        """
        if target_mask is None:
            target_mask = Mask(target_img.savepath.srcmaskpath, masktype = 'source', load=False)
            if target_mask.is_exists:
                target_mask.remove(remove_main = True, remove_connected_files = True, skip_exts = [], verbose = False)
        else:
            self.helper.print("External mask is loaded.", verbose)
        self.helper.print(f"Masking source... [sigma = {sigma}, mask_radius_factor = {mask_radius_factor}]", verbose)
        npixels = self.helper.load_config(target_img.config['SEX_CONFIG'])['DETECT_MINAREA']
        image_data, image_header = target_img.data, target_img.header
        sigma_clip = SigmaClip(sigma=sigma)
        threshold = detect_threshold(data = image_data, nsigma=sigma/np.sqrt(npixels), mask = target_mask.data, sigma_clip=sigma_clip)
        segment_img = detect_sources(image_data, threshold, npixels=npixels, mask = target_mask.data) 
        
        if segment_img:
            S = SourceCatalog(image_data, segment_img)
            props = S.to_table()
            new_mask = np.zeros(image_data.shape, dtype=bool)
            
            # Split props into saturated and non-saturated sources
            non_sat_rows = props[props['max_value'] <= saturation_level]
            sat_rows = props[props['max_value'] > saturation_level]

            self.helper.print(f"{len(non_sat_rows)} non-saturated, {len(sat_rows)} saturated sources", verbose)

            # Mask non-saturated sources (scaled by area)
            for row in non_sat_rows:
                area = row['area'].value
                y, x = row['ycentroid'], row['xcentroid']
                radius = mask_radius_factor * np.sqrt(area / np.pi)
                rr, cc = disk((y, x), radius, shape=image_data.shape)
                new_mask[rr, cc] = True

            # Mask saturated sources (fixed large radius)
            for row in sat_rows:
                area = row['area'].value
                y, x = row['ycentroid'], row['xcentroid']
                saturated_radius = mask_radius_factor * 2 * np.sqrt(area / np.pi)
                rr, cc = disk((y, x), saturated_radius, shape=image_data.shape)
                new_mask[rr, cc] = True

            mask_previous = target_mask.data
            target_mask.combine_mask(new_mask, 'or')
            target_mask.header = target_img.header
            
            # Update header/status
            update_header_kwargs = dict(
                TGTPATH = str(target_img.path),
                )
            
            current_masktype = target_mask.info.MASKTYPE
            if 'MASKTYPE' not in target_mask.header:
                update_header_kwargs['MASKTYPE'] = "Source"
            else:
                if "Source" not in current_masktype:
                    update_header_kwargs['MASKTYPE'] = f"{current_masktype},Source"
            
            ## Update attempt
            if 'MASKATMP' not in target_mask.info.to_dict():
                update_header_kwargs['MASKATMP'] = 1
            else:
                update_header_kwargs['MASKATMP'] = int(target_mask.info.MASKATMP) + 1
            target_mask.header.update(update_header_kwargs)
                
            ## Update status          
            event_details = dict(sigma = sigma, mask_radius_factor = mask_radius_factor, num_mask = segment_img.nlabels)
            target_mask.add_status("source_mask", **event_details)
            self.helper.print(f"{segment_img.nlabels} sources masked.", verbose)
            
            # Save mask
            if save:
                target_mask.write(verbose = verbose)

            # Visualize
            if visualize or save_fig:
                save_path = None
                if save_fig:
                    save_path = str(target_mask.savepath.savepath) + '.png'

                self._visualize(
                    target_img=target_img,
                    final_mask=target_mask,
                    previous_mask=mask_previous,
                    save_path=save_path,
                    show=visualize
                )
        else:
            print("No sources detected to mask.")
        return target_mask

    def mask_circle(self,
                    # Input parameters
                    target_img: Union[ScienceImage, ReferenceImage, CalibrationImage],
                    target_mask: Optional[Mask] = None,
                    mask_type: str = 'source',
                    x_position: float = None,
                    y_position: float = None,
                    radius_arcsec: float = None,
                    unit: str = 'coord',
                    
                    # Others
                    save: bool = False,
                    verbose: bool = True,
                    visualize: bool = True,
                    save_fig: bool = False,
                    ):
        """
        Add a circular mask to the mask image using photutils CircularAperture.
        
        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage or CalibrationImage
            The target image to generate the circular mask from.
        target_mask : Mask, optional
            The mask to use for the circular mask generation.
        mask_type : str, optional
            The type of mask to generate. ['invalid', 'source', 'cosmicray', 'badpixel', 'subtraction']
        x_position : float, optional
            The x position of the circular mask.
        y_position : float, optional
            The y position of the circular mask.
        radius_arcsec : float, optional
            The radius of the circular mask in arcseconds.
        unit : str, optional
            The unit of the radius. ['coord', 'pixel']
        save : bool, optional
            Whether to save the circular mask.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the circular mask.
        save_fig : bool, optional
            Whether to save the circular mask as a figure.
            
        Returns
        -------
        target_mask : Mask
            The circular mask.
        """
        if target_mask is None:
            if mask_type == 'invalid':
                target_mask = Mask(target_img.savepath.invalidmaskpath, masktype = mask_type, load=False)
            elif mask_type == 'source':
                target_mask = Mask(target_img.savepath.srcmaskpath, masktype = mask_type, load=False)
            elif mask_type == 'cosmicray':
                target_mask = Mask(target_img.savepath.crmaskpath, masktype = mask_type, load=False)
            elif mask_type == 'badpixel':
                target_mask = Mask(target_img.savepath.bpmaskpath, masktype = mask_type, load=False)
            elif mask_type == 'subtraction':
                target_mask = Mask(target_img.savepath.submaskpath, masktype = mask_type, load=False)
            else:
                self.helper.print(f"Unknown mask type: {mask_type}. Using 'invalid' as default.", verbose)
            if target_mask.is_exists:
                target_mask.remove(remove_main = True, remove_connected_files = True, skip_exts = [], verbose = False)
        else:
            self.helper.print("External mask is loaded.", verbose)

        if unit == 'coord':
            if target_img.header is None:
                raise ValueError("Header is required for RA/Dec conversion.")
            w = WCS(target_img.header)
            x_position_pixel, y_position_pixel = w.wcs_world2pix(x_position, y_position, 0)
            pixel_scales_deg = proj_plane_pixel_scales(w)  # [dy, dx] in deg/pixel
            pixel_scale_arcsec = np.mean(pixel_scales_deg) * 3600.0  # arcsec/pixel
            radius_pixel = radius_arcsec / pixel_scale_arcsec  # arcsec ? pixel
        else:
            x_position_pixel, y_position_pixel = x_position, y_position
            radius_pixel = radius_arcsec

        shape = target_img.data.shape
        aperture = CircularAperture((x_position_pixel, y_position_pixel), r=radius_pixel)
        new_mask = aperture.to_mask(method='center').to_image(shape)      

        mask_previous = target_mask.data
        target_mask.combine_mask(new_mask, 'or')
        target_mask.header = target_img.header  
        
        # Update header/status
        update_header_kwargs = dict(
            TGTPATH = str(target_img.path),
            )
        
        current_masktype = target_mask.info.MASKTYPE
        if 'MASKTYPE' not in target_mask.header:
            update_header_kwargs['MASKTYPE'] = "Aperture"
        else:
            if "Aperture" not in current_masktype:
                update_header_kwargs['MASKTYPE'] = f"{current_masktype},Aperture"
        
        ## Update attempt
        if 'MASKATMP' not in target_mask.info.to_dict():
            update_header_kwargs['MASKATMP'] = 1
        else:
            update_header_kwargs['MASKATMP'] = int(target_mask.info.MASKATMP) + 1
        target_mask.header.update(update_header_kwargs)
                    
        ## Update status
        event_details = dict(x=x_position, y=y_position, radius_arcsec=radius_arcsec, unit=unit)
        target_mask.add_status("circular_mask", **event_details)
        self.helper.print(f"Added circular mask at ({x_position:.2f}, {y_position:.2f}) with radius {radius_arcsec}arcsec", verbose)
        
        # Save mask
        if save:
            target_mask.write(verbose = verbose)
        
        # Visualize
        if visualize or save_fig:
            save_path = None
            if save_fig:
                save_path = str(target_mask.savepath.savepath) + '.png'

            self._visualize(
                target_img=target_img,
                final_mask=target_mask,
                previous_mask=mask_previous,
                save_path=save_path,
                show=visualize
            )
        return target_mask
    
    def mask_cosmicray(self,
                       # Input parameters
                       target_img: Union[ScienceImage, ReferenceImage, CalibrationImage],
                       target_mask: Optional[Mask] = None,
                       gain: float = None,
                       readnoise: float = None,
                       sigclip: float = 6,
                       sigfrac: float = 0.5,
                       objlim: float = 5.0,
                       niter: int = 4,
                       cleantype: str = 'medmask',
                       fsmode: str = 'median',
                       psffwhm: float = None,
                       saturation_level: float = 30000,
                       
                       # Others
                       save: bool = False,
                       verbose: bool = True,
                       visualize: bool = True,
                       save_fig: bool = False,
                       ):
        """
        Generate cosmic ray mask.
        
        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage or CalibrationImage
            The target image to generate the cosmic ray mask from.
        target_mask : Mask, optional
            The mask to use for the cosmic ray mask generation.
        gain : float, optional
            The gain of the image.
        readnoise : float, optional
            The readnoise of the image.
        sigclip : float, optional
            The sigma clip for the cosmic ray mask generation.
        sigfrac : float, optional
            The sigma fraction for the cosmic ray mask generation.
        objlim : float, optional
            The object limit for the cosmic ray mask generation.
        niter : int, optional
            The number of iterations for the cosmic ray mask generation.
        cleantype : str, optional
            The clean type for the cosmic ray mask generation.
        fsmode : str, optional
            The fsmode for the cosmic ray mask generation.
        psffwhm : float, optional
            The psf fwhm for the cosmic ray mask generation.
        saturation_level : float, optional
            The saturation level for the cosmic ray mask generation.
        save : bool, optional
            Whether to save the cosmic ray mask.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the cosmic ray mask.
        save_fig : bool, optional
            Whether to save the cosmic ray mask as a figure.
            
        Returns
        -------
        target_mask : Mask
            The cosmic ray mask.
        """

        # Perform cosmic ray detection and cleaning
        if target_mask is None:
            target_mask = Mask(target_img.savepath.crmaskpath, masktype = 'cosmicray', load=False)
            if target_mask.is_exists:
                target_mask.remove(remove_main = True, remove_connected_files = True, skip_exts = [], verbose = False)
        else:
            self.helper.print("External mask is loaded.", verbose)
        # Load information from target_img
        if gain is None:
            gain = target_img.egain
        if readnoise is None:
            readnoise = target_img.telinfo['readnoise']
        if (gain is None) or (readnoise is None):
            raise ValueError("Gain and readnoise are required for cosmic ray detection.")
        if psffwhm is None:
            psffwhm = 2 / target_img.telinfo['pixelscale']
        
        self.helper.print(f'Detecting cosmic ray... [sigma = {sigclip}, n_iter = {niter}, mode = {fsmode}]', verbose)
        new_mask, clean_image = cr.detect_cosmics(
            target_img.data, gain=gain, readnoise=readnoise, 
            sigclip=sigclip, sigfrac=sigfrac, 
            objlim=objlim, niter=niter, 
            cleantype=cleantype, fsmode=fsmode, 
            psffwhm = psffwhm, verbose=verbose,
            satlevel = saturation_level)
        
        mask_previous = target_mask.data
        target_mask.combine_mask(new_mask, 'or')
        target_mask.header = target_img.header        
        
        # Update header/status
        update_header_kwargs = dict(
            TGTPATH = str(target_img.path),
            )
        
        current_masktype = target_mask.info.MASKTYPE
        if 'MASKTYPE' not in target_mask.header:
            update_header_kwargs['MASKTYPE'] = "CosmicRay"
        else:
            if "CosmicRay" not in current_masktype:
                update_header_kwargs['MASKTYPE'] = f"{current_masktype},CosmicRay"
        
        ## Update attempt
        if 'MASKATMP' not in target_mask.info.to_dict():
            update_header_kwargs['MASKATMP'] = 1
        else:
            update_header_kwargs['MASKATMP'] = int(target_mask.info.MASKATMP) + 1
        target_mask.header.update(update_header_kwargs)
        
        ## Update status
        event_details = dict(gain = gain, readnoise = readnoise, sigclip = sigclip, sigfrac = sigfrac, objlim = objlim, niter = niter, cleantype = cleantype, fsmode = fsmode)
        target_mask.add_status("cr_mask", **event_details)
        self.helper.print(f"{new_mask.sum()} cosmic rays masked.", verbose)
        
        # Save mask
        if save:
            target_mask.write(verbose = verbose)
        
        # Visualize
        if visualize or save_fig:
            save_path = None
            if save_fig:
                save_path = str(target_mask.savepath.savepath) + '.png'

            self._visualize(
                target_img=target_img,
                final_mask=target_mask,
                previous_mask=mask_previous,
                save_path=save_path,
                show=visualize
            )
        return target_mask
        
    def _visualize(self,
                   target_img: Union[ScienceImage, ReferenceImage, CalibrationImage],
                   final_mask: Optional[Mask],
                   previous_mask: np.ndarray = None,
                   save_path: str = None,
                   show: bool = False):
        """
        Visualize the image and mask.
        """
        from astropy.visualization import ZScaleInterval
        
        interval = ZScaleInterval()
        
        def downsample(data, factor=4):
            return data[::factor, ::factor]
        
        image_data = target_img.data
        image_data_small = downsample(image_data)
        bkg_value = np.mean(image_data_small)
        bkg_rms = np.std(image_data_small)
        if previous_mask is not None:
            previous_mask_small = downsample(previous_mask)
        new_mask = final_mask.data
        new_mask_small = downsample(new_mask)
        len_figure = 1 + sum(mask is not None for mask in [previous_mask, new_mask])
        # Visualization of the image
        
        fig, ax = plt.subplots(1, len_figure, figsize=(6 * len_figure, 6))
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        vmin, vmax = interval.get_limits(image_data_small)
        im0 = ax[0].imshow(image_data_small, origin='lower', cmap='Greys_r', vmin=vmin, vmax=vmax)
        ax[0].set_title('Original Image')
        fig.colorbar(im0, cax=cax, orientation='vertical')
        
        if previous_mask is not None:
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im1 = ax[1].imshow(previous_mask_small, origin='lower', cmap='Greys_r', vmin=0, vmax=1)
            ax[1].set_title('Previous Mask')
            fig.colorbar(im1, cax=cax, orientation='vertical')
        
        divider = make_axes_locatable(ax[-1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im2 = ax[-1].imshow(new_mask_small, origin='lower', cmap='Greys_r', vmin=0, vmax=1)
        ax[-1].set_title('New Mask')
        fig.colorbar(im2, cax=cax, orientation='vertical')
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        plt.close(fig)

