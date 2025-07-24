#%%
from typing import Union, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog
from photutils.utils import circular_footprint
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np
from photutils.aperture import CircularAperture
import astroscrappy as cr
from skimage.draw import disk
from tqdm import tqdm

from tippy.image import Mask
from tippy.image import ScienceImage, ReferenceImage, CalibrationImage
from tippy.helper import Helper
#%%
class TIPMasking(Helper): ############## CHECKED ##############
    
    def __init__(self):
        super().__init__()
        
    def mask_invalidpixel(self,
                    target_img: Union[ScienceImage, ReferenceImage, CalibrationImage],
                    target_mask: Optional[Mask] = None,
                    threshold_invalid_connection: int = 100000,
                    save: bool = False,
                    verbose: bool = True,
                    visualize: bool = True,
                    save_fig: bool = False,
                    **kwargs):
        import numpy as np
        from scipy.ndimage import label

        if target_mask is None:
            target_mask = Mask(target_img.savepath.invalidmaskpath, masktype='invalid', load=False)
            target_mask.remove()
        else:
            self.print("External mask is loaded.", verbose)

        image_data = target_img.data

        # Mask NaNs
        nan_mask = np.isnan(image_data)
        if np.any(nan_mask):
            self.print(f"Masked {np.sum(nan_mask)} NaN pixels.", verbose)

        # Mask large connected regions of 0s
        zero_mask = (image_data == 0)
        labeled_array, num_features = label(zero_mask)
        self.print(f"Found {num_features} connected regions", verbose)

        # Efficient region size filtering using np.bincount
        label_sizes = np.bincount(labeled_array.ravel())
        large_labels = np.where(label_sizes > threshold_invalid_connection)[0]
        large_labels = large_labels[large_labels != 0]  # exclude background

        large_zero_mask = np.isin(labeled_array, large_labels)
        self.print(f"{len(large_labels)} regions larger than {threshold_invalid_connection} pixels", verbose)

        # Combine all invalid pixel masks
        invalidmask = nan_mask | large_zero_mask

        mask_previous = target_mask.data
        target_mask.combine_mask(invalidmask, 'or')
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
            target_mask.write()

        # Visualize the mask
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
                     **kwargs): 
        if target_mask is None:
            target_mask = Mask(target_img.savepath.srcmaskpath, masktype = 'source', load=False)
        else:
            self.print("External mask is loaded.", verbose)
        self.print(f"Masking source... [sigma = {sigma}, mask_radius_factor = {mask_radius_factor}]", verbose)
        npixels = self.load_config(target_img.config['SEX_CONFIG'])['DETECT_MINAREA']
        image_data, image_header = target_img.data, target_img.header
        sigma_clip = SigmaClip(sigma=sigma)
        threshold = detect_threshold(data = image_data, nsigma=sigma/np.sqrt(npixels), mask = target_mask.data, sigma_clip=sigma_clip)
        segment_img = detect_sources(image_data, threshold, npixels=npixels) 
        
        if segment_img:
            S = SourceCatalog(image_data, segment_img)
            props = S.to_table()
            new_mask = np.zeros(image_data.shape, dtype=bool)
            
            # Split props into saturated and non-saturated sources
            non_sat_rows = props[props['max_value'] <= saturation_level]
            sat_rows = props[props['max_value'] > saturation_level]

            self.print(f"{len(non_sat_rows)} non-saturated, {len(sat_rows)} saturated sources", verbose)

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
            self.print(f"{segment_img.nlabels} sources masked.", verbose)
            
            # Save mask
            if save:
                target_mask.write()
            
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
                    mask_type: str = 'invalid',
                    x_position: float = None,
                    y_position: float = None,
                    radius_arcsec: float = None,
                    unit: str = 'deg',
                    
                    # Others
                    save: bool = False,
                    verbose: bool = True,
                    visualize: bool = True,
                    save_fig: bool = False,
                    **kwargs):
        """
        Add a circular mask to the mask image using photutils CircularAperture.
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
                self.print(f"Unknown mask type: {mask_type}. Using 'invalid' as default.", verbose)
        else:
            self.print("External mask is loaded.", verbose)

        if unit == 'deg':
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
        self.print(f"Added circular mask at ({x_position:.2f}, {y_position:.2f}) with radius {radius_arcsec}arcsec", verbose)
        
        # Save mask
        if save:
            target_mask.write()
        
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
                       **kwargs
                       ):

        # Perform cosmic ray detection and cleaning
        if target_mask is None:
            target_mask = Mask(target_img.savepath.crmaskpath, masktype = 'cosmicray', load=False)
        else:
            self.print("External mask is loaded.", verbose)
        # Load information from target_img
        if gain is None:
            gain = target_img.egain
        if readnoise is None:
            readnoise = target_img.telinfo['readnoise']
        if (gain is None) or (readnoise is None):
            raise ValueError("Gain and readnoise are required for cosmic ray detection.")
        if psffwhm is None:
            psffwhm = 2 / target_img.telinfo['pixelscale']
        
        self.print(f'Detecting cosmic ray... [sigma = {sigclip}, n_iter = {niter}, mode = {fsmode}]', verbose)
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
        self.print(f"{new_mask.sum()} cosmic rays masked.", verbose)
        
        # Save mask
        if save:
            target_mask.write()
        
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


# %%
if __name__ == '__main__':
    path = Path('/home/hhchoi1022/data/refdata/main/7DT/7DT_C361K_HIGH_1x1/T22956/7DT14/m600/calib_7DT14_T22956_20250329_044114_m600_600.com.fits')
    S = ScienceImage(path=path, telinfo=Helper().get_telinfo('7DT', 'C361K', 'HIGH', 1))
    self = TIPMasking()
    
    target_img: Union[ScienceImage, ReferenceImage, CalibrationImage] = S
    target_mask = None
    sigma: float = 5.0
    mask_radius_factor: int = 3
    verbose: bool = True
    visualize: bool = True
    save: bool = True
    #M = self.mask_sources(S, save = save, mask_radius_factor = mask_radius_factor)
    #M = T.mask_circle(S, M, x_position = 1000, y_position = 1000, radius = 500, unit = 'pixel', save = save)
    #M = T.mask_cosmicray(S, save = save)
# %%
if __name__ == "__main__":
    from numba import jit

    mask_A = new_mask
    mask_B = new_mask

    
    @jit(nopython=False)
    def myfunc(mask_A, mask_B):
        return np.logical_or(mask_A, mask_B)
    
    start = time.time()
    mask_C = myfunc(mask_A, mask_B)
    print(time.time() - start)
    
    import time
    start = time.time()
    mask_C = np.logical_or(mask_A, mask_B)
    print(time.time() - start)

    from numba import njit

    @njit
    def mask_or(mask_A, mask_B):
        result = np.empty_like(mask_A)
        for i in range(mask_A.shape[0]):
            for j in range(mask_A.shape[1]):
                result[i, j] = mask_A[i, j] | mask_B[i, j]
        return result
    start = time.time()
    mask_C = mask_or(mask_A, mask_B)
    print(time.time() - start)
# %%
