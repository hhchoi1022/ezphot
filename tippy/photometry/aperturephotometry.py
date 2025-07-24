#%%
from typing import Union, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle
from astropy.table import Table, vstack
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel, convolve
from photutils.segmentation import detect_threshold, detect_sources, deblend_sources, SourceCatalog
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.units import Quantity
from photutils.background import Background2D, MedianBackground, MeanBackground, SExtractorBackground
from photutils.aperture import EllipticalAperture, EllipticalAnnulus
from photutils.utils import calc_total_error
from scipy.spatial import cKDTree
from tqdm import tqdm
from astropy.stats import SigmaClip
from scipy.ndimage import mean as ndi_mean
from matplotlib.patches import Ellipse, Rectangle, Circle
import os

import sep
from scipy.signal import fftconvolve
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel

from tippy.photometry import TIPBackground, TIPMasking, TIPErrormap
from tippy.helper import Helper
from tippy.image import ScienceImage, ReferenceImage, CalibrationImage
from tippy.image import Background, Mask, Errormap
from tippy.catalog.utils import *
from tippy.catalog import TIPCatalog
from astropy.modeling import models, fitting

class TIPAperturePhotometry(Helper):
    def __init__(self):
        super().__init__()
        self.background = TIPBackground()
        self.masking = TIPMasking()
        self.errormap = TIPErrormap()

    def sex_photometry(self,
                       # Input parameters
                       target_img: Union[ScienceImage, ReferenceImage, CalibrationImage], 
                       target_bkg: Optional[Background] = None, # If target_bkg is given, subtract background before sextractor
                       target_bkgrms: Optional[Errormap] = None, # It must be background error map
                       target_mask: Optional[Mask] = None, # For masking certain source (such as hot pixels)
                       sex_params: dict = None,
                       detection_sigma: float = 5,
                       aperture_diameter_arcsec: Union[float, list] = [5,7,10],
                       saturation_level: float = 60000,
                       kron_factor: float = 2.5,
                       
                       # Others
                       save: bool = True,
                       verbose: bool = True,
                       visualize: bool = True,
                       save_fig: bool = False,
                       **kwargs
                       ):
          
        if not isinstance(target_img, (ScienceImage, ReferenceImage)):
            raise ValueError('target_img must be a ScienceImage, ReferenceImage, or CalibrationImage.')
        
        # If target_bkg is given, subtract background before sextractor
        if not target_img.is_saved:
            target_img.write()
        
        # If sex_params is None, use default parameters
        if sex_params is None:
            sex_params = dict()
        
        #target_img = target_img.copy()
        img_path = target_img.savepath.savepath
        cat_path = target_img.savepath.catalogpath
        sexconfig_path = target_img.config['SEX_CONFIG']
        all_sexconfig = self.load_config(sexconfig_path)
        
        # Image
        remove_subbkg = False
        if target_bkg is not None:
            target_img_sub = self.background.subtract_background(
                target_img = target_img, 
                target_bkg = target_bkg,
                save = True,
                overwrite = False,
                visualize = visualize,
                save_fig = save_fig,
            )
            saturation_level -= target_bkg.info.BKGVALU
            img_path = target_img_sub.savepath.savepath
            sex_params['BACK_TYPE'] = 'MANUAL'
            remove_subbkg = True
        else:
            target_img_sub = target_img
            if 'BACK_TYPE' not in sex_params.keys():
                sex_params['BACK_TYPE'] = 'AUTO'
            
        # Background RMS
        remove_bkgrms = False
        if target_bkgrms is not None:
            if not target_bkgrms.is_saved:
                target_bkgrms.write()
                remove_bkgrms = True
            bkgrms_path = target_bkgrms.savepath.savepath
        else:
            bkgrms_path = None
        
        # Mask
        remove_mask = False
        if target_mask is not None:
            if not target_mask.is_saved:
                target_mask.write()
                remove_mask = True
            mask_path = target_mask.savepath.savepath
        else:
            mask_path = None
                    
        # First Sextractor run to estimate seeing if not provided
        if target_img.seeing is None:
            sex_params['DETECT_THRESH'] = 10
            result, catalog_first, global_bkgval, global_bkgrms = self.run_sextractor(
                target_path = img_path,
                sex_configfile = sexconfig_path,
                sex_params = sex_params,
                target_mask = mask_path,
                target_weight = None,
                weight_type = 'MAP_RMS',
                target_outpath = cat_path,
                return_result = True,
                verbose = verbose
            ) 
            if not result:
                raise RuntimeError('Source extractor is failed.')
            
            seeing_estimate = None
            if 'FLAGS' in catalog_first.colnames:
                rough_flags = (catalog_first['FLAGS'] == 0)
            if 'MAG_AUTO' in catalog_first.colnames:
                rough_flags &= (catalog_first['MAG_AUTO'] < 0)
                rough_flags &= (catalog_first['MAG_AUTO'] > -20)
            if 'FLUX_MAX' in catalog_first.colnames:
                rough_flags &= (catalog_first['FLUX_MAX'] < 60000)
            if 'ELONGATION' in catalog_first.colnames:
                rough_flags &= (catalog_first['ELONGATION'] < 1.3)
            if 'CLASS_STAR' in catalog_first.colnames:
                rough_flags &= (catalog_first['CLASS_STAR'] > 0.1)
            seeing_arcsec = catalog_first['FWHM_WORLD'] * 3600
            rough_flags &= (seeing_arcsec < 10)
            rough_flags &= (seeing_arcsec > 0.5)
            catalog_filtered = catalog_first[rough_flags]
            seeing_estimate = max(1.2, np.mean(catalog_filtered['FWHM_WORLD']) * 3600)
        else:
            seeing_estimate = target_img.seeing
        
        if "SEEING_FWHM" not in sex_params.keys():
            sex_params['SEEING_FWHM'] = '%.2f' %seeing_estimate            
        sex_params['PIXEL_SCALE'] = np.mean(target_img.pixelscale)
        if isinstance(aperture_diameter_arcsec, (float, int)):
            aperture_diameter_arcsec = [aperture_diameter_arcsec]
        aperture_diameter_pixel = ','.join(["%.2f"%(float(size / target_img.telinfo['pixelscale'])) for size in aperture_diameter_arcsec])
        sex_params['PHOT_APERTURES'] = aperture_diameter_pixel # This is aperture size in pixel
        sex_params['SATUR_LEVEL'] = saturation_level
        sex_params['PHOT_AUTOPARAMS'] = f"{kron_factor},3.5"
        sex_params['DETECT_THRESH'] = detection_sigma
        sex_params['ANALYSIS_THRESH'] = detection_sigma
        sex_params['DETECT_MINAREA'] = np.pi* (float(sex_params['SEEING_FWHM']) *0.8 / np.mean(target_img.pixelscale) / 2)**2
        
        for key, value in sex_params.items():
            all_sexconfig[key] = value
        
        sex_params['DETECT_THRESH'] = detection_sigma / np.sqrt(all_sexconfig['DETECT_MINAREA'])
        sex_params['ANALYSIS_THRESH'] = sex_params['DETECT_THRESH']
        
        # Second Sextractor run with the estimated parameters
        result, catalog, global_bkgval, global_bkgrms = self.run_sextractor(
            target_path = img_path,
            sex_configfile = sexconfig_path,
            sex_params = sex_params.copy(),
            target_mask = mask_path,
            target_weight = bkgrms_path,
            weight_type = 'MAP_RMS',
            target_outpath = cat_path,
            return_result = True,
            verbose = verbose
        ) 
        
        if not result:
            raise RuntimeError('Source extractor is failed.')
        
        # Modification of the catalog
        catalog['X_IMAGE'] = catalog['X_IMAGE'] -1 # 0-based index
        catalog['Y_IMAGE'] = catalog['Y_IMAGE'] -1 # 0-based index
        catalog['SKYSIG'] = global_bkgrms
        catalog['SKYVAL'] = global_bkgval
        catalog['DETECT_THRESH'] = detection_sigma

        # Kron aperture area
        a = catalog['KRON_RADIUS'] * catalog['A_IMAGE']
        b = catalog['KRON_RADIUS'] * catalog['B_IMAGE']
        catalog['NPIX_AUTO'] = np.pi * a * b

        # Circular aperture areas
        aperture_diameter_arcsec = np.atleast_1d(aperture_diameter_arcsec)
        pixelscale = np.mean(target_img.pixelscale)
        for i, ap_size_arcsec in enumerate(aperture_diameter_arcsec):
            radius_pixel = ap_size_arcsec / pixelscale / 2
            area_pixel = np.pi * (radius_pixel)**2
            colname = 'NPIX_APER' if i == 0 else f'NPIX_APER_{i}'
            catalog[colname] = np.full(len(catalog), area_pixel)

        target_catalog = TIPCatalog(path = cat_path, catalog_type = 'all', load = False) 
        target_catalog.data = catalog
        target_catalog.load_target_img(target_img = target_img)
        
        if save:
            target_catalog.write()
        else:
            target_catalog.remove()
        
        if visualize:
            save_path = None
            if save_fig:
                save_path = str(target_catalog.path) + '.png'
            self._visualize_objects(target_img=target_img, 
                                    objects=target_catalog.data, 
                                    size=1000, 
                                    save_path = save_path,
                                    show = visualize)
            
        for remove_trigger, remove_object in zip(
            [remove_subbkg, remove_bkgrms, remove_mask],
            [target_img_sub, target_bkgrms, target_mask]):
            if remove_trigger:
                remove_object.remove()
        
            
        return target_catalog
    
    def photutils_photometry(self,
                             # Input parameters
                             target_img: Union[ScienceImage, ReferenceImage, CalibrationImage], 
                             target_bkg: Optional[Background] = None,
                             target_bkgrms: Optional[Errormap] = None,
                             target_mask: Optional[Mask] = None,
                             detection_sigma: float = 1.5,
                             aperture_diameter_arcsec: Union[float, list] = [5,7,10],
                             kron_factor: float = 2.5,
                             minarea_pixels: int = 5,
                             calc_accurate_fwhm: bool = True,
                            
                             # Other options
                             save: bool = True,
                             verbose: bool = True,
                             visualize: bool = True,
                             save_fig: bool = False,
                             **kwargs):
        """
        Perform circular and Kron photometry using photutils.SourceCatalog.
        """
        if not isinstance(target_img, (ScienceImage, ReferenceImage, CalibrationImage)):
            raise ValueError("target_img must be a ScienceImage, ReferenceImage, or CalibrationImage.")
        #target_img = target_img.copy()

        bkgrms_map = None
        bkgrms = None
        mask_map = target_mask.data if target_mask is not None else None
        mask_map = self.to_native(mask_map)
        target_data = self.to_native(target_img.data)
        # Step 1: Background subtraction
        if target_bkg is not None:
            target_img_sub = self.background.subtract_background(
                target_img=target_img,
                target_bkg=target_bkg,
                save = False,
                overwrite=False,
                visualize = visualize,
                save_fig = save_fig,
            )
            target_data = self.to_native(target_img_sub.data)
            target_bkg_data = target_bkg.data
        else:
            target_bkg_data = None

        # Step 2: Set error map
        if target_bkgrms is not None:
            bkgrms = self.to_native(target_bkgrms.data)
        elif bkgrms_map is not None:
            bkgrms = self.to_native(bkgrms_map.data)
        else:
            # Use sigma-clipped std if no error map available
            bkgrms_map, _, _ = self.errormap.calculate_from_sourcemask(
                target_img=target_img,
                target_mask=None,
                mode='photutils',
                errormap_type='bkgrms',
                n_iterations = 0,
                save=False,
                visualize=visualize,
                save_fig = False,
                **kwargs
            )
            bkgrms = bkgrms_map.data
        
        error = calc_total_error(data=target_data, bkg_error=bkgrms, effective_gain=target_img.egain)

        # 3. Segmentation and deblending
        threshold = detection_sigma * bkgrms
        segm = detect_sources(target_data, threshold, npixels=minarea_pixels, mask=mask_map)
        if segm is None:
            return None
        segm = deblend_sources(target_data, segm, npixels=minarea_pixels, nlevels=32, contrast=0.005)
        
        # 4. SourceCatalog (deblended)
        cat = SourceCatalog(data=target_data, segment_img=segm, error=error, mask=mask_map, background = target_bkg_data, wcs = target_img.wcs, kron_params = (kron_factor, 1.0, 0))
        cat_tbl = cat.to_table()
        cat_tbl['kron_radius'] = cat.kron_radius * kron_factor
        cat_tbl['flux_radius'] = cat.fluxfrac_radius(0.5)
        cat_tbl['ellipticity'] = cat.ellipticity
        cat_tbl['fwhm_pixel'] = 2.3548 / 1.1774 * cat_tbl['flux_radius'] # From flux radius, gaussian approximation
        coords = cat_tbl['sky_centroid']
        cat_tbl['ra'] = coords.ra.value
        cat_tbl['dec'] = coords.dec.value

        # FWHM calculation
        if calc_accurate_fwhm:
            all_fwhm = []
            for source in tqdm(cat_tbl, desc = 'Calculating FWHM...'):
                x0, y0 = source['xcentroid'], source['ycentroid']
                stamp = self.img_extract_stamp(target_data, x0, y0, size=25)  # make your own stamp function
                y, x = np.indices(stamp.shape)
                
                model = models.Gaussian2D(amplitude=np.max(stamp), x_mean=12.5, y_mean=12.5, x_stddev=2, y_stddev=2)
                fitter = fitting.LevMarLSQFitter()
                fit_model = fitter(model, x, y, stamp)
                
                fwhm_x = 2.3548 * fit_model.x_stddev.value
                fwhm_y = 2.3548 * fit_model.y_stddev.value
                fwhm_avg = np.sqrt(fwhm_x * fwhm_y)
                all_fwhm.append(fwhm_avg)
            cat_tbl['fwhm_pixel'] = all_fwhm
        
        rename_map = {
            'label': 'NUMBER',
            'xcentroid': 'X_IMAGE',
            'ycentroid': 'Y_IMAGE',
            'ra': 'X_WORLD',
            'dec': 'Y_WORLD',
            'bbox_xmin': 'XMIN_IMAGE',
            'bbox_xmax': 'XMAX_IMAGE',
            'bbox_ymin': 'YMIN_IMAGE',
            'bbox_ymax': 'YMAX_IMAGE',
            'area': 'ISOAREA_IMAGE',
            'semimajor_sigma': 'A_IMAGE',
            'semiminor_sigma': 'B_IMAGE',
            'orientation': 'THETA_IMAGE', 
            'eccentricity': 'ECCENTRICITY',
            'ellipticity': 'ELLIPTRICITY',
            'min_value': 'FLUX_MIN',
            'max_value': 'FLUX_MAX',
            'segment_flux': 'FLUX_ISO',
            'segment_fluxerr': 'FLUXERR_ISO',
            'kron_flux': 'FLUX_AUTO',
            'kron_fluxerr': 'FLUXERR_AUTO',
            'kron_radius': 'KRON_RADIUS',
            'flux_radius': 'FLUX_RADIUS',
            'fwhm_pixel': 'FWHM_IMAGE',
            'max_value': 'FLUX_MAX',
            'fwhm_pixel': 'FWHM_IMAGE'
        }
        
        catalog = Table()#objects.copy()

        # Modification of the catalog
        for old, new in rename_map.items():
            if old in cat_tbl.colnames:
                catalog[new] = cat_tbl[old]
                
        catalog['MAG_AUTO'] = -2.5 * np.log10(catalog['FLUX_AUTO'])
        catalog['MAGERR_AUTO'] = 2.5 / np.log(10) * catalog['FLUXERR_AUTO'] / catalog['FLUX_AUTO']
        a = catalog['KRON_RADIUS'] * catalog['A_IMAGE']
        b = catalog['KRON_RADIUS'] * catalog['B_IMAGE']
        catalog['NPIX_AUTO'] = np.pi * a * b
        catalog['ELONGATION'] = catalog['A_IMAGE'] / catalog['B_IMAGE']
        catalog['SKYSIG'] = ndi_mean(bkgrms, labels=segm.data, index=segm.labels)
        catalog['THRESHOLD'] = catalog['SKYSIG'] * detection_sigma
        catalog['DETECT_THRESH'] = detection_sigma

        # Circular photometry 
        pixelscale = np.mean(target_img.pixelscale)
        aperture_diameter_arcsec = np.atleast_1d(aperture_diameter_arcsec)
        aperture_diameter_pixel = aperture_diameter_arcsec / pixelscale

        for i, diameter_pixel in enumerate(aperture_diameter_pixel):
            radius_pixel = diameter_pixel / 2
            circular_phot = cat.circular_photometry(radius=radius_pixel)
            suffix_key = '' if i == 0 else '_%d' % i
            catalog[f'FLUX_APER{suffix_key}'] = circular_phot[0]
            catalog[f'FLUXERR_APER{suffix_key}'] = circular_phot[1]
            catalog[f'MAG_APER{suffix_key}'] = -2.5*np.log10(circular_phot[0])
            catalog[f'MAGERR_APER{suffix_key}'] = 2.5/np.log(10) * circular_phot[1] / circular_phot[0]
            
            area_pixel = np.pi * (radius_pixel)**2
            catalog[f'NPIX_APER{suffix_key}'] = np.full(len(catalog), area_pixel)

        cat_path = target_img.savepath.catalogpath  
        target_catalog = TIPCatalog(path = cat_path, catalog_type = 'all', load = False) 
        target_catalog.data = catalog
        target_catalog.load_target_img(target_img = target_img)
        
        if save:
            target_catalog.write()
        else:
            target_catalog.remove()
            
        if visualize:
            save_path = None
            if save_fig:
                save_path = str(target_catalog.path) + '.png'
            self._visualize_objects(target_img=target_img, 
                                    objects=target_catalog.data, 
                                    size=1000, 
                                    save_path = save_path,
                                    show = visualize)
            
        if target_bkg is not None:
            target_img_sub.remove()
        return target_catalog
        
    def circular_photometry(self,
                            target_img: Union[ScienceImage, ReferenceImage],
                            x_arr: Union[float, list, np.ndarray],
                            y_arr: Union[float, list, np.ndarray],
                            aperture_diameter_arcsec: Union[float, list] = [5,7,10],
                            annulus_diameter_arcsec: Union[float, list] = None, # When local background is used
                            unit: str = 'pixel',
                            target_bkg: Optional[Background] = None,
                            target_mask: Union[str, Path, np.ndarray] = None,
                            target_bkgrms: Optional[Errormap] = None,
                            
                            # Other paramters
                            save: bool = True,
                            visualize: bool = True,
                            save_fig: bool = False,
                            **kwargs
                            ):
        '''
        x_arr: Union[float, list, np.ndarray] = objects_photutils['X_IMAGE']
        y_arr: Union[float, list, np.ndarray]= objects_photutils['Y_IMAGE']
        aperture_diameter_arcsec: Union[float, list] = [3,6,9]
        annulus_diameter_arcsec: Union[float, list] = None # When local background is used
        unit: str = 'pixel'
        target_bkg: Optional[Background] = None
        target_mask: Union[str, Path, np.ndarray] = None
        target_bkgrms: Optional[Errormap] = None
        
        # Other paramters
        visualize: bool = True
        '''

        # Step 1: Background subtraction
        data = self.to_native(target_img.data)
        if target_bkg is not None:
            target_img_sub = self.background.subtract_background(
                target_img=target_img,
                target_bkg=target_bkg,
                save = False,
                overwrite=False,
                visualize = visualize,
                save_fig = save_fig,
            )
            data = self.to_native(target_img_sub.data)
            
        # If target_bkgrms is not given, calculate it from the target image
        if target_bkgrms is None:
            target_bkgrms, _, _ = self.errormap.calculate_from_sourcemask(
                target_img = target_img,
                errormap_type = 'bkgrms',
                save = False,
                verbose = True,
                visualize = visualize,
                save_fig = False
            )
            
        # Step 2: Prepare data
        mask = self.to_native(target_mask.data) if target_mask is not None else None
        bkgrms = self.to_native(target_bkgrms.data)
        error = calc_total_error(data=data, bkg_error=bkgrms, effective_gain=target_img.egain)

        # Step 3: Pixel scale (arcsec/pix)
        pixelscale = np.mean(target_img.pixelscale)

        # Step 4: Normalize radius inputs
        aperture_diameter_arcsec = np.atleast_1d(aperture_diameter_arcsec)
        annulus_diameter_arcsec = np.atleast_1d(annulus_diameter_arcsec) if annulus_diameter_arcsec is not None else None
        aperture_diameter_pixel = aperture_diameter_arcsec / pixelscale
        annulus_diameter_pixel = annulus_diameter_arcsec / pixelscale if annulus_diameter_arcsec is not None else None

        # Step 5: Source positions
        x_arr = np.atleast_1d(x_arr)
        y_arr = np.atleast_1d(y_arr)

        skycoord = None
        if unit == 'pixel':
            positions = np.transpose((x_arr, y_arr))
            wcs = target_img.wcs
            if wcs:
                skycoord = pixel_to_skycoord(x_arr, y_arr, wcs)
        elif unit == 'coord':
            skycoord = SkyCoord(ra=x_arr, dec=y_arr, unit='deg')
            x_pix, y_pix = skycoord_to_pixel(skycoord, target_img.wcs)
            positions = np.transpose((x_pix, y_pix))
        else:
            raise ValueError("unit must be either 'pixel' or 'coord'")

        # Step 6: Photometry
        results = Table()
        results['X_IMAGE'] = positions[:, 0]
        results['Y_IMAGE'] = positions[:, 1]
        if skycoord:
            results['X_WORLD'] = skycoord.ra.value
            results['Y_WORLD'] = skycoord.dec.value
        
        for i, diameter_pixel in enumerate(aperture_diameter_pixel):
            radius_pixel = diameter_pixel /2
            aperture = CircularAperture(positions, r=radius_pixel)

            # Photometry on background-subtracted image
            phot_table = aperture_photometry(data, aperture, error=error, mask=mask)
            
            # Calculation for threshold (when error is defined)
            if bkgrms is not None:
                if i == 0:
                    rms_tbl = aperture_photometry(bkgrms, aperture, mask=mask)
                    results['SKYSIG'] = rms_tbl['aperture_sum'] / aperture.area
                
            suffix_key = '' if i == 0 else '_%d'%i
            flux_key = f'FLUX_APER{suffix_key}'
            fluxerr_key = f'FLUXERR_APER{suffix_key}'
            mag_key = f'MAG_APER{suffix_key}'
            magerr_key = f'MAGERR_APER{suffix_key}'
            annul_key = f'FLUX_ANNULUS{suffix_key}'
            magannul_key = f'MAG_ANNULUS{suffix_key}'
            npix_key = f'NPIX_APER{suffix_key}'
            
            # Aperture area
            results[npix_key] = np.full(len(results), aperture.area)

            # When annulus is defined
            if annulus_diameter_pixel is not None:
                annulus_pixel = annulus_diameter_pixel[i] /2
                annulus = CircularAnnulus(positions, r_in=radius_pixel, r_out=annulus_pixel)
                bkg_table = aperture_photometry(data, annulus, mask=mask)
                bkg_area_ratio = aperture.area / annulus.area
                annulus_bkg_flux = bkg_table['aperture_sum'] * bkg_area_ratio

                flux_net = phot_table['aperture_sum'] - annulus_bkg_flux
                results[flux_key] = flux_net
                results[annul_key] = annulus_bkg_flux
                results[mag_key] = -2.5 * np.log10(flux_net)
                results[magannul_key] = -2.5 * np.log10(annulus_bkg_flux)
            # When only aperutre is defined
            else:
                results[flux_key] = phot_table['aperture_sum']
                results[mag_key] = -2.5 * np.log10(phot_table['aperture_sum'])

            # When error is defined
            if 'aperture_sum_err' in phot_table.colnames:
                results[fluxerr_key] = phot_table['aperture_sum_err']
                results[magerr_key] = 2.5 / np.log(10) * phot_table['aperture_sum_err'] / phot_table['aperture_sum']

        cat_path = target_img.savepath.catalogpath.with_suffix('.circ.cat')
        target_catalog = TIPCatalog(path = cat_path, catalog_type = 'all', load = False) 
        target_catalog.data = results
        target_catalog.load_target_img(target_img = target_img)
        
        if save:
            target_catalog.write()
        else:
            target_catalog.remove()
        
        if visualize:
            save_path = None
            if save_fig:
                save_path = str(target_catalog.path) + '.png'
            self._visualize_objects(target_img=target_img, 
                                    objects=target_catalog.data, 
                                    size=100, 
                                    save_path = save_path,
                                    show = visualize)  
        
        if target_bkg is not None:
            target_img_sub.remove()
                  
        return target_catalog

    def elliptical_photometry(self,
                              target_img: Union[ScienceImage, ReferenceImage],
                              x_arr: Union[float, list, np.ndarray],  # pixel or RA
                              y_arr: Union[float, list, np.ndarray],  # pixel or Dec
                              sma_arr: Union[float, list, np.ndarray],  # semi-major (arcsec or pixel)
                              smi_arr: Union[float, list, np.ndarray],  # semi-minor (arcsec or pixel)
                              theta_arr: Union[float, list, np.ndarray],  # degrees
                              unit: str = 'pixel',             # 'pixel' or 'coord'
                              annulus_ratio: float = None, 
                              target_bkg: Optional[Background] = None,
                              target_mask: Union[str, Path, np.ndarray] = None,
                              target_bkgrms: Optional[Errormap] = None,
                              # Other parameters
                              save: bool = True,
                              visualize: bool = True,
                              save_fig: bool = False,
                              **kwargs
                              ):
        """
        Perform elliptical aperture photometry.

        Parameters
        ----------
        unit : str
            Unit of a/b axes. 'pixel' or 'coord' (arcsec). Will be converted to pixel using image pixelscale.
        coord_type : str
            Coordinate type for x/y positions. 'pixel' or 'sky' (RA/Dec in degrees).
        """

        # Step 1: Background subtraction
        if target_bkg is not None:
            target_img = self.background.subtract_backgrkound(
                target_img=target_img,
                target_bkg=target_bkg,
                save = False,
                overwrite=False,
                visualize = visualize,
                save_fig = save_fig,
            )

        # Step 2: Prepare image data
        data = self.to_native(target_img.data)
        mask = self.to_native(target_mask.data) if target_mask is not None else None
        bkgrms = self.to_native(target_bkgrms.data) if target_bkgrms is not None else None

        error = None
        if bkgrms is not None:
            error = calc_total_error(data=data, bkg_error=bkgrms, effective_gain=target_img.egain)

        # Step 3: Convert inputs
        x_raw = np.atleast_1d(x_arr)
        y_raw = np.atleast_1d(y_arr)
        sma_raw = np.atleast_1d(sma_arr)
        smi_raw = np.atleast_1d(smi_arr)
        theta = np.radians(np.atleast_1d(theta_arr))  # convert to radians

        # Step 4: Convert (RA, Dec) to (x, y) if needed
        skycoord = None
        if unit == 'coord':
            skycoord = SkyCoord(ra=x_raw, dec=y_raw, unit='deg')
            x, y = skycoord_to_pixel(skycoord, target_img.wcs)
        elif unit == 'pixel':
            wcs = target_img.wcs
            if wcs:
                skycoord = pixel_to_skycoord(x_raw, y_raw, wcs)
            x, y = x_raw, y_raw
        else:
            raise ValueError("coord_type must be either 'pixel' or 'sky'")

        # Step 5: Convert a/b from arcsec to pixels if needed
        pixelscale = np.mean(target_img.pixelscale)  # arcsec/pixel
        if unit == 'coord':
            sma_image = sma_raw / pixelscale
            smi_image = smi_raw / pixelscale
        elif unit == 'pixel':
            sma_image = sma_raw
            smi_image = smi_raw
        else:
            raise ValueError("unit must be either 'pixel' or 'coord'")

        # Step 6: Initialize results table
        results = Table()
        results['X_IMAGE'] = x
        results['Y_IMAGE'] = y
        results['SMA_IMAGE'] = sma_image
        results['SMI_IMAGE'] = smi_image
        results['THETA_IMAGE'] = np.degrees(theta)
        if skycoord:
            results['X_WORLD'] = skycoord.ra.value
            results['Y_WORLD'] = skycoord.dec.value
            results['SMA_WORLD'] = sma_image * pixelscale
            results['SMI_WORLD'] = smi_image * pixelscale

        # Step 8: Aperture photometry
        fluxes, fluxerrs, areas = [], [], []
        apertures = []
        for xi, yi, smai, smii, thetai in zip(x, y, sma_image, smi_image, theta):
            aperture = EllipticalAperture((xi, yi), a=smai, b=smii, theta=thetai)
            apertures.append(aperture)
            tbl = aperture_photometry(data, aperture, error=error, mask=mask)
            fluxes.append(tbl['aperture_sum'][0])
            areas.append(aperture.area)
            if 'aperture_sum_err' in tbl.colnames:
                fluxerrs.append(tbl['aperture_sum_err'][0])
        
        bkgrms_all = []
        if bkgrms is not None:
            for aperture in apertures:
                rms_tbl = aperture_photometry(bkgrms, aperture, mask=mask)
                bkgrms_all.append(rms_tbl['aperture_sum'][0] / aperture.area)
            results['SKYSIG'] = bkgrms_all
            
        results['FLUX_ELIP'] = fluxes
        results['MAG_ELIP'] = -2.5 * np.log10(fluxes)
        results['NPIX_ELIP'] = areas 

        if fluxerrs:
            results['FLUXERR_ELIP'] = fluxerrs
            results['MAGERR_ELIP'] = 2.5 / np.log(10) * np.array(fluxerrs) / np.array(fluxes)

        # Step 9: Annulus subtraction
        if annulus_ratio is not None:
            ann_fluxes, ann_areas = [], []
            for xi, yi, smai, smii, thetai in zip(x, y, sma_image, smi_image, theta):
                annulus = EllipticalAnnulus((xi, yi), a_in=smai, a_out=smai * annulus_ratio,
                                            b_in=smii, b_out=smii * annulus_ratio, theta=thetai)
                tbl = aperture_photometry(data, annulus, error=error, mask=mask)
                ann_fluxes.append(tbl['aperture_sum'][0])
                ann_areas.append(annulus.area)

            bkg_area_ratio = np.array(areas) / np.array(ann_areas)
            annulus_bkg_flux = np.array(ann_fluxes) * bkg_area_ratio
            results['FLUX_ELIP'] = np.array(fluxes) - annulus_bkg_flux
            results['FLUX_EANNULUS'] = annulus_bkg_flux
            results['MAG_ELIP'] = -2.5 * np.log10(results['FLUX_ELIP'])
            results['MAG_EANNULUS'] = -2.5 * np.log10(results['FLUX_EANNULUS'])
        
        cat_path = target_img.savepath.catalogpath.with_suffix('.ellip.cat')
        target_catalog = TIPCatalog(path = cat_path, catalog_type = 'all', load = False) 
        target_catalog.data = results
        target_catalog.load_target_img(target_img = target_img)
        
        if save:
            target_catalog.write()
        else:
            target_catalog.remove()
        
        if visualize:
            save_path = None
            if save_fig:
                save_path = str(target_catalog.path) + '.png'
            self._visualize_objects(target_img=target_img, 
                                    objects=target_catalog.data, 
                                    size=1000, 
                                    save_path = save_path,
                                    show = visualize)

        return target_catalog

    def _visualize_objects(self, 
                           target_img: Union[ScienceImage, ReferenceImage],
                           objects: Table,
                           size: int = 1000,
                           save_path: str = None,
                           show: bool = False):

        data = target_img.data
        h, w = data.shape

        # Step 1: Compute mean position of all sources
        mean_x = np.mean(objects['X_IMAGE'])
        mean_y = np.mean(objects['Y_IMAGE'])

        # Step 2: Find the object closest to that mean position
        dx = np.array(objects['X_IMAGE']) - mean_x
        dy = np.array(objects['Y_IMAGE']) - mean_y
        dist2 = dx**2 + dy**2
        closest_idx = np.argmin(dist2)

        # Use this object to center the zoom-in box
        center_x = float(objects[closest_idx]['X_IMAGE'])
        center_y = float(objects[closest_idx]['Y_IMAGE'])

        # Step 3: Define cropped region centered on that object
        half_box = size // 2
        x_min = int(max(0, center_x - half_box))
        x_max = int(min(w, center_x + half_box))
        y_min = int(max(0, center_y - half_box))
        y_max = int(min(h, center_y + half_box))

        cropped_data = data[y_min:y_max, x_min:x_max]

        # Step 4: Plot setup
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # --- Full image view ---
        m, s = np.mean(data), np.std(data)
        im0 = axes[0].imshow(data, interpolation='nearest', cmap='gray',
                            vmin=m - s, vmax=m + s, origin='lower')
        axes[0].set_title("Full Background-Subtracted Image")
        plt.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.04)

        # Draw red rectangle showing zoomed region
        zoom_box = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(zoom_box)

        # --- Zoomed-in region ---
        m_crop, s_crop = np.mean(cropped_data), np.std(cropped_data)
        im1 = axes[1].imshow(cropped_data, interpolation='nearest', cmap='gray',
                            vmin=m_crop - s_crop, vmax=m_crop + s_crop, origin='lower')
        axes[1].set_title(f"Zoomed Region Centered on Closest to Mean Position")

        # Step 5: Draw apertures for all sources in zoomed region
        for obj in objects:
            x, y = float(obj['X_IMAGE']), float(obj['Y_IMAGE'])
            if x_min <= x <= x_max and y_min <= y <= y_max:
                dx_local = x - x_min
                dy_local = y - y_min

                if 'A_IMAGE' in obj.colnames and 'B_IMAGE' in obj.colnames:
                    a = float(obj['A_IMAGE'])
                    b = float(obj['B_IMAGE'])
                    theta = float(obj['THETA_IMAGE']) if 'THETA_IMAGE' in obj.colnames else 0.0
                    patch = Ellipse((dx_local, dy_local), width=6*a, height=6*b, angle=theta,
                                    edgecolor='lime', facecolor='none', linewidth=1.5, alpha=0.6)
                else:
                    patch = Circle((dx_local, dy_local), radius= 5 / target_img.pixelscale[0],
                                edgecolor='lime', facecolor='none', linewidth=1.5, alpha=0.6)

                axes[1].add_patch(patch)

        plt.tight_layout()
        
        
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
            
        plt.close()
    


# %%

#%%
if __name__ == '__main__':
    from tippy.helper import TIPDataBrowser
    dbrowser = TIPDataBrowser('scidata')
    dbrowser.telkey = 'RASA36_KL4040_HIGH_1x1'
    target_imglist = dbrowser.search('Calib*60.fits', 'science')
#%%
    target_img  = target_imglist[0]
    target_bkg = Background(path = target_img.savepath.bkgpath, load = True)
    target_bkgrms = Errormap(target_img.savepath.bkgrmspath, emaptype = 'bkgrms', load = True)
    target_mask: Optional[Mask] = None # For masking certain source (such as hot pixels)
    mode: str = 'sep'
    detection_sigma: float = 1.5
    fwhm_estimate_pixel: float = 5.0
    minarea_pixels: int = 5
    
    # Save options
    save: bool = True
    save_region = True
    
    # Others
    visualize: bool = True
    verbose = True
    aperture_diameter_arcsec = [5,7,10]
    saturation_level = 60000
    kron_factor = 2.5
    x_key: str = 'X_IMAGE'
    y_key: str = 'Y_IMAGE'
    semimajor_key: str = 'A_IMAGE'
    semiminor_key: str = 'B_IMAGE'
    theta_key: str = 'THETA_IMAGE'
    kron_key: str = 'KRON_RADIUS'
    ra = 41.57941250
    dec = -30.2749
    fov_ra = 1
    fov_dec = 0.5

    # cat =  get_catalogs_coord(
    #     ra = ra,
    #     dec = dec,
    #     fov_ra = fov_ra,
    #     fov_dec = fov_dec,
    #     catalog_type = 'GAIAXP')
#%%
if __name__ == '__main__':
    cat = self.sex_photometry(
        target_img = target_img,
        target_bkg = target_bkg,
        target_bkgrms = target_bkgrms,
        target_mask = target_mask,
        detection_sigma = detection_sigma,
        aperture_diameter_arcsec = aperture_diameter_arcsec,
        saturation_level = saturation_level,
        kron_factor= kron_factor,
        save = save,
        visualize = visualize
    )
#%%
if __name__ == '__main__':
    img, cat = self.photutils_photometry(
        target_img = target_img,
        target_bkg = target_bkg,
        target_bkgrms = target_bkgrms,
        target_mask = target_mask,
        detection_sigma = detection_sigma,
        aperture_diameter_arcsec=aperture_diameter_arcsec,
        kron_factor = kron_factor,
        minarea_pixels = minarea_pixels,
        save = save,
        visualize = visualize
    )
#%%
if __name__ == '__main__':
    x_arr = cat.data['X_WORLD'][5000:5100]
    y_arr = cat.data['Y_WORLD'][5000:5100]
    aperture_diameter_arcsec = [5,7,10]

    circ_result = self.circular_photometry(
        target_img = target_img,
        x_arr = np.atleast_1d(x_arr),
        y_arr = np.atleast_1d(y_arr),
        aperture_diameter_arcsec = aperture_diameter_arcsec,
        annulus_diameter_arcsec= None,
        unit = 'coord',
        #annulus_size_arcsec= aperture_diameter_arcsec + 1,
        target_bkg = target_bkg,
        target_mask = None,
        target_bkgrms = target_bkgrms,
        visualize = visualize
    )

    # aperture_diameter_arcsec = [3,6,9]
    # circ_result = self.circular_photometry(
    #     target_img = target_img,
    #     x_arr = x_arr,
    #     y_arr = y_arr,
    #     aperture_diameter_arcsec = aperture_diameter_arcsec,
    #     annulus_diameter_arcsec = None,
    #     unit = 'pixel',
    #     #annulus_size_arcsec= aperture_diameter_arcsec + 1,
    #     target_bkg = target_bkg,
    #     target_mask = target_mask,
    #     target_bkgrms = target_bkgrms,
    #     visualize = visualize

    # )
#%%
if __name__ == '__main__':
    tbl = cat.data[5000:5100]    
    plt.scatter(tbl['MAG_APER_1'], circ_result['MAG_APER_1'])

#%%
if __name__ == '__main__':
    plt.scatter(cata.data['SKYSIG'], circ_result['SKYSIG'])
#%%
if __name__ == '__main__':
    x_arr = objects_sex['X_IMAGE']
    y_arr = objects_sex['Y_IMAGE']
    sma_arr = objects_sex['A_IMAGE'] * objects_sex['KRON_RADIUS'] 
    smi_arr = objects_sex['B_IMAGE'] * objects_sex['KRON_RADIUS'] 
    theta_arr = objects_sex['THETA_IMAGE']
    # #%%
    ellip_result = self.elliptical_photometry(
        target_img = target_img,
        x_arr = x_arr,
        y_arr = y_arr,
        sma_arr = sma_arr,
        smi_arr = smi_arr,
        theta_arr = theta_arr,
        unit = 'pixel',
        annulus_ratio = None,
        target_bkg = target_bkg,
        target_mask = target_mask,
        target_bkgrms = target_bkgrms
    )
#%%
if __name__ == '__main__':
    self.photometric_calibration(
        target_img = target_img,
        target_catalog = objects_sex[0],
    )
#%%
# for i in range(len(aperture_diameter_arcsec)):
#     if i == 0:
#         sex_key = 'FLUX_APER'
#     else:
#         sex_key = 'FLUX_APER_%s' %i
#     circ_key = 'FLUX_APER_%.1f' % aperture_diameter_arcsec[i]
#     plt.scatter(objects_sex[sex_key], circ_result[circ_key])
# # Outlier comes from the blending of two sources
# #%%
# #%%
# #plt.scatter(objects_sex['MAGERR_APER'], circ_result['MAGERR_APER_6.0'])
# #plt.scatter(objects_sex['MAGERR_APER_1'], circ_result['MAGERR_APER_9.0'])
# plt.scatter(objects_sex['MAGERR_APER_2'], circ_result['MAGERR_APER_12.0'])
# #%%
# x_arr = objects_sex['X_IMAGE']
# y_arr = objects_sex['Y_IMAGE']
# sma_arr = objects_sex['A_IMAGE'] * objects_sex['KRON_RADIUS']
# smi_arr = objects_sex['B_IMAGE'] * objects_sex['KRON_RADIUS']
# theta_arr = objects_sex['THETA_IMAGE']

# ellip_result = self.elliptical_photometry(
#     target_img = target_img,
#     x_arr = x_arr,
#     y_arr = y_arr,
#     sma_arr = sma_arr,
#     smi_arr = smi_arr,
#     theta_arr = theta_arr,
#     unit = 'pixel',
#     annulus_ratio = None,
#     target_bkg = target_bkg,
#     target_mask = target_mask,
#     target_bkgrms = target_bkgrms
# )
# #%%
# plt.scatter(objects_sex['MAG_AUTO'], ellip_result['MAG_ELIP'])
# #%%
# plt.scatter(objects_sex['FLUXERR_AUTO'], ellip_result['FLUXERR_ELIP'], c = objects_sex['FLUX_AUTO'])
# plt.plot([0, 10000], [0, 10000], 'r--')
# #%%
# #%%
# #%% objects_phot
# x_arr = objects_photutils['X_WORLD']
# y_arr = objects_photutils['Y_WORLD']

# #%%
# aperture_diameter_arcsec = aperture_diameter_arcsec
# circ_result = self.circular_photometry(
#     target_img = target_img,
#     x_arr = x_arr,
#     y_arr = y_arr,
#     aperture_diameter_arcsec = aperture_diameter_arcsec,
#     annulus_size = None,
#     unit = 'coord',
#     #annulus_size_arcsec= aperture_diameter_arcsec + 1,
#     target_bkg = target_bkg,
#     target_mask = target_mask,
#     target_bkgrms = target_bkgrms
# )
# #%%
# for i in range(len(aperture_diameter_arcsec)):
#     photutils_key = 'FLUX_APER_%.1f' %aperture_diameter_arcsec[i]
#     circ_key = 'FLUX_APER_%.1f' % aperture_diameter_arcsec[i]
#     plt.scatter(objects_photutils[photutils_key], circ_result[circ_key])
# #%%
# #plt.scatter(objects_photutils['MAGERR_APER_6.0'], circ_result['MAGERR_APER_6.0'])
# #plt.scatter(objects_photutils['MAGERR_APER_9.0'], circ_result['MAGERR_APER_9.0'])
# plt.scatter(objects_photutils['MAGERR_APER_12.0'], circ_result['MAGERR_APER_12.0'])
# # Outlier comes from the blending of two sources
# #%%
# #%%
# for i in range(len(aperture_diameter_arcsec)):
#     photutils_key = 'FLUX_APER_%d' %aperture_diameter_arcsec[i]
#     circ_key = 'FLUX_APER_%.1f' % aperture_diameter_arcsec[i]
#     plt.scatter(objects_photutils[photutils_key], circ_result[circ_key])

# #%%
# x_arr = objects_sex['X_IMAGE']
# y_arr = objects_sex['Y_IMAGE']
# sma_arr = objects_sex['A_IMAGE'] * objects_sex['KRON_RADIUS'] 
# smi_arr = objects_sex['B_IMAGE'] * objects_sex['KRON_RADIUS'] 
# theta_arr = objects_sex['THETA_IMAGE']
# # #%%
# ellip_result = self.elliptical_photometry(
#     target_img = target_img,
#     x_arr = x_arr,
#     y_arr = y_arr,
#     sma_arr = sma_arr,
#     smi_arr = smi_arr,
#     theta_arr = theta_arr,
#     unit = 'pixel',
#     annulus_ratio = None,
#     target_bkg = target_bkg,
#     target_mask = target_mask,
#     target_bkgrms = target_bkgrms
# )
# # #%%
# plt.scatter(objects_sex['FLUX_AUTO'], ellip_result['FLUX_ELIP'])
# plt.plot([0, 2e7], [0, 2e7], 'r--')
#plt.xlim(0, 2e5)
#plt.ylim(0, 2e5)
# #%%
# plt.scatter(objects_photutils['FLUXERR_KRON'], ellip_result['FLUXERR_ELIP'], c = objects_photutils['FLUX_KRON'])
# plt.plot([0, 10000], [0, 10000], 'r--')

# #%%
# # %%
# from astropy.table import Table
# import numpy as np
# from scipy.spatial import cKDTree
# from typing import List, Tuple

# def match_sources(tbl_1: Table, tbl_2: Table,
#                   keys_1: List[str], keys_2: List[str],
#                   max_distance: float = 0.5
#                  ):
#     """
#     Match sources from table1 to table2 using 2D Cartesian nearest-neighbor search.

#     Parameters
#     ----------
#     tbl_1, tbl_2 : QTable
#         Astropy tables containing source positions.
#     keys_1 : list of str
#         Column names in table1 for 2D coordinates, e.g., ['x', 'y'].
#     keys_2 : list of str
#         Column names in table2 for 2D coordinates, e.g., ['x', 'y'].
#     max_distance : float
#         Maximum matching distance (in same units as coordinate values).

#     Returns
#     -------
#     matched : QTable
#         Table with columns from both inputs and distance of matched pairs.
#     unmatched1 : QTable
#         Rows from table1 with no match in table2.
#     unmatched2 : QTable
#         Rows from table2 not matched by any in table1.
#     """
#     coords1 = np.vstack([tbl_1[keys_1[0]], tbl_1[keys_1[1]]]).T
#     coords2 = np.vstack([tbl_2[keys_2[0]], tbl_2[keys_2[1]]]).T

#     tree = cKDTree(coords2)
#     distances, indices = tree.query(coords1, distance_upper_bound=max_distance)

#     matched_mask = distances != np.inf
#     matched1 = tbl_1[matched_mask]
#     matched2 = tbl_2[indices[matched_mask]]

#     # Combine matched rows
#     matched = Table()
#     for col in matched1.colnames:
#         matched[f"{col}_1"] = matched1[col]
#     for col in matched2.colnames:
#         matched[f"{col}_2"] = matched2[col]
#     matched['distance'] = distances[matched_mask]

#     # Unmatched rows
#     unmatched1 = tbl_1[~matched_mask]
#     matched_indices2 = set(indices[matched_mask])
#     unmatched_indices2 = list(set(range(len(tbl_2))) - matched_indices2)
#     unmatched2 = tbl_2[unmatched_indices2]

#     return matched, unmatched1, unmatched2
# #%%
# from astropy.io import ascii
# objects_psf = ascii.read(target_img.savepath.catalogpath)

# matched, unmatched_1, unmatched_2 = match_sources(tbl_1 = objects_psf, tbl_2 = objects_sex, keys_1 = ['x_fit', 'y_fit'], keys_2 = ['X_IMAGE', 'Y_IMAGE'], max_distance = 0.5)

# plt.figure(figsize=(10, 10))
# plt.scatter(-2.5*np.log10(matched['flux_fit_1']), matched['MAG_AUTO_2'], alpha = 0.7, c = matched['FLUX_AUTO_2'])
# plt.colorbar()
# plt.plot([-16, -11], [-16, -11], 'r--')
# tbl = matched[((-2.5*np.log10(matched['flux_fit_1']) - matched['MAG_AUTO_2']) > 0.35) & (matched['MAG_AUTO_2'] > -14)]
# tbl2 = matched[((-2.5*np.log10(matched['flux_fit_1']) - matched['MAG_AUTO_2']) > 0.2) & (matched['MAG_AUTO_2'] > -14)]
# tbl3 = matched[((-2.5*np.log10(matched['flux_fit_1']) - matched['MAG_AUTO_2']) < 0.1) & (matched['MAG_AUTO_2'] > -14)]

# plt.scatter(-2.5*np.log10(tbl['flux_fit_1']), tbl['MAG_AUTO_2'], alpha = 0.7, c = 'r')
# plt.scatter(-2.5*np.log10(tbl2['flux_fit_1']), tbl2['MAG_AUTO_2'], alpha = 0.7, c = 'b')
# # #plt.xlim(0, 15e5)
# # #plt.ylim(0, 15e5)
# # A = PhotometryHelper()
# # reg = to_regions(reg_x = tbl['X_IMAGE_2'], reg_y = tbl['Y_IMAGE_2'], reg_size = 5, unit = 'pixel', output_file_path = target_img.savepath.savepath.with_suffix('.reg'))
# from tippy.helper import PhotometryHelper
# helper = PhotometryHelper()
# reg = helper.to_regions(reg_x = tbl2['X_IMAGE_2'], reg_y = tbl2['Y_IMAGE_2'], reg_size = 5, unit = 'pixel', output_file_path = target_img.savepath.savepath.with_suffix('.reg'))
# # reg = to_regions(reg_x = tbl3['X_IMAGE_2'], reg_y = tbl3['Y_IMAGE_2'], reg_size = 5, unit = 'pixel', output_file_path = target_img.savepath.savepath.with_suffix('.reg'))

# #%%

# #%%

# matched, unmatched_1, unmatched_2 = match_sources(tbl_1 = objects_photutils, tbl_2 = objects_sex, keys_1 = ['X_IMAGE', 'Y_IMAGE'], keys_2 = ['X_IMAGE', 'Y_IMAGE'], max_distance = 0.5)
# common_colnames = list(set(objects_photutils.colnames) & set(objects_sex.colnames))

# for colname in common_colnames:
#     plt.title(colname)
#     plt.scatter(matched[f'{colname}_1'], matched[f'{colname}_2'], alpha = 0.3)
#     val_min, vam_max = min(matched[f'{colname}_1'].min(), matched[f'{colname}_2'].min()), max(matched[f'{colname}_1'].max(), matched[f'{colname}_2'].max())
#     plt.plot([0, vam_max], [0, vam_max], 'r--')
#     plt.title(colname)
#     plt.show()
# #%%

# for colname in common_colnames:
#     plt.title(colname)
#     plt.hist(unmatched_1[colname], bins = 100, color = 'r', label = 'unmatched') # 대부분 Signal 아닌 것이 잡힌듯. PSF 사이즈보다 작음.
#     plt.hist(matched[f'{colname}_1'], bins = 100, color = 'k', label = 'sep', histtype = 'step')
#     plt.hist(matched[f'{colname}_2'], bins = 100, color = 'b', label = 'phot', histtype = 'step', linestyle = '--', linewidth = 2)
#     plt.show()

# # %%
# plt.scatter(matched['FLUX_KRON_1'], matched['FLUX_AUTO_2'])
# #plt.scatter(matched['FLUX_APER_12_1'], matched['FLUX_APER_2_2'])
# val_min, vam_max = min(matched['FLUX_KRON_1'].min(), matched['FLUX_AUTO_2'].min()), max(matched['FLUX_KRON_1'].max(), matched['FLUX_AUTO_2'].max())
# plt.plot([0, vam_max], [0, vam_max], 'r--')
# # %%
# plt.scatter(matched['FLUXERR_APER_2'], matched['FLUXERR_APER_6_1'])
# # %%
# plt.scatter(objects_photutils['FLUX_APER_6'], objects_photutils['FLUXERR_APER_6'], alpha =0.1)
# plt.scatter(objects_sex['FLUX_APER'], objects_sex['FLUXERR_APER'], alpha = 0.1)
# # %%

# #%%
# matched, unmatched_1, unmatched_2 = match_sources(tbl_1 = objects_photutils, tbl_2 = ellip_result, keys_1 = ['X_IMAGE', 'Y_IMAGE'], keys_2 = ['X_IMAGE', 'Y_IMAGE'], max_distance = 0.5)
# common_colnames = list(set(objects_photutils.colnames) & set(ellip_result.colnames))
# for colname in common_colnames:
#     plt.title(colname)
#     plt.scatter(matched[f'{colname}_1'], matched[f'{colname}_2'], alpha = 0.3)
#     val_min, vam_max = min(matched[f'{colname}_1'].min(), matched[f'{colname}_2'].min()), max(matched[f'{colname}_1'].max(), matched[f'{colname}_2'].max())
#     plt.plot([0, vam_max], [0, vam_max], 'r--')
#     plt.title(colname)
#     plt.show()

# for colname in common_colnames:
#     plt.title(colname)
#     plt.hist(unmatched_1[colname], bins = 100, color = 'r', label = 'unmatched') # 대부분 Signal 아닌 것이 잡힌듯. PSF 사이즈보다 작음.
#     plt.hist(matched[f'{colname}_1'], bins = 100, color = 'k', label = 'sep', histtype = 'step')
#     plt.hist(matched[f'{colname}_2'], bins = 100, color = 'b', label = 'phot', histtype = 'step', linestyle = '--', linewidth = 2)
#     plt.show()

# # %%
# plt.scatter(matched['FLUXERR_KRON_1'], matched['FLUXERR_ELIP_2'])
# # %%
