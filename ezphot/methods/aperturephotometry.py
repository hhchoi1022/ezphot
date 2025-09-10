#%%
import inspect
from typing import Union, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle
from tqdm import tqdm
from scipy.ndimage import mean as ndi_mean
from astropy.table import Table
from astropy.modeling import models, fitting
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog
from photutils.aperture import (
    CircularAperture, CircularAnnulus, aperture_photometry,
    EllipticalAperture, EllipticalAnnulus
)
from photutils.utils import calc_total_error

from ezphot.methods import BackgroundGenerator, ErrormapGenerator
from ezphot.helper import Helper
from ezphot.imageobjects import (
    ScienceImage, ReferenceImage, CalibrationImage,
    Background, Mask, Errormap
)
from ezphot.dataobjects import Catalog
from ezphot.utils import *


class AperturePhotometry:
    """
    Method class for performing aperture photometry.
    
    This class provides methods 
    
    1. SExtractor-based aperture photometry.
    
    2. Photutils-based aperture photometry.
    
    3. Circular aperture photometry.
    
    4. Elliptical aperture photometry.
    
    5. Kron aperture photometry.
    
    6. Circular aperture photometry.
    
    """
    def __init__(self):
        """
        Initialize the AperturePhotometry class.
        
        Parameters
        ----------
        None
        """ 
        self.helper = Helper()
        self.background = BackgroundGenerator()
        self.errormap = ErrormapGenerator() 
        
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
        
    def run_photometry(self,
                       target_img: Union[ScienceImage, ReferenceImage, CalibrationImage],
                       target_bkg: Optional[Background] = None,
                       target_bkgrms: Optional[Errormap] = None,
                       target_mask: Optional[Mask] = None,
                       sex_params: dict = None,
                       detection_sigma: float = 5,
                       aperture_diameter_arcsec: Union[float, list] = [5, 7, 10],
                       aperture_diameter_seeing: Union[float, list] = [3.5, 4.5],
                       saturation_level: float = 60000,
                       kron_factor: float = 2.5,
                       save: bool = True,
                       verbose: bool = True,
                       visualize: bool = True,
                       save_fig: bool = False,
                       **kwargs
                       ):
        """
        Run photometry on the target image.
        """
        pass

    def sex_photometry(self,
                       # Input parameters
                       target_img: Union[ScienceImage, ReferenceImage, CalibrationImage], 
                       target_bkg: Optional[Background] = None, # If target_bkg is given, subtract background before sextractor
                       target_bkgrms: Optional[Errormap] = None, # It must be background error map
                       target_mask: Optional[Mask] = None, # For masking certain source (such as hot pixels)
                       sex_params: dict = None,
                       detection_sigma: float = 5,
                       aperture_diameter_arcsec: Union[float, list] = [5, 7, 10],
                       aperture_diameter_seeing: Union[float, list] = [3.5, 4.5], # If given, use seeing to calculate aperture size
                       saturation_level: float = 60000,
                       kron_factor: float = 2.5,
                       
                       # Others
                       save: bool = True,
                       verbose: bool = True,
                       visualize: bool = True,
                       save_fig: bool = False,
                       **kwargs
                       ):
        """
        Perform aperture photometry using SExtractor.

        Parameters
        ----------
        target_img : Union[ScienceImage, ReferenceImage, CalibrationImage]
            The target image to perform aperture photometry on.
        target_bkg : Optional[Background], optional
            The background to subtract from the target image. Defaults to None.
        target_bkgrms : Optional[Errormap], optional
            The background RMS to use for the aperture photometry. Defaults to None.
        detection_sigma : float, optional
            The detection sigma for the aperture photometry. Defaults to 5.
        aperture_diameter_arcsec : Union[float, list], optional
            The aperture diameter in arcseconds. Defaults to [5, 7, 10].
        aperture_diameter_seeing : Union[float, list], optional
            The aperture diameter in seeing units. Defaults to [3.5, 4.5].
        saturation_level : float, optional
            The saturation level for the aperture photometry. Defaults to 60000.
        kron_factor : float, optional
            The Kron factor for the aperture photometry. Defaults to 2.5.
        save : bool, optional
            Whether to save the aperture photometry catalog. Defaults to True.
        verbose : bool, optional
            Whether to print verbose output. Defaults to True.
        visualize : bool, optional
            Whether to visualize the aperture photometry. Defaults to True.
        save_fig : bool, optional
            Whether to save the aperture photometry figure. Defaults to False.

        Raises
        ------
        ValueError: If target_img is not a ScienceImage, ReferenceImage, or CalibrationImage.
        RuntimeError: If the source extractor fails.
        RuntimeError: If the source extractor fails.

        Returns
        -------
        target_catalog : Catalog
            The aperture photometry catalog.
        """
        if not isinstance(target_img, (ScienceImage, ReferenceImage, CalibrationImage)):
            raise ValueError('target_img must be a ScienceImage, ReferenceImage, or CalibrationImage.')
        
        if not target_img.is_saved:
            target_img.write(verbose = False)
        
        # If sex_params is None, use default parameters
        if sex_params is None:
            sex_params = dict()
        
        #target_img = target_img.copy()
        img_path = target_img.savepath.savepath
        cat_path = target_img.savepath.catalogpath
        sexconfig_path = target_img.config['SEX_CONFIG']
        all_sexconfig = self.helper.load_config(sexconfig_path)
        
        # If target_bkg is given, subtract background before sextractor
        remove_subbkg = False
        if target_bkg is not None:
            target_img_sub = self.background.subtract_background(
                target_img = target_img, 
                target_bkg = target_bkg,
                save = True,
                overwrite = False,
                visualize = visualize,
                save_fig = save_fig,
                verbose = verbose
            )
            saturation_level -= target_bkg.info.BKGVALU
            img_path = target_img_sub.savepath.savepath
            sex_params['BACK_TYPE'] = 'MANUAL'
            remove_subbkg = True
        else:
            target_img_sub = target_img
            
        # Background RMS
        remove_bkgrms = False
        if target_bkgrms is not None:
            if not target_bkgrms.is_saved:
                target_bkgrms.write(verbose = False)
                remove_bkgrms = True
            bkgrms_path = target_bkgrms.savepath.savepath
        else:
            bkgrms_path = None
        
        # Mask
        remove_mask = False
        if target_mask is not None:
            if not target_mask.is_saved:
                target_mask.write(verbose = False)
                remove_mask = True
            mask_path = target_mask.savepath.savepath
        else:
            mask_path = None
                    
        # First Sextractor run to estimate seeing if not provided
        if target_img.seeing is None:
            sex_params['DETECT_THRESH'] = 10
            result, catalog_first, global_bkgval, global_bkgrms = self.helper.run_sextractor(
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
                rough_flags &= (catalog_first['CLASS_STAR'] > 0.2)
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
        
        all_apertures = []
        aperture_diameter_arcsec = np.atleast_1d(aperture_diameter_arcsec)
        for aperture_size in aperture_diameter_arcsec:
            all_apertures.append(aperture_size)
        if aperture_diameter_seeing is not None:
            aperture_diameter_seeing = np.atleast_1d(aperture_diameter_seeing)
            for aperture_seeing_ratio in aperture_diameter_seeing:
                all_apertures.append(seeing_estimate * aperture_seeing_ratio)
        
        aperture_diameter_pixel = ','.join(["%.2f"%(float(size / target_img.telinfo['pixelscale'])) for size in all_apertures])
        sex_params['PHOT_APERTURES'] = aperture_diameter_pixel # This is aperture size in pixel
        sex_params['SATUR_LEVEL'] = saturation_level
        sex_params['PHOT_AUTOPARAMS'] = f"{kron_factor},3.5"
        if 'DETECT_MINAREA' in sex_params.keys():
            if sex_params['DETECT_MINAREA'] < 3:
                sex_params['DETECT_MINAREA'] = 3
                self.helper.print(f"[WARNING] DETECT_MINAREA is less than 3. It is set to 3.", verbose)
        if 'DETECT_THRESH' in sex_params.keys():
            if sex_params['DETECT_THRESH'] < 1:
                sex_params['DETECT_THRESH'] = 1
                self.helper.print('[WARNING] DETECT_THRESH is less than 1. It is set to 1.', verbose)
        for key, value in sex_params.items():
            all_sexconfig[key] = value

        # If DETECT_MINAREA is not set, use the default value from the config
        # if 'DETECT_MINAREA' not in sex_params.keys():
        #     sex_params['DETECT_MINAREA'] = all_sexconfig['DETECT_MINAREA']
        # If DETECT_THRESH is not set, use the detection_sigma
        #if 'DETECT_THRESH' not in sex_params.keys():
        #    sex_params['DETECT_THRESH'] = detection_sigma / np.sqrt(sex_params['DETECT_MINAREA'])
        #if 'ANALYSIS_THRESH' not in sex_params.keys():
        #    sex_params['ANALYSIS_THRESH'] = sex_params['DETECT_THRESH']

        
        # Second Sextractor run with the estimated parameters
        result, catalog, global_bkgval, global_bkgrms = self.helper.run_sextractor(
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

        pixelscale = np.mean(target_img.pixelscale)
        for i, ap_size_arcsec in enumerate(all_apertures):
            radius_pixel = ap_size_arcsec / pixelscale / 2
            area_pixel = np.pi * (radius_pixel)**2
            npix_colname = 'NPIX_APER' if i == 0 else f'NPIX_APER_{i}'
            catalog[npix_colname] = np.full(len(catalog), area_pixel)
            pixsize_colname = 'PIXSIZE_APER' if i == 0 else f'PIXSIZE_APER_{i}'
            catalog[pixsize_colname] = np.full(len(catalog), ap_size_arcsec / pixelscale)

        target_catalog = Catalog(path = cat_path, catalog_type = 'all', load = False) 
        target_catalog.data = catalog
        
        if save:
            target_catalog.write(verbose = verbose)
        else:
            target_catalog.remove(verbose = verbose)
        
        if visualize:
            save_path = None
            if save_fig:
                save_path = str(target_catalog.path) + '.png'
            self._visualize_objects(target_img=target_img_sub, 
                                    objects=target_catalog.data, 
                                    size=1000, 
                                    save_path = save_path,
                                    show = visualize)
            
        for remove_trigger, remove_object in zip(
            [remove_subbkg, remove_bkgrms, remove_mask],
            [target_img_sub, target_bkgrms, target_mask]):
            if remove_trigger:
                remove_object.remove(verbose = verbose)
        
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
        
        Parameters
        ----------
        target_img : Union[ScienceImage, ReferenceImage, CalibrationImage]
            The target image to perform aperture photometry on.
        target_bkg : Optional[Background], optional
            The background to subtract from the target image. Defaults to None.
        target_bkgrms : Optional[Errormap], optional
            The background RMS to use for the aperture photometry. Defaults to None.
        target_mask : Optional[Mask], optional
            The mask to use for the aperture photometry. Defaults to None.
        detection_sigma : float, optional
            The detection sigma for the aperture photometry. Defaults to 1.5.
        aperture_diameter_arcsec : Union[float, list], optional
            The aperture diameter in arcseconds. Defaults to [5,7,10].
        kron_factor : float, optional
            The Kron factor for the aperture photometry. Defaults to 2.5.
        minarea_pixels : int, optional
            The minimum area in pixels for the aperture photometry. Defaults to 5.
        calc_accurate_fwhm : bool, optional
            Whether to calculate the accurate FWHM. Defaults to True.
        save : bool, optional
            Whether to save the aperture photometry catalog. Defaults to True.
        verbose : bool, optional
            Whether to print verbose output. Defaults to True.
        visualize : bool, optional
            Whether to visualize the aperture photometry. Defaults to True.
        save_fig : bool, optional
            Whether to save the aperture photometry figure. Defaults to False.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        target_catalog : Catalog
            The aperture photometry catalog.
        """
        if not isinstance(target_img, (ScienceImage, ReferenceImage, CalibrationImage)):
            raise ValueError("target_img must be a ScienceImage, ReferenceImage, or CalibrationImage.")
        
        bkgrms_map = None
        bkgrms = None
        mask_map = target_mask.data if target_mask is not None else None
        mask_map = self.helper.to_native(mask_map)
        target_data = self.helper.to_native(target_img.data)
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
            target_data = self.helper.to_native(target_img_sub.data)
            target_bkg_data = target_bkg.data
        else:
            target_bkg_data = None

        # Step 2: Set error map
        if target_bkgrms is not None:
            bkgrms = self.helper.to_native(target_bkgrms.data)
        elif bkgrms_map is not None:
            bkgrms = self.helper.to_native(bkgrms_map.data)
        else:
            # Use sigma-clipped std if no error map available
            bkgrms_map, _, _ = self.errormap.calculate_errormap_from_image(
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
                stamp = self.helper.img_extract_stamp(target_data, x0, y0, size=25)  # make your own stamp function
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
        target_catalog = Catalog(path = cat_path, catalog_type = 'all', load = False) 
        target_catalog.data = catalog
        
        if save:
            target_catalog.write(verbose = verbose)
        else:
            target_catalog.remove(verbose = verbose)
            
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
            target_img_sub.remove(verbose = verbose)
        return target_catalog
        
    def circular_photometry(self,
                            target_img: Union[ScienceImage, ReferenceImage],
                            x_arr: Union[float, list, np.ndarray],
                            y_arr: Union[float, list, np.ndarray],
                            aperture_diameter_arcsec: Union[float, list] = [5,7,10],
                            aperture_diameter_seeing: Union[float, list] = [3.5, 4.5], # If given, use seeing to calculate aperture size
                            annulus_width_arcsec: Union[float, list] = None, # When local background is used
                            unit: str = 'pixel',
                            target_bkg: Optional[Background] = None,
                            target_mask: Union[str, Path, np.ndarray] = None,
                            target_bkgrms: Optional[Errormap] = None,
                            
                            # Other paramters
                            save: bool = True,
                            verbose: bool = True,
                            visualize: bool = True,
                            save_fig: bool = False,
                            **kwargs
                            ):
        """
        Perform circular aperture photometry.
        
        Parameters
        ----------
        target_img : Union[ScienceImage, ReferenceImage, CalibrationImage]
            The target image to perform aperture photometry on.
        x_arr : Union[float, list, np.ndarray]
            The x-coordinates of the sources.
        y_arr : Union[float, list, np.ndarray]
            The y-coordinates of the sources.
        aperture_diameter_arcsec : Union[float, list], optional
            The aperture diameter in arcseconds. Defaults to [5,7,10].
        annulus_diameter_arcsec : Union[float, list], optional
            The annulus diameter in arcseconds. Defaults to None.
        unit : str, optional
            The unit of the coordinates. Defaults to 'pixel'.
        target_bkg : Optional[Background], optional
            The background to subtract from the target image. Defaults to None.
        target_mask : Optional[Mask], optional
            The mask to use for the aperture photometry. Defaults to None.
        target_bkgrms : Optional[Errormap], optional
            The background RMS to use for the aperture photometry. Defaults to None.
        save : bool, optional
            Whether to save the aperture photometry catalog. Defaults to True.
        visualize : bool, optional
            Whether to visualize the aperture photometry. Defaults to True.
        save_fig : bool, optional
            Whether to save the aperture photometry figure. Defaults to False.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        target_catalog : Catalog
            The aperture photometry catalog.
        """

        # Step 1: Background subtraction
        data = self.helper.to_native(target_img.data)
        if target_bkg is not None:
            target_img_sub = self.background.subtract_background(
                target_img=target_img,
                target_bkg=target_bkg,
                save = False,
                overwrite=False,
                visualize = visualize,
                save_fig = save_fig,
            )
            data = self.helper.to_native(target_img_sub.data)
            
        # If target_bkgrms is not given, calculate it from the target image
        if target_bkgrms is None:
            target_bkgrms, _, _ = self.errormap.calculate_errormap_from_image(
                target_img = target_img,
                errormap_type = 'bkgrms',
                save = False,
                verbose = True,
                visualize = visualize,
                save_fig = False
            )
            
        # Step 2: Prepare data
        mask = self.helper.to_native(target_mask.data) if target_mask is not None else None
        bkgrms = self.helper.to_native(target_bkgrms.data)
        error = calc_total_error(data=data, bkg_error=bkgrms, effective_gain=target_img.egain)

        # Step 3: Pixel scale (arcsec/pix)
        pixelscale = np.mean(target_img.pixelscale)

        # Step 4: Normalize radius inputs
        all_apertures = []
        aperture_diameter_arcsec = np.atleast_1d(aperture_diameter_arcsec)
        for aperture_size in aperture_diameter_arcsec:
            all_apertures.append(aperture_size)
        if aperture_diameter_seeing is not None:
            if target_img.seeing is not None:
                aperture_diameter_seeing = np.atleast_1d(aperture_diameter_seeing)
                for aperture_seeing_ratio in aperture_diameter_seeing:
                    all_apertures.append(target_img.seeing * aperture_seeing_ratio)
            else:
                print("[WARNING] target_img.seeing is not defined. Using aperture_diameter_seeing is ignored.")
        aperture_diameter_pixel = all_apertures / pixelscale
        
        all_annulus = []
        if annulus_width_arcsec is not None:
            for aperture_diameter in all_apertures:
                annulus_diameter = aperture_diameter + annulus_width_arcsec
                all_annulus.append(annulus_diameter)
            print('ANNULUS applied')
        annulus_diameter_pixel = all_annulus / pixelscale if all_annulus else None

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
                
            suffix_key = '' if i == 0 else '_%d'%i
            flux_key = f'FLUX_APER{suffix_key}'
            fluxerr_key = f'FLUXERR_APER{suffix_key}'
            mag_key = f'MAG_APER{suffix_key}'
            magerr_key = f'MAGERR_APER{suffix_key}'
            annul_key = f'FLUX_ANNULUS{suffix_key}'
            magannul_key = f'MAG_ANNULUS{suffix_key}'
            npix_key = f'NPIX_APER{suffix_key}'
            skysig_key = f'SKYSIG_APER{suffix_key}'
            ul5_key = f'UL5_APER{suffix_key}'
            ul3_key = f'UL3_APER{suffix_key}'

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
                
            # Calculation for threshold (when error is defined)
            if bkgrms is not None:
                var_tbl = aperture_photometry(bkgrms**2, aperture, mask=mask)
                results[skysig_key] = np.sqrt(var_tbl['aperture_sum'])
                ul5_flux = 5 * results[skysig_key]
                ul3_flux = 3 * results[skysig_key]
                results[ul5_key] = -2.5 * np.log10(ul5_flux)
                results[ul3_key] = -2.5 * np.log10(ul3_flux)

        cat_path = target_img.savepath.catalogpath.with_suffix('.circ.cat')
        target_catalog = Catalog(path = cat_path, catalog_type = 'all', load = False) 
        target_catalog.data = results
        
        if save:
            target_catalog.write(verbose = verbose)
        else:
            target_catalog.remove(verbose = verbose)
        
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
            target_img_sub.remove(verbose = verbose)
                  
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
                              verbose: bool = True,
                              visualize: bool = True,
                              save_fig: bool = False,
                              **kwargs
                              ):
        """
        Perform elliptical aperture photometry.

        Parameters
        ----------
        target_img : Union[ScienceImage, ReferenceImage, CalibrationImage]
            The target image to perform aperture photometry on.
        x_arr : Union[float, list, np.ndarray]
            The x-coordinates of the sources.
        y_arr : Union[float, list, np.ndarray]
            The y-coordinates of the sources.
        sma_arr : Union[float, list, np.ndarray]
            The semi-major axis of the sources.
        smi_arr : Union[float, list, np.ndarray]
            The semi-minor axis of the sources.
        theta_arr : Union[float, list, np.ndarray]
            The position angle of the sources.
        unit : str, optional
            The unit of the coordinates. Defaults to 'pixel'.
        annulus_ratio : float, optional
            The ratio of the annulus diameter to the semi-major axis. Defaults to None.
        target_bkg : Optional[Background], optional
            The background to subtract from the target image. Defaults to None.
        target_mask : Optional[Mask], optional
            The mask to use for the aperture photometry. Defaults to None.
        target_bkgrms : Optional[Errormap], optional
            The background RMS to use for the aperture photometry. Defaults to None.
        save : bool, optional
            Whether to save the aperture photometry catalog. Defaults to True.
        visualize : bool, optional
            Whether to visualize the aperture photometry. Defaults to True.
        save_fig : bool, optional
            Whether to save the aperture photometry figure. Defaults to False.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        target_catalog : Catalog
            The aperture photometry catalog.
        """

        # Step 1: Background subtraction
        if target_bkg is not None:
            target_img = self.background.subtract_background(
                target_img=target_img,
                target_bkg=target_bkg,
                save = False,
                overwrite=False,
                visualize = visualize,
                save_fig = save_fig,
            )

        # Step 2: Prepare image data
        data = self.helper.to_native(target_img.data)
        mask = self.helper.to_native(target_mask.data) if target_mask is not None else None
        bkgrms = self.helper.to_native(target_bkgrms.data) if target_bkgrms is not None else None

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
        target_catalog = Catalog(path = cat_path, catalog_type = 'all', load = False) 
        target_catalog.data = results
        
        if save:
            target_catalog.write(verbose = verbose)
        else:
            target_catalog.remove(verbose = verbose)
        
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
                elif 'SMA_IMAGE' in obj.colnames and 'SMI_IMAGE' in obj.colnames:
                    a = float(obj['SMA_IMAGE'])
                    b = float(obj['SMI_IMAGE'])
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
    