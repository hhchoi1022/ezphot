
#%%# 
import inspect
from typing import Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from astropy.table import Table, vstack, hstack
from astropy.stats import sigma_clipped_stats
from astropy.wcs.utils import pixel_to_skycoord
from astropy.visualization import simple_norm
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.io import fits
from astropy.coordinates import SkyCoord
from photutils.segmentation import deblend_sources
from photutils.utils import calc_total_error
from photutils.psf import EPSFModel
from tqdm import tqdm
import sep

from ezphot.methods import BackgroundGenerator, MaskGenerator, ErrormapGenerator
from ezphot.helper import Helper
from ezphot.imageobjects import (
    ScienceImage, ReferenceImage, CalibrationImage,
    Background, Mask, Errormap
)
from ezphot.utils import *


class PSFPhotometry:
    """
    PSFPhotometry class for performing PSF photometry.
    
    This class provides methods 
    
    1. Select PSF stars
    
    2. Extract sources
    
    3. Build EPSF model
    
    4. PSF photometry
    
    5. Build EPSF model with PSFEx
    
    6. PSF photometry with PSFEx
    
    7. Build EPSF model with SExtractor
    
    """
    def __init__(self):
        """
        Initialize the PSFPhotometry class.
        
        Parameters
        ----------
        None
        """
        self.helper = Helper()
        self.background = BackgroundGenerator()
        self.masking = MaskGenerator()
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
        self.helper.print(f"Help for {self.__class__.__name__}\n{help_text}\nPublic methods:\n" + "\n".join(lines), True)

    def select_psfstars(self,
                        target_catalog: Table,
                        image_shape: tuple,
                        num_grids: Optional[int] = 4,
                        max_per_grid: int = 30,
                        
                        # Selection criteria
                        snr_lower: float = 15,
                                                
                        eccentricity_upper: float = None,
                        eccentricity_sigma: float = 5,
                        fwhm_lower: float = 2,
                        fwhm_sigma: float = 5,
                        
                        inner_fraction = 0.9,
                        isolation_radius: float = 30.0,
                        verbose: bool = True,
                        visualize: bool = True,
                        
                        mag_key: str = 'MAG_AUTO',
                        flux_key: str = 'FLUX_APER',
                        fluxerr_key: str = 'FLUXERR_APER',
                        fwhm_key: str = 'FWHM_IMAGE',
                        x_key: str = 'X_IMAGE',
                        y_key: str = 'Y_IMAGE',  
                        eccentricity_key: str = 'ECCENTRICITY',                      
                        **kwargs) -> Table:
        """
        Select PSF stars from the catalog.
        
        Parameters
        ----------
        target_catalog : Table
            The catalog to select PSF stars from.
        image_shape : tuple
            The shape of the image.
        num_grids : int or None
            If None or 0, apply selection globally instead of grid-by-grid.
        max_per_grid : int
            The maximum number of stars to select per grid.
        snr_lower : float
            The minimum SNR for the stars.
        eccentricity_upper : float or None
            The maximum eccentricity for the stars.
        eccentricity_sigma : float
            The sigma for the eccentricity cut.
        fwhm_lower : float
            The minimum FWHM for the stars.
        fwhm_sigma : float
            The sigma for the FWHM cut.
        inner_fraction : float
            The fraction of the image to select stars from.
        isolation_radius : float
            The radius to use for the isolation cut.
        verbose : bool
            Whether to print verbose output.
        visualize : bool
            Whether to visualize the stars.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        target_catalog : Table
            The catalog of selected PSF stars.
        """
        from astropy.table import Table
        import numpy as np
        from scipy.spatial import cKDTree
        
        def select_subset(sources: Table, visualize: bool = False, verbose = True) -> Table:
            if sources is None or len(sources) == 0:
                return sources
            
            if fwhm_key not in sources.keys():
                visualize = False
                self.helper.print(f"Warning: '{fwhm_key}' not found in sources. Visualization disabled.", verbose)
            if visualize:
                plt.figure(dpi=300)
                plt.xlabel(mag_key)
                plt.ylabel(fwhm_key)
                plt.title("Star selection filtering")

            def _plot_if_visualize(x, y, color, label, alpha=0.4):
                if visualize:  # or pass `visualize` as a parameter
                    plt.scatter(x, y, c=color, alpha=alpha, label=label)
            _plot_if_visualize(sources[mag_key], sources[fwhm_key], 'k', label = 'All sources', alpha = 0.3)#, c = sources[x_key])
            filtered_catalog = sources.copy()
            self.helper.print(f'Initial sources: {len(filtered_catalog)}', verbose)

            # Step 1: Inner region cut
            if x_key not in filtered_catalog.keys() or y_key not in filtered_catalog.keys():
                self.helper.print(f"Warning: '{x_key}' or '{y_key}' not found in sources.", verbose)
            else:
                x_vals = filtered_catalog[x_key]
                y_vals = filtered_catalog[y_key]

                x_min, x_max = np.min(x_vals), np.max(x_vals)
                y_min, y_max = np.min(y_vals), np.max(y_vals)

                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2
                x_half_range = (x_max - x_min) * inner_fraction // 2
                y_half_range = (y_max - y_min) * inner_fraction // 2
                
                x_inner_min = x_center - x_half_range
                x_inner_max = x_center + x_half_range
                y_inner_min = y_center - y_half_range
                y_inner_max = y_center + y_half_range

                inner_mask = (
                    (x_vals >= x_inner_min) & (x_vals <= x_inner_max) &
                    (y_vals >= y_inner_min) & (y_vals <= y_inner_max)
                )
                filtered_catalog = filtered_catalog[inner_mask]
                self.helper.print(f'[INNERREGION CUT] {len(filtered_catalog)} sources passed within X = [{x_inner_min},{x_inner_max}], Y = [{y_inner_min},{y_inner_max}]', verbose)
            
            _plot_if_visualize(filtered_catalog[mag_key], filtered_catalog[fwhm_key], 'r', label = 'InnerRegion cut', alpha = 0.3)

            # Step 2: Isolation
            if x_key not in filtered_catalog.keys() or y_key not in filtered_catalog.keys():
                self.helper.print(f"Warning: '{x_key}' or '{y_key}' not found in sources.")
            else:
                # Step 1.1: Build KD-tree
                positions = np.vstack([filtered_catalog[x_key].value, filtered_catalog[y_key].value]).T
                tree = cKDTree(positions)
                neighbors = tree.query_ball_tree(tree, r=isolation_radius)

                # Step 1.2: Keep only isolated sources
                isolated_mask = np.array([len(nbrs) == 1 for nbrs in neighbors])
                filtered_catalog = filtered_catalog[isolated_mask]
                self.helper.print(f'[ISOLATION CUT] {len(filtered_catalog)} sources passed with isolation radius {isolation_radius} pixels', verbose)

            _plot_if_visualize(filtered_catalog[mag_key], filtered_catalog[fwhm_key], 'g', label = 'Isolation cut', alpha = 0.3)

            # Step 3: SNR cut
            if flux_key not in filtered_catalog.keys():
                self.helper.print(f"Warning: '{flux_key}' not found in sources.", verbose)
            else:
                filtered_catalog = filtered_catalog[(filtered_catalog[flux_key] / filtered_catalog[fluxerr_key]) > snr_lower]
                self.helper.print(f"[SNR CUT]: {len(filtered_catalog)} sources passed with SNR > {snr_lower}", verbose)
                
            _plot_if_visualize(filtered_catalog[mag_key], filtered_catalog[fwhm_key], 'b', label = 'SNR cut', alpha = 0.3)

            # Step 5: FWHM absolute and relative cut
            if fwhm_key not in filtered_catalog.keys():
                self.helper.print(f"Warning: '{fwhm_key}' not found in sources.", verbose)
            else:
                # Step 5.1: Absolute cut: remove too small sources
                abs_fwhm_mask = filtered_catalog[fwhm_key] > fwhm_lower
                filtered_catalog = filtered_catalog[abs_fwhm_mask]

                # Step 5.2: Relative cut: sigma-clipped sources
                fwhm_values = filtered_catalog[fwhm_key]
                fwhm_mean, fwhm_median, fwhm_std = sigma_clipped_stats(fwhm_values, sigma=5.0, maxiters=5)
                clip_mask = np.abs(fwhm_values - fwhm_median) <= fwhm_sigma * fwhm_std
                filtered_catalog = filtered_catalog[clip_mask]
                self.helper.print(
                    f"[FWHM CUT]: {len(filtered_catalog)} sources passed with FWHM > {fwhm_lower} and within ±{fwhm_sigma} sigma"
                    f"around median ({fwhm_median:.2f} ± {fwhm_std:.2f})",
                    verbose
                ) 
                
            _plot_if_visualize(filtered_catalog[mag_key], filtered_catalog[fwhm_key], 'orange', label = 'FWHM cut', alpha = 0.3)


            # Step 6: Eccentricity cut
            if eccentricity_key not in filtered_catalog.keys():
                self.helper.print(f"Warning: '{eccentricity_key}' not found in sources.", verbose)
            else:
                eccen_vals = filtered_catalog[eccentricity_key]
                
                # Step 6.1: Absolute limit
                if eccentricity_upper is not None:
                    abs_eccen_mask = eccen_vals < eccentricity_upper
                    filtered_catalog = filtered_catalog[abs_eccen_mask]

                # Step 6.2: Sigma-clipping
                eccen_mean, eccen_median, eccen_std = sigma_clipped_stats(eccen_vals, sigma=5.0, maxiters=5)
                sigclip_mask = np.abs(eccen_vals - eccen_median) < eccentricity_sigma * eccen_std
                filtered_catalog = filtered_catalog[sigclip_mask]

                self.helper.print(f"[ECCENTRICITY CUT]: {len(filtered_catalog)} passed eccentricity < {eccentricity_upper} and within ±{eccentricity_sigma} sigma of median ({eccen_median:.2f} ± {eccen_std:.2f})", verbose)

            _plot_if_visualize(filtered_catalog[mag_key], filtered_catalog[fwhm_key], 'purple', label = 'Final selected', alpha = 0.3)
            seeing = np.median(filtered_catalog[fwhm_key])
            
            if visualize:
                plt.legend()
                plt.ylim(seeing - 2, seeing + 10)
                valid_mag = target_catalog[mag_key][~np.isnan(target_catalog[mag_key])]
                if len(valid_mag) > 0:
                    plt.xlim(np.min(valid_mag) - 0.5, np.max(valid_mag) + 0.5)
                else:
                    # No valid data to set xlim
                    self.helper.print("Warning: No valid magnitudes for setting xlim.", verbose)
            return filtered_catalog

        # ---------- Global Mode ----------
        if num_grids is None or num_grids <= 1:
            result = select_subset(target_catalog)
            total_rows = len(result)
            n = min(max_per_grid, total_rows)
            random_indices = np.random.choice(total_rows, size=n, replace=False)
            result = result[random_indices]

            if result is None:
                if verbose:
                    self.helper.print("No suitable stars found (global mode).", verbose)
                return Table()
            if verbose:
                self.helper.print(f"Selected {len(result)} stars from {len(target_catalog)} candidates (global mode).", verbose)
            return result

        # ---------- Grid Mode ----------
        h, w = image_shape
        y_step = h // num_grids
        x_step = w // num_grids

        selected_catalog = []

        for i in range(num_grids):
            for j in range(num_grids):
                # Grid bounds
                x_min = j * x_step
                x_max = (j + 1) * x_step
                y_min = i * y_step
                y_max = (i + 1) * y_step

                in_cell = (target_catalog['X_IMAGE'] >= x_min) & (target_catalog['X_IMAGE'] < x_max) & \
                        (target_catalog['Y_IMAGE'] >= y_min) & (target_catalog['Y_IMAGE'] < y_max)
                sub_tbl = target_catalog[in_cell]
                result = select_subset(sub_tbl)
                total_rows = len(result)
                n = min(max_per_grid, total_rows)
                random_indices = np.random.choice(total_rows, size=n, replace=False)
                result = result[random_indices]
                
                if result is not None:
                    selected_catalog.append(result)
                    
                self.helper.print(f"Grid ({i},{j}): {len(sub_tbl)} candidates, {len(result) if result is not None else 0} selected.", verbose)

        if len(selected_catalog) == 0:
            if verbose:
                self.helper.print("No suitable stars found in any grid.", verbose)
            return Table()

        result_tbl = Table(np.hstack(selected_catalog))

        self.helper.print(f"Selected {len(result_tbl)} stars from {len(target_catalog)} candidates (grid mode).", verbose)

        if visualize:
            self._visualize_objects(target_catalog.target_img, result_tbl, size=2000)
        return result_tbl

    #TODO CHANGE EXTRACT SOURCES TO FIND PEAKS OR MAKE ANOTHER METHOD TO FIND PEAKS
    def extract_sources(self,
                        target_img: Union[ScienceImage, ReferenceImage, CalibrationImage],
                        target_bkg: Optional[Background] = None,
                        target_bkgrms: Optional[Errormap] = None,
                        target_mask: Optional[Mask] = None,
                     
                        # Detection criteria
                        detection_sigma: float = 5,
                        minarea_pixels: int = 5,
                        deblend_nlevels: int = 32,
                        deblend_contrast: float = 0.003,
                        aperture_diameter_arcsec: float = 7.0,
                                             
                        # Other parameters
                        visualize: bool = True,
                        verbose: bool = True,
                        save: bool = False,
                        **kwargs) -> Table:
        """
        Detect stars using SEP (with background RMS).
        
        Parameters
        ----------
        target_img : Union[ScienceImage, ReferenceImage, CalibrationImage]
            The target image to detect sources from.
        target_bkg : Optional[Background], optional
            The background to subtract from the target image. Defaults to None.
        target_bkgrms : Optional[Errormap], optional
            The background RMS to use for the detection. Defaults to None.
        target_mask : Optional[Mask], optional
            The mask to use for the detection. Defaults to None.
        detection_sigma : float, optional
            Detection threshold in units of background RMS. Defaults to 5.
        minarea_pixels : int, optional
            The minimum area in pixels for the detection. Defaults to 5.
        deblend_nlevels : int, optional
            The number of levels for the deblending. Defaults to 32.
        deblend_contrast : float, optional
            The contrast for the deblending. Defaults to 0.003.
        aperture_diameter_arcsec : float, optional
            The aperture diameter in arcseconds. Defaults to 7.0.
        visualize : bool, optional
            Whether to visualize the detection. Defaults to True.
        verbose : bool, optional
            Whether to print verbose output. Defaults to True.
        save : bool, optional
            Whether to save the detection. Defaults to False.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        new_objects : Table
            The catalog of detected sources.
        target_img : ScienceImage
            The target image with detected sources.
        """
        from photutils.detection import DAOStarFinder, find_peaks
        from photutils.segmentation import detect_sources, SourceCatalog
        
        bkgrms_map = None
        bkgrms = None
        mask_map = target_mask.data if target_mask is not None else None
        mask_map = self.helper.to_native(mask_map)
        # Step 1: Background subtraction
        if target_bkg is not None:
            target_img = self.background.subtract_background(
                target_img=target_img,
                target_bkg=target_bkg,
                overwrite=False
            )

        # Step 2: Set error map
        if bkgrms_map is not None:
            bkgrms = self.helper.to_native(bkgrms_map.data)
        elif target_bkgrms is not None:
            bkgrms = self.helper.to_native(target_bkgrms.data)
        else:
            # Use sigma-clipped std if no error map available
            bkgrms_map, _, _ = self.errormap.calculate_errormap_from_image(
                target_img=target_img,
                target_mask=None,
                mode='sep',
                errormap_type='bkgrms',
                save=save,
                visualize=visualize,
                **kwargs
            )
            bkgrms = bkgrms_map.data

        target_data = self.helper.to_native(target_img.data).astype(np.float32)
        error = calc_total_error(data=target_data, bkg_error=bkgrms, effective_gain=target_img.egain)

        threshold = detection_sigma * bkgrms / np.sqrt(minarea_pixels)
        segm = detect_sources(target_data, threshold, npixels=minarea_pixels, mask=mask_map)
        if segm is None:
            self.helper.print("No sources detected.", verbose)
            #return Table()
        segm = deblend_sources(data = target_data, segment_img = segm, npixels=minarea_pixels, nlevels = deblend_nlevels, contrast = deblend_contrast)
        cat = SourceCatalog(data=target_data, segment_img=segm, error=error, wcs = target_img.wcs)        
        cat_tbl = cat.to_table()

        cat_tbl['flux_radius'] = cat.fluxfrac_radius(0.5)
        cat_tbl['ellipticity'] = cat.ellipticity
        cat_tbl['fwhm_pixel'] = 2.3548 / 1.1774 * cat_tbl['flux_radius'] # From flux radius, gaussian approximation
        coords = cat_tbl['sky_centroid']
        cat_tbl['ra'] = coords.ra.value
        cat_tbl['dec'] = coords.dec.value
        
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
        }
        
        new_objects = Table()#objects.copy()

        # Modification of the catalog
        for old, new in rename_map.items():
            if old in cat_tbl.colnames:
                new_objects[new] = cat_tbl[old]
        
        pixelscale = np.mean(target_img.pixelscale)
        aperture_diameter_arcsec = np.atleast_1d(aperture_diameter_arcsec)
        aperture_diameter_pixel = aperture_diameter_arcsec / pixelscale
        aper_phot = cat.circular_photometry(radius=aperture_diameter_pixel/2)
        new_objects['FLUX_APER'] = aper_phot[0]
        new_objects['MAG_AUTO'] = -2.5 * np.log10(new_objects['FLUX_APER'])
        new_objects['MAG_AUTO'] = -2.5 * np.log10(new_objects['FLUX_AUTO'])
        if error is not None:
            new_objects['FLUXERR_APER'] = aper_phot[1]
            new_objects['MAGERR_APER'] = 2.5 / np.log(10) * new_objects['FLUXERR_APER'] / new_objects['FLUX_APER']
            new_objects['MAGERR_AUTO'] = 2.5 / np.log(10) * new_objects['FLUXERR_AUTO'] / new_objects['FLUX_AUTO']
        self.helper.print('Detected {} sources'.format(len(new_objects)), verbose)       
        
        if visualize:
            self._visualize_objects(target_img, new_objects, size=1000)
         
        return new_objects, target_img
    
    # Deprecated, use build_epsf_model_psfex
    def build_epsf_model(self,
                         image_data: np.ndarray,
                         sources: Table,
                         num_grids: Optional[int] = None,
                         fwhm_estimate_pixel: float = 4.0,
                         oversampling: int = 4,
                         min_stars_per_grid: int = 15,
                         verbose: bool = True,
                         visualize: bool = True,
                         num_show: int = 9,
                         **kwargs):
        """
        Build EPSF model from sources.
        
        Parameters
        ----------
        image_data : np.ndarray
            The image data.
        sources : Table
            The sources to build the EPSF model from.
        num_grids : int or None
            If None or 0, apply selection globally instead of grid-by-grid.
        fwhm_estimate_pixel : float, optional
            The estimated FWHM of the stars in pixels. Defaults to 4.0.
        oversampling : int, optional
            The oversampling factor for the EPSF model. Defaults to 4.
        min_stars_per_grid : int, optional
            The minimum number of stars per grid. Defaults to 15.
        verbose : bool, optional
            Whether to print verbose output. Defaults to True.
        visualize : bool, optional
            Whether to visualize the EPSF model. Defaults to True.
        num_show : int, optional
            The number of stars to show in the visualization. Defaults to 9.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        epsf_model : EPSFModel
            The EPSF model.
        """
        from astropy.nddata import NDData
        from photutils.psf import extract_stars, EPSFBuilder
        from astropy.visualization import simple_norm
        from astropy.nddata import NDData
        from photutils.psf import extract_stars, EPSFBuilder
        from astropy.visualization import simple_norm
        import matplotlib.pyplot as plt
        from astropy.table import Table
        import numpy as np

        size = int(np.ceil(6 * fwhm_estimate_pixel)) | 1
        stamp_size = (size, size)
        nddata = NDData(image_data)

        if num_grids is None or num_grids <= 1:
            tbl = Table()
            tbl['x'] = sources['X_IMAGE']
            tbl['y'] = sources['Y_IMAGE']

            stars = extract_stars(nddata, tbl, size=stamp_size)
            if len(stars) < min_stars_per_grid:
                raise ValueError(f"Only {len(stars)} stars found, not enough to build global EPSF.")

            builder = EPSFBuilder(oversampling=oversampling, maxiters=10, progress_bar=False)
            epsf_model, fitted_stars = builder.build_epsf(stars)
            ny, nx = epsf_model.data.shape
            epsf_model.x_0 = nx / 2
            epsf_model.y_0 = ny / 2
            if verbose:
                self.helper.print(f"Built global EPSF from {len(fitted_stars)} stars.", verbose)
            if visualize:
                self._visualize_epsf_stars_and_model(epsf_model, fitted_stars)
                self._visualize_epsf_residuals(epsf_model, fitted_stars, num_show = num_show)
            return epsf_model
        else:
            h, w = image_data.shape
            x_step = w // num_grids
            y_step = h // num_grids
            epsf_model_dict = {}
            epsf_stars_dict = {}
            failed_grids = []
            failed_stars = []

            # First pass: Try building EPSF model for each grid
            for i in range(num_grids):
                for j in range(num_grids):
                    x_min, x_max = j * x_step, (j + 1) * x_step
                    y_min, y_max = i * y_step, (i + 1) * y_step

                    in_cell = (sources['X_IMAGE'] >= x_min) & (sources['X_IMAGE'] < x_max) & \
                            (sources['Y_IMAGE'] >= y_min) & (sources['Y_IMAGE'] < y_max)
                    grid_sources = sources[in_cell]
                    
                    tbl = Table()
                    tbl['x'] = grid_sources['X_IMAGE']
                    tbl['y'] = grid_sources['Y_IMAGE']

                    try:
                        from photutils.psf import EPSFStars
                        stars = extract_stars(nddata, tbl, size=stamp_size)
                        stars = [star for star in stars if np.all(np.isfinite(star.data))]
                        stars = EPSFStars(stars)
                        builder = EPSFBuilder(oversampling=oversampling, maxiters=10, progress_bar=False)
                        epsf_model, fitted_stars = builder.build_epsf(stars)
                        ny, nx = epsf_model.data.shape
                        epsf_model.x_0 = nx / 2
                        epsf_model.y_0 = ny / 2
                        
                        if len(stars) < min_stars_per_grid:
                            if verbose:
                                self.helper.print(f"Grid ({i},{j}): Only {len(stars)} stars extracted, marking for fallback.", verbose)
                            failed_grids.append((i, j))
                            failed_stars.append(fitted_stars)
                            continue
                        
                        epsf_model_dict[(i, j)] = epsf_model
                        epsf_stars_dict[(i, j)] = fitted_stars

                        if verbose:
                            self.helper.print(f"Grid ({i},{j}): EPSF built from {len(fitted_stars)} stars.", verbose)

                    except Exception as e:
                        self.helper.print(f"Grid ({i},{j}) failed: {e}", verbose)
                        failed_grids.append((i, j))
                        failed_stars.append(fitted_stars)
                        continue

            # Second pass: fallback to adjacent grids
            from math import sqrt

            center_i, center_j = num_grids // 2, num_grids // 2

            for (i, j), fitted_stars in zip(failed_grids, failed_stars):
                # Generate all valid neighbors excluding (i, j) itself
                neighbors = [
                    (i + di, j + dj)
                    for di in [-1, 0, 1]
                    for dj in [-1, 0, 1]
                    if not (di == 0 and dj == 0) and
                    0 <= i + di < num_grids and
                    0 <= j + dj < num_grids
                ]

                # Sort neighbors by distance from the center (prefer inner grids)
                neighbors.sort(key=lambda ij: sqrt((ij[0] - center_i)**2 + (ij[1] - center_j)**2))

                neighbor_found = False
                for ni, nj in neighbors:
                    if (ni, nj) in epsf_model_dict:
                        epsf_model = epsf_model_dict[(ni, nj)]
                        epsf_model_dict[(i, j)] = epsf_model
                        epsf_stars_dict[(i, j)] = fitted_stars
                        neighbor_found = True
                        if verbose:
                            self.helper.print(f"Grid ({i},{j}): Fallback to inner neighbor ({ni},{nj})", verbose)
                        break

                if not neighbor_found and verbose:
                    self.helper.print(f"Grid ({i},{j}): No valid inner neighbors found for fallback.", verbose)

            from collections import OrderedDict

            # Sort epsf_dict by grid keys (i, j)
            epsf_model_dict = OrderedDict(sorted(epsf_model_dict.items(), key=lambda x: (x[0][0], x[0][1])))
            epsf_stars_dict = OrderedDict(sorted(epsf_stars_dict.items(), key=lambda x: (x[0][0], x[0][1])))

            if visualize:
                self._visualize_epsf_stars_and_model_grid(epsf_model_dict= epsf_model_dict, epsf_stars_dict=epsf_stars_dict)
                self._visualize_epsf_residuals_grid(epsf_model_dict= epsf_model_dict, epsf_stars_dict = epsf_stars_dict, num_show_each_grid=num_show)

            return epsf_model_dict

    def build_epsf_model_psfex(self,
                               target_img: Union[ScienceImage, ReferenceImage],
                               target_bkg: Optional[Background] = None,
                               target_bkgrms: Optional[Errormap] = None,
                               target_mask: Optional[Mask] = None,
                            
                               # Detection criteria
                               detection_sigma: float = 5.0,
                               minarea_pixels: int = 5,
                               fwhm_estimate_pixel: float = 4.0,
                               saturation_level: float = 20000,
                                                    
                               # PSFEx parameters 
                               psf_size: int = None,
                               num_grids: int = 1,
                               oversampling: int = 1,
                               eccentricity_upper: float = 0.4,
                               verbose: bool = True,
                               visualize: bool = True,
                               **kwargs
                               ):
        """
        Build EPSF models per grid using PSFEx with dynamic parameter handling.

        Parameters
        ----------
        target_img : Union[ScienceImage, ReferenceImage]
            The target image to run PSFEx on.
        target_bkg : Optional[Background], optional
            The background to subtract from the target image. Defaults to None.
        target_bkgrms : Optional[Errormap], optional
            The background RMS to use for the detection. Defaults to None.
        target_mask : Optional[Mask], optional
            The mask to use for the detection. Defaults to None.
        detection_sigma : float, optional
            Detection threshold in units of background RMS. Defaults to 5.0.
        minarea_pixels : int, optional
            The minimum area in pixels for the detection. Defaults to 5.
        fwhm_estimate_pixel : float, optional
            Estimated FWHM of stars in pixels. Defaults to 4.0.
        saturation_level : float, optional
            The saturation level for the detection. Defaults to 20000.
        psf_size : int, optional
            The size of the PSF model. Defaults to None.
        num_grids : int, optional
            The number of grids along one axis (e.g., 3 for 3x3). Defaults to 1.
        oversampling : int, optional
            Oversampling factor for subpixel modeling. Defaults to 1.
        eccentricity_upper : float, optional
            The maximum eccentricity for the stars. Defaults to 0.4.
        verbose : bool, optional
            Whether to print verbose output. Defaults to True.
        visualize : bool, optional
            Whether to visualize the EPSF model. Defaults to True.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        epsf_model_dict : dict
            The dictionary of EPSF models.
        """

        bkgrms_map = None
        bkgrms = None
        # mask_map = target_mask.data if target_mask is not None else None
        # mask_map = self.to_native(mask_map)
        # Step 1: Background subtraction
        if target_bkg is not None:
            target_img = self.background.subtract_background(
                target_img=target_img,
                target_bkg=target_bkg,
                save= True,
                overwrite=False,
                visualize=False,
                save_fig=False,
            )
        else:
            pass
        
        if target_mask is not None:
            if not target_mask.is_exists:
                target_mask.write(verbose = verbose)

        # Step 2: Set error map
        if target_bkgrms is not None:
            if not target_bkgrms.is_exists:
                target_bkgrms.write(verbose = verbose)

        # FOR SExtractor
        psfex_sexparams = dict()
        psfex_sexparams['CATALOG_TYPE'] = 'FITS_LDAC' # Required format for PSFEx
        psfex_sexparams['PARAMETERS_NAME'] = 'psfex.param' # Required format for PSFEx
        psfex_sexparams['DETECT_MINAREA'] = '%d'%(minarea_pixels) # Minimum area for detection
        psfex_sexparams["DETECT_THRESH"] = '%.1f'%(detection_sigma) # Detection threshold
        psfex_sexparams["ANALYSIS_THRESH"] = '%.1f'%(detection_sigma) # Analysis threshold
        if target_bkgrms is not None:
            psfex_sexparams["WEIGHT_TYPE"] = 'MAP_RMS' # When target_bkgrms is given, this RMS map will be used for calculating SNR 
            psfex_sexparams["WEIGHT_IMAGE"] = '%s'%target_bkgrms.path # Path of the RMS map
            psfex_sexparams["WEIGHT_GAIN"] = 'N'
        if target_mask is not None:
            sex_params['FLAG_IMAGE'] = str(target_mask)
            sex_params['PARAMETERS_NAME'] = 'sexflag.param'       
        psfex_sexparams['PHOT_APERTURES'] = '%.1f'%(fwhm_estimate_pixel*3) # fwhm * 3
        psfex_sexparams['SATUR_LEVEL'] = str(saturation_level)
        if target_img.status.BKGSUB['status']:
            psfex_sexparams['BACK_TYPE'] = 'MANUAL' # Disable background estimation
        
        # FOR PSFEx
        def estimate_psf_size(fwhm: float, margin: float = 6.0) -> str:
            size = int(np.ceil(margin * fwhm)) | 1  # ensure odd
            return size

        psfex_params = dict()
        if psf_size is None:
            psf_size = estimate_psf_size(fwhm_estimate_pixel) * oversampling
            
        ecc = np.clip(eccentricity_upper, 0, 0.999999)  # prevent sqrt of negative
        sqrt_term = np.sqrt(1 - ecc**2)
        ellipticity_upper = 0.4#(1 - sqrt_term) / (1 + sqrt_term)
        psfex_params["PSF_SAMPLING"] = '%.3f'%(1 / oversampling) # Oversampling parameter
        psfex_params['PSF_SIZE'] = f'%d,%d'%(psf_size,psf_size) # Size of the PSF model
        psfex_params["PSFVAR_NSNAP"] = '%d'%num_grids # Dimension of the grid
        if num_grids > 1:
            psfex_params["PSFVAR_DEGREES"] = "1,1" # Polynomial degree for PSF model
        else:
            psfex_params["PSFVAR_DEGREES"] = "2,2" # Polynomial degree for PSF model
        
        # Sampling parameters
        psfex_params['SAMPLE_AUTOSELECT'] = 'Y'
        psfex_params["SAMPLE_FWHMRANGE"] = '%.1f,%.1f'%(0.5*fwhm_estimate_pixel, 2.5*fwhm_estimate_pixel) # FWHM range for sampling
        psfex_params["SAMPLE_MINSN"] = '%.1f'%detection_sigma
        psfex_params["SAMPLE_MAXELLIP"] = '%.4f'%ellipticity_upper
        
        psfex_params["CHECKIMAGE_TYPE"] = "SNAPSHOTS"
        psfex_params["CHECKIMAGE_NAME"] = "snap.fits"
        psfex_params["SAMPLE_MAXELLIP"] = str(ellipticity_upper)

        # Run PSFEx
        snapshot = self.helper.run_psfex(
            target_path=target_img.path,
            psfex_sexconfigfile=target_img.config['PSFEX_SEXCONFIG'],
            psfex_configfile=target_img.config['PSFEX_CONFIG'],
            psfex_sexparams=psfex_sexparams,
            psfex_params=psfex_params,
            target_outpath=target_img.savepath.savepath,
            verbose=verbose
        )[0]

        from astropy.io import fits
        psf_data = fits.getdata(snapshot)

        epsf_model_dict = {}
        epsf_stars_dict = {}

        for i in range(num_grids):
            for j in range(num_grids):
                y_start = i * psf_size
                y_end = y_start + psf_size
                x_start = j * psf_size
                x_end = x_start + psf_size
                psf_patch = psf_data[y_start:y_end, x_start:x_end]

                epsf_model = EPSFModel(data=psf_patch, oversampling = oversampling)
                ny, nx = epsf_model.data.shape
                epsf_model.x_0 = nx / 2
                epsf_model.y_0 = ny / 2
                
                epsf_model_dict[(i, j)] = epsf_model
                epsf_stars_dict[(i, j)] = []

        if verbose:
            self.helper.print(f"Assigned PSFEx EPSF models from {num_grids}x{num_grids} grid (shape {psf_data.shape}).", verbose)
            
        if visualize:
            self._visualize_epsf_stars_and_model_grid(epsf_model_dict=epsf_model_dict, epsf_stars_dict=epsf_stars_dict)

        return epsf_model_dict
    
    def psf_photometry(self,
                    target_img: Union[ScienceImage, ReferenceImage],
                    epsf_model_dict: Union[EPSFModel, dict],
                    target_bkg: Optional[Background] = None, 
                    target_bkgrms: Optional[Errormap] = None,
                    sources: Optional[Table] = None,
                    target_mask: Optional[Mask] = None,
                    
                    # Detection parameters
                    detection_sigma: float = 1.5,
                    minarea_pixels: int = 5,
                    deblend_nlevels: int = 32,
                    deblend_contrast: float = 0.003,

                    # PSF Photometry parameters
                    fwhm_estimate_pixel: float = 5.0,
                    n_iterations: int = 0,
                                        
                    visualize: bool = True,
                    verbose: bool = True,
                    save: bool = True,
                    apply_aperture_correction: bool = True,
                    **kwargs):
        """
        Perform PSF photometry on a target image using an EPSF model.

        Parameters
        ----------
        target_img : Union[ScienceImage, ReferenceImage]
            The target image to perform PSF photometry on.
        epsf_model_dict : Union[EPSFModel, dict]
            The EPSF model to use for the PSF photometry.
        target_bkg : Optional[Background], optional
            The background to subtract from the target image. Defaults to None.
        target_bkgrms : Optional[Errormap], optional
            The background RMS to use for the detection. Defaults to None.
        sources : Optional[Table], optional
            The sources to use for the PSF photometry. Defaults to None.
        target_mask : Optional[Mask], optional
            The mask to use for the detection. Defaults to None.
        detection_sigma : float, optional
            Detection threshold in units of background RMS. Defaults to 1.5.
        minarea_pixels : int, optional
            The minimum area in pixels for the detection. Defaults to 5.
        deblend_nlevels : int, optional
            The number of levels for the deblending. Defaults to 32.
        deblend_contrast : float, optional
            The contrast for the deblending. Defaults to 0.003.
        fwhm_estimate_pixel : float, optional
            Estimated FWHM of stars in pixels. Defaults to 5.0.
        n_iterations : int, optional
            The number of iterations for the PSF photometry. Defaults to 0.
        visualize : bool, optional
            Whether to visualize the PSF photometry. Defaults to True.
        verbose : bool, optional
            Whether to print verbose output. Defaults to True.
        save : bool, optional
            Whether to save the PSF photometry. Defaults to True.
        apply_aperture_correction : bool, optional
            Whether to apply aperture correction to the PSF photometry. Defaults to True.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        result : Table
            The PSF photometry results.
        """
        from photutils.psf import PSFPhotometry, SourceGrouper
        from astropy.nddata import NDData
        from astropy.modeling.fitting import LevMarLSQFitter
        from photutils.utils import calc_total_error
        from collections import defaultdict
        import numpy as np

        # Background subtraction
        if target_bkg is not None:
            target_img_sub = self.background.subtract_background(
                target_img=target_img,
                target_bkg=target_bkg,
                save=False,
                overwrite=False,
                visualize = False
            )
        else:
            target_img_sub = target_img
        
        # Background RMS map
        if target_bkgrms is None:
            # Use sigma-clipped std if no error map available
            target_bkgrms, _, _ = self.errormap.calculate_errormap_from_image(
                target_img=target_img,
                target_mask=None,
                mode='sep',
                errormap_type='bkgrms',
                save=False,
                visualize=visualize,
                **kwargs
            )
        
        # Errormap calculation 
        error = calc_total_error(data=target_img_sub.data,
                                bkg_error=target_bkgrms.data,
                                effective_gain=target_img_sub.egain)
        
        # Invalid pixel mask
        invalid_mask = self.masking.mask_invalidpixel(target_img = target_img_sub)
        if target_mask is not None:
            target_mask.combine_mask(invalid_mask.data)
        else:
            target_mask = invalid_mask
        
        if sources is None:
            #TODO CHANGE THIS PART TO FIND PEAKS RATHER THAN FIND SOURCES
            sources, _ = self.extract_sources(
                target_img=target_img_sub,
                target_bkg=None,
                target_bkgrms=target_bkgrms,
                target_mask=target_mask,
                detection_sigma=detection_sigma,
                minarea_pixels= minarea_pixels,
                deblend_nlevels=deblend_nlevels,
                deblend_contrast=deblend_contrast,
                fwhm_estimate_pixel=fwhm_estimate_pixel,
                visualize=visualize,
                save=False,
                verbose=verbose
            )
            sources = self._match_catalog_format(sources)
        else:
            sources = self._match_catalog_format(sources)
            if 'X_IMAGE' not in sources.colnames or 'Y_IMAGE' not in sources.colnames or 'segment_flux' not in sources.colnames:
                raise ValueError("Sources table must contain 'X_IMAGE' and 'Y_IMAGE' and 'segment_flux' columns.")
            if apply_aperture_correction and 'FLUX_APER' not in sources.colnames:
                raise ValueError("Sources table must contain 'FLUX_APER' column for aperture correction.")
        
        # Check if sources are within the image bounds
        def cut_edge_sources(sources, edge_buffer=100):
            edge_buffer = int(100)
            image_edge_mask = (
                (sources['X_IMAGE'] > edge_buffer) & 
                (sources['X_IMAGE'] < target_img.naxis1 - edge_buffer) &
                (sources['Y_IMAGE'] > edge_buffer) & 
                (sources['Y_IMAGE'] < target_img.naxis2 - edge_buffer)
            )
            # mask edge region of the invalidemask
            from scipy.ndimage import binary_dilation
            dilated_mask = binary_dilation(invalid_mask.data, iterations=100)
            # Convert source coordinates to integer pixel indices
            y_pix = np.clip(np.round(sources['Y_IMAGE']).astype(int), 0, invalid_mask.data.shape[0] - 1)
            x_pix = np.clip(np.round(sources['X_IMAGE']).astype(int), 0, invalid_mask.data.shape[1] - 1)

            # Identify sources landing on or near invalid pixels
            invalid_edge_mask = dilated_mask[y_pix, x_pix]
            edge_mask = image_edge_mask & ~invalid_edge_mask
            
            self.helper.print(f'{len(sources[~edge_mask])} sources filtered out due to edge mask.', verbose)
            filtered_sources = sources[edge_mask]
            return filtered_sources
        
        filtered_sources = cut_edge_sources(sources, edge_buffer=100)

        # PSF Photometry
        self.helper.print('Starting PSF photometry...', verbose)
        fitter = LevMarLSQFitter()
        group_maker = SourceGrouper(min_separation=3.0 * fwhm_estimate_pixel)
        def run_psf_photometry(data, sources):
            # Prepare full-size model and residual images
            sigma = target_img.info.SKYSIG if target_img.info.SKYSIG is not None else 10
            model_image = np.random.normal(loc = 0, scale = sigma, size = data.shape)
            residual_image = np.random.normal(loc = 0, scale = sigma, size = data.shape)
            if isinstance(epsf_model_dict, EPSFModel):
                self.helper.print('Using global EPSF model for photometry.', verbose)
                epsf_model = epsf_model_dict

                psf_phot = PSFPhotometry(
                    psf_model=epsf_model,
                    fit_shape=(int(epsf_model.x_0*2)|1, int(epsf_model.y_0*2)|1),
                    grouper=group_maker,
                    fitter=fitter,
                    xy_bounds = (3,3),
                    progress_bar=verbose
                )
                phot = psf_phot(data = data, mask = None, error = error, init_params = sources)
                self.helper.print('Start generating model images...', verbose)
                model_image += psf_phot.make_model_image(shape=data.shape, psf_shape=(int(epsf_model.x_0*2)|1, int(epsf_model.y_0*2)|1), include_localbkg = False)
                self.helper.print('Start generating residual images...', verbose)
                residual_image += psf_phot.make_residual_image(data=data, psf_shape=(int(epsf_model.x_0*2)|1, int(epsf_model.y_0*2)|1), include_localbkg = False)
                
                result = sources.copy()
                for col in phot.colnames:
                    result[col] = phot[col]

            elif isinstance(epsf_model_dict, dict):
                self.helper.print('Using grid-based EPSF models for photometry.', verbose)
                h, w = data.shape
                num_grids = int(np.sqrt(len(epsf_model_dict)))
                x_step = w // num_grids
                y_step = h // num_grids
            
                result = Table()
                for (i, j), epsf_model in epsf_model_dict.items():
                    self.helper.print(f'Processing grid ({i},{j}) with EPSF model shape {epsf_model.data.shape}', verbose)
                    x_min, x_max = j * x_step, (j + 1) * x_step
                    y_min, y_max = i * y_step, (i + 1) * y_step

                    in_cell = (sources['X_IMAGE'] >= x_min) & (sources['X_IMAGE'] < x_max) & \
                            (sources['Y_IMAGE'] >= y_min) & (sources['Y_IMAGE'] < y_max)
                    sources_grid = sources[in_cell]
                    if len(sources_grid) == 0:
                        continue
                    psf_phot = PSFPhotometry(
                        psf_model=epsf_model,
                        fit_shape=(int(epsf_model.x_0*2)|1, int(epsf_model.y_0*2)|1),
                        grouper=group_maker,
                        fitter=fitter,
                        xy_bounds = (3,3),
                        progress_bar=verbose
                    )
                    phot_tbl = psf_phot(data =  data, mask = None, error = error, init_params = sources_grid)
                    result = vstack([result, hstack([sources_grid, phot_tbl])])
                    
                    # Generate model and residual for *this grid only*
                    self.helper.print('Start generating model images...:', verbose)
                    model_patch = psf_phot.make_model_image(shape=data.shape, psf_shape=(int(epsf_model.x_0*2)|1, int(epsf_model.y_0*2)|1), include_localbkg = False)
                    self.helper.print('Start generating residual images...:', verbose)
                    residual_patch = psf_phot.make_residual_image(data=data, psf_shape=(int(epsf_model.x_0*2)|1, int(epsf_model.y_0*2)|1), include_localbkg = False)

                    # Fill only the current grid region in model and residual
                    model_image[y_min:y_max, x_min:x_max] += model_patch[y_min:y_max, x_min:x_max]
                    residual_image[y_min:y_max, x_min:x_max] += residual_patch[y_min:y_max, x_min:x_max]

            else:
                raise ValueError("epsf_model_dict must be either an EPSFModel or a dict of EPSFModels.")
            
            return result, model_image, residual_image
        result, model_image, residual_image = run_psf_photometry(data=target_img_sub.data, sources = filtered_sources)
        
        # Apply aperture correction if requested
        aper_correction = False
        if apply_aperture_correction and 'FLUX_APER' in result.colnames:
            self.helper.print('Applying aperture correction...', verbose)
            
            # Filter high-quality stars for aperture correction calculation
            
            good_stars = self.select_psfstars(
                target_catalog = result,
                image_shape = target_img.data.shape,
                num_grids = 1,
                max_per_grid = 500,
                snr_lower = 30,
                verbose = True,
                visualize = False)
        
            #good_stars_mask = (result['QFIT_PSF'] < 0.05) & (result['FLUX_APER'] > 0) & (result['FLUX_PSF'] > 0)
            if len(good_stars) > 10:
                from numpy.polynomial.polynomial import Polynomial

                #good_stars = result[good_stars_mask]
                
                # Calculate correction factor as median ratio of aperture flux to PSF flux
                good_stars['MAG_PSF'] = -2.5 * np.log10(good_stars['flux_fit'])
                ap_corrections = good_stars['MAG_APER'] - good_stars['MAG_PSF']
                fit_poly = Polynomial.fit(good_stars['MAG_PSF'], ap_corrections, deg=3)
                fit_poly = fit_poly.convert()
                coefs = fit_poly.coef
                #ap_corrections = good_stars['FLUX_APER'] / good_stars['FLUX_PSF']
                
                # Use sigma-clipped median for robust correction factor
                from astropy.stats import sigma_clipped_stats
                mean, median, std = sigma_clipped_stats(ap_corrections, sigma=3.0)
                ap_corr_factor = median
                
                # Apply correction to PSF magnitude and error
                delta_mag_per_star = fit_poly(-2.5 * np.log10(result['flux_fit']))

                flux_scale = 10 ** (-0.4 * delta_mag_per_star)
                aper_correction = True
            else:
                aper_correction = False
                self.helper.print('No aperture correction applied due to lack of the good stars', verbose)
        
        # Visualize residual with original image
        def visualize_residual(target_img_sub, model_image, residual_image, downsample_factor=6, visualize_apercorr: bool = True):
            target_img_sub_smalldata = target_img_sub.data[::downsample_factor, ::downsample_factor]
            model_img_smalldata = model_image[::downsample_factor, ::downsample_factor]
            residual_image_smalldata = residual_image[::downsample_factor, ::downsample_factor]
            fig, axes = plt.subplots(1,3, figsize=(24, 8), dpi = 300)
            yticks = np.linspace(0, target_img_sub_smalldata.shape[0]-1, num=6, dtype=int)
            xticks = np.linspace(0, target_img_sub_smalldata.shape[1]-1, num=6, dtype=int)
            norm = simple_norm(target_img_sub_smalldata, stretch='sqrt', percent=99)
            ax = axes[0]
            ax.imshow(target_img_sub_smalldata, origin='lower', cmap='viridis', norm=norm)
            ax.set_title("Original Image")
            plt.colorbar(ax.imshow(target_img_sub_smalldata, origin='lower', cmap='viridis', norm=norm), ax=ax, fraction=0.046, pad=0.04)
            ax.set_yticks(yticks)
            ax.set_yticklabels((yticks * downsample_factor).astype(int))
            ax.set_xticks(xticks)
            ax.set_xticklabels((xticks * downsample_factor).astype(int))
            ax = axes[1]
            norm = simple_norm(model_img_smalldata, stretch='sqrt', percent=99)
            ax.imshow(model_img_smalldata, origin='lower', cmap='viridis', norm=norm)
            ax.set_title("Model Image")
            plt.colorbar(ax.imshow(model_img_smalldata, origin='lower', cmap='viridis', norm=norm), ax=ax, fraction=0.046, pad=0.04)
            ax.set_yticks(yticks)
            ax.set_yticklabels((yticks * downsample_factor).astype(int))
            ax.set_xticks(xticks)
            ax.set_xticklabels((xticks * downsample_factor).astype(int))
            ax = axes[2]
            norm = simple_norm(residual_image_smalldata, stretch='sqrt', percent=99)
            ax.imshow(residual_image_smalldata, origin='lower', cmap='viridis', norm=norm)
            ax.set_title("Residual Image")
            plt.colorbar(ax.imshow(residual_image_smalldata, origin='lower', cmap='viridis', norm=norm), ax=ax, fraction=0.046, pad=0.04)
            ax.set_yticks(yticks)
            ax.set_yticklabels((yticks * downsample_factor).astype(int))
            ax.set_xticks(xticks)
            ax.set_xticklabels((xticks * downsample_factor).astype(int))
            plt.tight_layout()
            plt.show()
            
            # If aperture correction was applied, also visualize the correction factor
            if visualize_apercorr and apply_aperture_correction and 'MAG_PSF_CORR' in result.colnames:
                fig, ax = plt.subplots(figsize=(10, 8))
                x = np.linspace(np.min(good_stars['MAG_PSF']) - 0.5, np.max(good_stars['MAG_PSF']) + 0.5, 100)
                y = fit_poly(x)
                ax.scatter(good_stars['MAG_PSF'], ap_corrections, alpha=0.7, label=f'Stars used for correction')
                ax.plot(x, y, color='orange', label='Aperture Correction Fit')
                #ax.axhline(ap_corr_factor, color='r', linestyle='-', label=f'Correction factor: {ap_corr_factor:.4f}')
                #ax.axhline(ap_corr_factor + std, color='r', linestyle='--', alpha=0.5)
                #ax.axhline(ap_corr_factor - std, color='r', linestyle='--', alpha=0.5, label='%.3f±%.3f'%(ap_corr_factor,std))
                ax.set_xlabel('PSF Magnitude')
                ax.set_ylabel('MAG_APER - MAG_PSF')
                ax.set_title('Aperture Correction')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(ap_corr_factor -1, ap_corr_factor +1)
                #ax.set_xlim(np.min(good_stars['MAG_PSF']) - 0.5, np.max(good_stars['MAG_PSF']) + 0.5)
                plt.tight_layout()
                plt.show()
        
        if visualize:
            visualize_residual(target_img_sub, model_image, residual_image)
            
        result_final = result
        model_final = model_image
        residual_final = residual_image
        residual_img = target_img_sub.copy()
        if n_iterations > 0:
            result_final['ITERATION'] = 0
            residual_img.data = residual_final
            for i in range(n_iterations):
                self.helper.print(f'Running {i+1}/{n_iterations} iterations of PSF photometry...', verbose)

                sources_iter, _ = self.extract_sources(
                    target_img=residual_img,
                    target_bkg=None,
                    target_bkgrms=target_bkgrms,
                    target_mask=target_mask,
                    detection_sigma=detection_sigma,
                    minarea_pixels= minarea_pixels,
                    deblend_nlevels=deblend_nlevels,
                    deblend_contrast=deblend_contrast,
                    fwhm_estimate_pixel=fwhm_estimate_pixel,
                    visualize=False,
                    save=False,
                    verbose=verbose
                )
                sources_iter = self._match_catalog_format(sources_iter)
                # Cross-match with previous iteration results
                previous_coord = SkyCoord(result_final['X_WORLD'], result_final['Y_WORLD'], unit='deg', frame='icrs')
                new_coord = SkyCoord(sources_iter['X_WORLD'], sources_iter['Y_WORLD'], unit='deg', frame='icrs')
                separation_in_arcsec = 2* np.mean(target_img.pixelscale)*fwhm_estimate_pixel
                new_matched, prev_matched, new_unmatched = self.helper.cross_match(new_coord, previous_coord, max_distance_second = separation_in_arcsec)
                sources_iter = sources_iter[new_unmatched]
                self.helper.print(f'{len(new_matched)} are filtered out due to cross-match with small separation.', verbose)
                
                # Check if sources are within the image bounds
                filtered_sources_iter = cut_edge_sources(sources_iter, edge_buffer=100)
                self.helper.print(f'{len(filtered_sources_iter)} source are detected in iteration {i+1}.', verbose)
                if visualize:
                    self._visualize_objects(residual_img, filtered_sources_iter, size=1000)
                
                result_iter, model_iter, residual_iter = run_psf_photometry(residual_img.data, filtered_sources_iter)
                result_iter['ITERATION'] = i + 1
                result_final = vstack([result_final, result_iter])
                model_final += model_iter
                residual_final = residual_iter
                residual_img.data = residual_final
                
                if visualize:
                    visualize_residual(residual_img, model_iter, residual_iter, visualize_apercorr = False)
        
        result_final = self._match_catalog_format(result_final)
        # Add standard PSF photometry columns
        result_final['FLUX_PSF'] = result_final['flux_fit']
        result_final['FLUXERR_PSF'] = result_final['flux_err']
        result_final['MAG_PSF'] = -2.5 * np.log10(result_final['flux_fit'])
        result_final['MAGERR_PSF'] = 2.5/np.log(10) * (result_final['flux_err']/result_final['flux_fit'])
        result_final['X_IMAGE_PSF'] = result_final['x_fit']
        result_final['Y_IMAGE_PSF'] = result_final['y_fit']
        wcs = target_img_sub.wcs
        if wcs:
            x_arr = result_final['X_IMAGE_PSF']
            y_arr = result_final['Y_IMAGE_PSF']
            skycoord = pixel_to_skycoord(x_arr, y_arr, wcs)
            result_final['X_WORLD_PSF'] = skycoord.ra.value
            result_final['Y_WORLD_PSF'] = skycoord.dec.value
        result_final['QFIT_PSF'] = result_final['qfit']
        result_final['CFIT_PSF'] = result_final['cfit']
        #Apply aperture correction if requested
        if apply_aperture_correction:
                delta_mag_per_star = fit_poly((result_final['MAG_PSF']))
                flux_scale = 10 ** (-0.4 * delta_mag_per_star)
                result_final['MAG_PSF_CORR'] = result_final['MAG_PSF'] + delta_mag_per_star
                result_final['MAGERR_PSF_CORR'] = result_final['MAGERR_PSF']  # optionally add scatter term
                result_final['FLUX_PSF_CORR'] = result_final['FLUX_PSF'] * flux_scale
                result_final['FLUXERR_PSF_CORR'] = result_final['FLUXERR_PSF'] * flux_scale
                result_final['AP_CORR_FACTOR'] = delta_mag_per_star
        else:
            result_final['MAG_PSF_CORR'] = result_final['MAG_PSF']
            result_final['MAGERR_PSF_CORR'] = result_final['MAGERR_PSF']
            result_final['FLUX_PSF_CORR'] = result_final['FLUX_PSF']
            result_final['FLUXERR_PSF_CORR'] = result_final['FLUXERR_PSF']
            result_final['AP_CORR_FACTOR'] = 0.0
        
        if save:
            fits.writeto(target_img.savepath.savepath.with_suffix('.residual'), residual_final, overwrite=True)
            fits.writeto(target_img.savepath.savepath.with_suffix('.model'), model_final, overwrite=True)
            result_final.write(str(target_img.savepath.psfcatalogpath), format = 'ascii', overwrite=True)
            
        return result_final

    def _visualize_objects(self, 
                           target_img: Union[ScienceImage, ReferenceImage],
                           objects: Table,
                           size: int = 1000):
        """
        Visualize detected sources on the full and zoomed image using circles.
        """
        data = target_img.data
        h, w = data.shape

        # Define zoom-in center box
        center_x, center_y = w // 2, h // 2
        half_box = size // 2
        x_min, x_max = max(0, center_x - half_box), min(w, center_x + half_box)
        y_min, y_max = max(0, center_y - half_box), min(h, center_y + half_box)

        cropped_data = data[y_min:y_max, x_min:x_max]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Full frame
        m, s = np.mean(data), np.std(data)
        im0 = axes[0].imshow(data, cmap='gray', origin='lower', vmin=m - s, vmax=m + s)
        axes[0].set_title("Full Background-Subtracted Image")
        plt.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.04)

        # Mark zoom box
        zoom_box = Rectangle((x_min, y_min), size, size, linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(zoom_box)

        # Cropped region
        m_crop, s_crop = np.mean(cropped_data), np.std(cropped_data)
        im1 = axes[1].imshow(cropped_data, cmap='gray', origin='lower', vmin=m_crop - s_crop, vmax=m_crop + s_crop)
        axes[1].set_title(f"{size}×{size} Center Region")

        # Get best x/y key names
        if 'X_IMAGE' in objects.colnames and 'Y_IMAGE' in objects.colnames:
            x_key, y_key = 'X_IMAGE', 'Y_IMAGE'
        else:
            raise ValueError("No recognized x/y keys found in catalog")

        for obj in objects:
            x, y = float(obj[x_key]), float(obj[y_key])
            if x_min <= x <= x_max and y_min <= y <= y_max:
                circ = Circle((x - x_min, y - y_min), radius=size * 0.005, edgecolor='red', facecolor='none', lw=1.5)
                axes[1].add_patch(circ)

        plt.tight_layout()
        plt.show()
        plt.close()
        
    def _visualize_epsf_stars_and_model_grid(self, epsf_model_dict, epsf_stars_dict):
        max_i = max(key[0] for key in epsf_model_dict.keys()) + 1
        max_j = max(key[1] for key in epsf_model_dict.keys()) + 1

        fig, axes = plt.subplots(max_i, max_j, figsize=(3 * max_j, 3 * max_i), squeeze=False)

        for (i, j), epsf_model in epsf_model_dict.items():
            epsf_stars = epsf_stars_dict.get((i, j), [])
            
            # Flip vertical index to put (0,0) at bottom-left
            flipped_i = max_i - 1 - i
            ax = axes[flipped_i][j]

            norm = simple_norm(epsf_model.data, stretch='sqrt', percent=99)
            ax.imshow(epsf_model.data, origin='lower', cmap='viridis', norm=norm)
            ax.set_title(f"Grid ({i},{j})\nN={len(epsf_stars)}")
            ax.axis('off')

        plt.suptitle("EPSF Models per Grid", fontsize=16)
        plt.tight_layout()
        plt.show()

    def _visualize_epsf_stars_and_model(self, epsf_model, epsf_stars):
        nstars = len(epsf_stars)
        nrows = min(4, nstars)
        ncols = int(np.ceil(nstars / nrows))
        fig1, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes = np.array(axes).reshape(-1)

        for i, star in enumerate(epsf_stars):
            norm = simple_norm(star.data, stretch='sqrt', percent=99)
            axes[i].imshow(star.data, origin='lower', cmap='viridis', norm=norm)
            axes[i].set_title(f"Star {i+1}")
            axes[i].axis('off')

        for i in range(nstars, len(axes)):
            axes[i].axis('off')

        fig1.suptitle("Stars Used for EPSF Modeling", fontsize=16)
        plt.tight_layout()
        plt.show()

        fig2, ax = plt.subplots(1, 1, figsize=(6, 6))
        norm_epsf = simple_norm(epsf_model.data, stretch='sqrt', percent=99)
        im = ax.imshow(epsf_model.data, origin='lower', cmap='viridis', norm=norm_epsf)
        ax.set_title("Final EPSF Model")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()
        
    def _visualize_epsf_residuals(self, epsf_model, epsf_stars, num_show: int = 9):


        fitter = LevMarLSQFitter()  

        n = min(len(epsf_stars), num_show)
        stars = epsf_stars[:n]
        ncols = 3
        model = epsf_model.copy()
        fig, axes = plt.subplots(n, ncols, figsize=(3 * ncols, 3 * n), squeeze=False)

        for i in range(n):
            data = stars[i].data
            ny, nx = data.shape
            model.flux = np.sum(data)
            model.x_0 = nx / 2
            model.y_0 = ny / 2
            y, x = np.mgrid[:ny, :nx]

            # Use EPSFModel like a function
            fit_model = fitter(model, x, y, data)
            model_vals = fit_model(x,y)

            # Residual
            residual = data - model_vals

            # Common normalization for data and model
            norm = simple_norm(data, stretch='log', percent=99.0)

            # Separate linear scale for residual (centered at 0)
            vres = np.nanpercentile(np.abs(residual), 99)  # robust vmax
            res_norm = plt.Normalize(vmin=-vres, vmax=vres)

            # Data
            axes[i, 0].imshow(data, origin='lower', cmap='gray', norm=norm)
            axes[i, 0].set_title('Data')

            # Model
            axes[i, 1].imshow(model_vals, origin='lower', cmap='gray', norm=norm)
            axes[i, 1].set_title('Model')

            # Residual (RdBu diverging colormap centered at 0)
            im_res = axes[i, 2].imshow(residual, origin='lower', cmap='RdBu_r', norm=res_norm)
            axes[i, 2].set_title('Residual')

            # Optional: Add colorbar to residual
            cbar = plt.colorbar(im_res, ax=axes[i, 2], fraction=0.046, pad=0.04)
            cbar.set_label('Flux Difference')


            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.show()
        
    def _visualize_epsf_residuals_grid(self, epsf_model_dict, epsf_stars_dict, num_show_each_grid: int = 3):
        """
        Visualize residuals of EPSF fits for each grid.

        Parameters
        ----------
        epsf_dict : dict
            Dictionary with keys (i,j) and values (epsf_model, fitted_stars)
        num_grids : int
            Number of grid divisions (assumes square layout)
        num_show : int
            Number of stars to visualize per grid
        """
        import matplotlib.pyplot as plt
        from astropy.visualization import simple_norm
        from astropy.modeling.fitting import LevMarLSQFitter
        from photutils.psf import EPSFModel
        import numpy as np

        fitter = LevMarLSQFitter()
        models_dict_copy = epsf_model_dict.copy()
        for (i, j), epsf_model in models_dict_copy.items():
            epsf_stars = epsf_stars_dict[(i, j)]
            n = min(len(epsf_stars), num_show_each_grid)
            if n == 0:
                continue

            stars = epsf_stars[:n]

            # Layout: 5 rows x 2 stars per row = 10 total
            ncols_stars = 2
            nrows_stars = (n + ncols_stars - 1) // ncols_stars  # ceil(n / 2), max 5
            nrows_stars = min(nrows_stars, np.ceil(num_show_each_grid/ncols_stars))

            fig, axes = plt.subplots(nrows_stars, ncols_stars * 3,
                                    figsize=(3 * ncols_stars * 3, 3 * nrows_stars),
                                    squeeze=False)
            fig.suptitle(f"EPSF Residuals - Grid ({i},{j})", fontsize=14)

            for idx in range(n):
                data = stars[idx].data
                ny, nx = data.shape

                epsf_model.flux = np.sum(data)
                epsf_model.x_0 = nx / 2
                epsf_model.y_0 = ny / 2

                y, x = np.mgrid[:ny, :nx]
                fit_model = fitter(epsf_model, x, y, data)
                model_vals = fit_model(x, y)
                residual = data - model_vals

                norm = simple_norm(data, stretch='log', percent=99.0)
                vres = np.nanpercentile(np.abs(residual), 99)
                res_norm = plt.Normalize(vmin=-vres, vmax=vres)

                row = idx // ncols_stars
                star_col = idx % ncols_stars

                for k, img, title, norm_ in zip(
                    range(3), [data, model_vals, residual], ['Data', 'Model', 'Residual'], [norm, norm, res_norm]
                ):
                    ax = axes[row, star_col * 3 + k]
                    im = ax.imshow(img, origin='lower', cmap='gray' if k < 2 else 'RdBu_r', norm=norm_)
                    ax.set_title(title)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if k == 2:
                        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label('Flux Diff')

            plt.tight_layout()
            plt.show()


    def _match_catalog_format(self, catalog: Table):
        import numpy as np

        column_set = dict(
            xcentroid='X_IMAGE',
            ycentroid='Y_IMAGE',
            segment_flux='FLUX_ISO',
            segment_flux_err='FLUXERR_ISO'
        )

        # Check if any key or value exists in catalog
        if any(col in catalog.colnames for col in column_set.keys()) or any(col in catalog.colnames for col in column_set.values()):
            for new_key, old_key in column_set.items():
                if new_key in catalog.colnames:
                    # Already exists, skip
                    continue
                elif old_key in catalog.colnames:
                    catalog[new_key] = catalog[old_key]
                else:
                    catalog[new_key] = np.full(len(catalog), np.nan)
        catalog = Table({name: catalog[name].value for name in catalog.colnames})
        return catalog
