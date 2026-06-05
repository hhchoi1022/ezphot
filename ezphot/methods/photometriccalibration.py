#%%
import inspect
from typing import Union, List

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.table import Table, vstack
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting
import astropy.units as u
import matplotlib.colors as mcolors


from ezphot.helper import Helper
from ezphot.imageobjects import ScienceImage, ReferenceImage, CalibrationImage
from ezphot.dataobjects import Catalog
from ezphot.skycatalog import SkyCatalogUtility
from ezphot.utils import *

#%%

class PhotometricCalibration:
    """
    Method class to perform photometric calibration of astronomical images.
    
    This class provides methods 
    
    1. Photometric calibration using reference catalogs.
    
    2. Photometric calibration using reference images.
    
    3. Photometric calibration using reference images and catalogs.
    """
    
    def __init__(self):
        self.helper = Helper()
        self.catalogutils = SkyCatalogUtility()

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

    def photometric_calibration(self,
                                target_img: Union[ScienceImage, ReferenceImage, CalibrationImage],
                                target_catalog: Catalog,
                                catalog_type: str = 'GAIAXP',
                                catalog_version: str = 'v1',
                                max_distance_second: float = 2.5,
                                min_number_of_sources: int = 10,
                                calculate_color_terms: bool = True,
                                calculate_mag_terms: bool = True,
                                
                                # Selection parameters
                                mag_lower: float = 12,
                                mag_upper: float = 16,
                                snr_lower: float = 10,
                                snr_upper: float = 150,
                                dynamic_mag_range: bool = False,
                                classstar_lower: float = 0.8,
                                elongation_upper: float = 1.7,
                                elongation_sigma: float = 5,
                                fwhm_lower_arcsec: float = 1,
                                fwhm_upper_arcsec: float = 10,
                                fwhm_sigma: float = 5,
                                flag_upper: int = 1,
                                maskflag_upper: int = 1,
                                ra_deg: float = None,
                                dec_deg: float = None,
                                radius_arcsec: float = 1500,    
                                inner_fraction: float = 0.9, # Fraction of the images. If ra, dec, radius_arcsec is given, inner_fraction is ignored.
                                isolation_radius_arcsec: float = 10.0,
                                
                                magnitude_key: str = 'MAG_AUTO',
                                magnitudeerr_key: str = 'MAGERR_AUTO',
                                fwhm_key: str = 'FWHM_WORLD',
                                ra_key: str = 'X_WORLD',
                                dec_key: str = 'Y_WORLD',
                                classstar_key: str = 'CLASS_STAR',
                                elongation_key: str = 'ELONGATION',
                                flag_key: str = 'FLAGS',
                                maskflag_key: str = 'IMAFLAGS_ISO',

                                # Other parameters
                                update_header: bool = True,
                                save: bool = True,
                                verbose: bool = True,
                                visualize: bool = True,
                                save_fig: bool = False,
                                save_refcat: bool = True):

        """
        Perform photometric calibration of astronomical images.
        
        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage or CalibrationImage
            The target image to perform photometric calibration on.
        target_catalog : Catalog
            The catalog to use for photometric calibration.
        catalog_type : str, optional
            The type of catalog to use for photometric calibration.
        max_distance_second : float, optional
            The maximum distance in arcseconds to search for secondaries.
        calculate_color_terms : bool, optional
            Whether to calculate color terms.
        calculate_mag_terms : bool, optional
            Whether to calculate magnitude terms.
        mag_lower : float, optional
            The lower magnitude limit for the reference star selection.
        mag_upper : float, optional
            The upper magnitude limit for the reference star selection.
        classstar_lower : float, optional
            The lower class star limit for the reference star selection.
        elongation_upper : float, optional
            The upper elongation limit for the reference star selection.
        elongation_sigma : float, optional
            The sigma of the elongation for the reference star selection.
        fwhm_lower : float, optional
            The lower FWHM limit for the reference star selection.
        fwhm_upper : float, optional
            The upper FWHM limit for the reference star selection.
        fwhm_sigma : float, optional
            The sigma of the FWHM for the reference star selection.
        flag_upper : int, optional
            The upper flag limit for the reference star selection.
        maskflag_upper : int, optional
            The upper mask flag limit for the reference star selection.
        inner_fraction : float, optional
            The inner fraction of the image to use for the reference star selection.
        isolation_radius : float, optional
            The isolation radius for the reference star selection.
        magnitude_key : str, optional
            The key of the magnitude column in the catalog.
        fwhm_key : str, optional
            The key of the FWHM column in the catalog.
        x_key : str, optional
            The key of the X column in the catalog.
        y_key : str, optional
            The key of the Y column in the catalog.
        classstar_key : str, optional
            The key of the class star column in the catalog.
        elongation_key : str, optional
            The key of the elongation column in the catalog.
        flag_key : str, optional
            The key of the flag column in the catalog.
        maskflag_key : str, optional
            The key of the mask flag column in the catalog.
        save : bool, optional
            Whether to save the catalog.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the photometric calibration.
        save_fig : bool, optional
            Whether to save the figure.
        save_refcat : bool, optional
            Whether to save the reference catalog.
            
        Returns
        -------
        target_img: ScienceImage or ReferenceImage or CalibrationImage
            The target image with photometric calibration.
        target_catalog: Catalog
            The updated catalog with photometric calibration.
        filtered_catalog: Catalog
            The reference catalog for photometric calibration.
        update_kwargs: dict
            The dictionary of updated keywords.
        """
        update_kwargs = dict()
        catalogs = self.catalogutils.get_catalogs(
            ra = target_img.ra,
            dec = target_img.dec,
            fov_ra = target_img.fovx,
            fov_dec = target_img.fovy,
            objname = target_img.objname,
            catalog_type = catalog_type,
            catalog_version = catalog_version,
            verbose = verbose
        )
        if len(catalogs) == 0:
            raise ValueError("No catalogs found in the given coordinates.")
        else:
            all_references = Table()
            for catalog in catalogs:
                data = self.catalogutils.select_reference_sources(catalog = catalog, mag_lower = 10, mag_upper = 20)[0]
                #data = catalog.data
                all_references = vstack([all_references, data])
                
        # Calculate Zero Point
        mag_key_ref = '%s_mag'%(target_img.filter)        
        zp_key_ref = magnitude_key.replace('MAG_', 'ZP_')
        all_catalog_data = target_catalog.data
        catalog_coord = SkyCoord(all_catalog_data['X_WORLD'], all_catalog_data['Y_WORLD'], unit='deg')
        reference_coord = SkyCoord(all_references['ra'], all_references['dec'], unit='deg')
        obj_indices, ref_indices, unmatched_obj_indices = self.helper.cross_match(catalog_coord, reference_coord, max_distance_second = max_distance_second)
        matched_obj_all = all_catalog_data[obj_indices]
        matched_ref_all = all_references[ref_indices]
        matched_obj_all['MAG_REF'] = matched_ref_all[mag_key_ref]
        zp_all = matched_ref_all[mag_key_ref] - matched_obj_all[magnitude_key]
        matched_obj_all['ZP_REF'] = zp_all
        matched_catalog = target_catalog.copy()
        matched_catalog.data = matched_obj_all
        zp_median_all = np.ma.median(zp_all)
        zp_err_all = np.ma.std(zp_all)

        if dynamic_mag_range:
            # Configuration sets for retries
            bin_width_options = [0.3, 0.4, 0.5]
            sigma_clip_options = [3, 5, 7]
            closest_diff = float("inf")  # Track the difference closest to 1.7

            closest_result = None
            # Loop through combinations of bin_width and sigma_clip
            for bw in bin_width_options:
                for sc in sigma_clip_options:
                    try:
                        self.helper.print(f"[INFO] bin_width={bw}, sigma_clip={sc}", verbose)
                        mag_min, mag_max, zp_rough, zp_err_rough, saturation_level = self.determine_reference_mag_range(
                            target_catalog=matched_catalog,
                            bin_width=bw,
                            sigma_clip=sc,
                            magnitude_key='MAG_REF',
                            zp_key='ZP_REF',
                            fwhm_key='FWHM_IMAGE',
                            verbose=verbose,
                            visualize=False,
                            save_fig=save_fig
                        )

                        # Save current try
                        try_set = dict(
                            bin_width=bw,
                            sigma_clip=sc,
                            mag_min=mag_min,
                            mag_max=mag_max,
                            zp_rough=zp_rough,
                            zp_err_rough=zp_err_rough,
                            saturation_level=saturation_level
                        )
                        diff = abs(mag_max - mag_min)

                        if (mag_min > mag_upper) or (mag_max < mag_lower) or (diff <= 1.0) or (diff >= 3):
                            continue
                        
                        if diff < closest_diff:
                            closest_diff = diff
                            closest_result = try_set
                        
                    except Exception as e:
                        pass
            # Finalize result
            if closest_result is not None:
                self.helper.print(f"[INFO] Dynamic mag range found: {closest_result['mag_min']:.2f} < mag < {closest_result['mag_max']:.2f}", verbose)
                mag_min = closest_result['mag_min']
                mag_max = closest_result['mag_max']
                zp_median_all = closest_result['zp_rough']
                zp_err_all = closest_result['zp_err_rough']
                saturation_level = closest_result['saturation_level']
                update_kwargs['SATURATE'] = (saturation_level, "Saturation level in ADU")
            else:
                self.helper.print(f"[WARN] No valid result found from dynamic mag range. Using default mag range.", verbose)
                mag_min = mag_lower
                mag_max = mag_upper
                zp_median_all = np.ma.median(zp_all)
                zp_err_all = np.ma.std(zp_all)
        else:
            zp_median_all = np.ma.median(zp_all)
            zp_err_all = np.ma.std(zp_all)
            mag_min = mag_lower
            mag_max = mag_upper
        magnitude_sky_key = magnitude_key.replace('MAG_', 'MAGSKY_')
        target_catalog.data[magnitude_sky_key] = target_catalog.data[magnitude_key] + zp_median_all

        # Filter out reference sources
        for i in range(5):
            filtered_catalog, _, target_seeing = self.select_stars(
                target_catalog = target_catalog,
                mag_lower = mag_min,
                mag_upper = mag_max,
                snr_lower = snr_lower,
                snr_upper = snr_upper,
                classstar_lower = classstar_lower,
                elongation_upper = elongation_upper,
                elongation_sigma = elongation_sigma,
                fwhm_lower_arcsec = fwhm_lower_arcsec,
                fwhm_upper_arcsec = fwhm_upper_arcsec,
                fwhm_sigma = fwhm_sigma,
                flag_upper = flag_upper,
                maskflag_upper = maskflag_upper,
                ra_deg = ra_deg,
                dec_deg = dec_deg,
                radius_arcsec = radius_arcsec,
                inner_fraction = inner_fraction,
                isolation_radius_arcsec = isolation_radius_arcsec,

                save = False,
                verbose = verbose,
                visualize = visualize,
                save_fig = save_fig, 

                magnitude_key = magnitude_sky_key,
                magnitudeerr_key = magnitudeerr_key,
                fwhm_key = fwhm_key,
                ra_key = ra_key,
                dec_key = dec_key,
                classstar_key = classstar_key,
                elongation_key = elongation_key,
                flag_key = flag_key,
                maskflag_key = maskflag_key,
                )
            filtered_catalog_data = filtered_catalog.data
            catalog_coord = SkyCoord(filtered_catalog_data['X_WORLD'], filtered_catalog_data['Y_WORLD'], unit='deg')
            reference_coord = SkyCoord(all_references['ra'], all_references['dec'], unit='deg')
            obj_indices, ref_indices, unmatched_obj_indices = self.helper.cross_match(catalog_coord, reference_coord, max_distance_second = max_distance_second)
            matched_obj = filtered_catalog_data[obj_indices]
            matched_ref = all_references[ref_indices]
            filtered_catalog.data = matched_obj
            if len(matched_obj) < min_number_of_sources:
                mag_max += 0.5
                classstar_lower = np.max([0.0, classstar_lower - 0.1])
                self.helper.print(f"[WARN] No enough sources found. Increasing mag_max to {mag_max}, classstar_lower to {classstar_lower}", verbose)
            else:
                break

        # Update the target image header
        update_kwargs['SEEING'] = (target_seeing, "Seeing FWHM in pixel")
        update_kwargs['PEEING'] = (target_seeing / np.mean(target_img.pixelscale), "Seeing FWHM in arcsec")

        if 'SKYVAL' in filtered_catalog_data.colnames:
            skyval = float(filtered_catalog_data['SKYVAL'][0])
        elif 'BACKGROUND' in filtered_catalog_data.colnames:
            skyval= float(np.mean(filtered_catalog_data['BACKGROUND']))
        else:
            skyval = target_img.info.SKYVAL
        update_kwargs['SKYVAL'] = (skyval, "Global Background level in ADU")
        
        skysig = None
        if 'SKYSIG' in filtered_catalog_data.colnames:
            skysig = float(filtered_catalog_data['SKYSIG'][0])
        else:
            skysig = target_img.info.SKYSIG
        update_kwargs['SKYSIG'] = (skysig, "Global background noise in ADU")
        
        ellip = None
        if 'ELLIPTICITY' in filtered_catalog_data.colnames:
            ellip = np.mean(filtered_catalog_data['ELLIPTICITY'])
        else:
            ellip = target_img.info.ELLIPTICITY
        update_kwargs['ELLIP'] = (ellip, "Mean ellipticity of the sources in the catalog")

        mag_key_ref = '%s_mag'%(target_img.filter)
        magerr_key_ref = 'e_%s_mag'%(target_img.filter)
        mag_key_all = [
            col for col in matched_obj.colnames
            if col.startswith('MAG_') and not np.all(matched_obj[col] == 0)
        ]
        
        def linear(x, a, b):
            return a * x + b

        zp_info = dict()
        color_term_info = dict()
        mag_term_info = dict()
        target_catalog_data = target_catalog.data
        # Update magnitude related keys
        for mag_key in mag_key_all:
            magerr_key = mag_key.replace('MAG_', 'MAGERR_')
            zp_key = mag_key.replace('MAG_', 'ZP_')
            zperr_key = magerr_key.replace('MAGERR_', 'ZPERR_')
            mag_key_sky = mag_key.replace('MAG_', 'MAGSKY_')
            npix_key = mag_key.replace('MAG_', 'NPIX_')
            ul3_key = mag_key.replace('MAG_', 'UL3SKY_')
            ul5_key = mag_key.replace('MAG_', 'UL5SKY_')
            
            # Calculate zero point
            zp = matched_ref[mag_key_ref] - matched_obj[mag_key]
            sc = SigmaClip(sigma=3.0, maxiters=5)
            masked = sc(zp)
            zp_cleaned_indices = np.where(~masked.mask)[0]
            masked_zp = zp[~masked.mask]
            zp_median = np.ma.median(masked_zp)
            zp_err = np.ma.std(masked_zp)     
            target_catalog_data[mag_key_sky] = target_catalog_data[mag_key] + zp_median
            target_catalog_data[zp_key] = zp_median
            target_catalog_data[zperr_key] = zp_err    
            matched_obj[mag_key_sky] = matched_obj[mag_key] + zp_median
            matched_obj[zp_key] = zp_median
            matched_obj[zperr_key] = zp_err  
            zp_info[mag_key] = dict(ZP_all = masked_zp, ZP_median = zp_median, ZP_err = zp_err, ZP_target = matched_obj[zp_cleaned_indices], ZP_reference = matched_ref[zp_cleaned_indices])
            update_kwargs[zp_key] = (zp_median, f"Zeropoint for {mag_key}")
            update_kwargs[zperr_key] = (zp_err, f"Zeropoint error for {mag_key}")
            
            # Calculate Depth
            if (npix_key in target_catalog_data.colnames):
                if skysig is not None:
                    npix_aperture = np.mean(target_catalog_data[npix_key])
                    bkg_noise = skysig * np.sqrt(npix_aperture)
                    ul3 = zp_median - 2.5 * np.log10(3 * bkg_noise)
                    ul5 = zp_median - 2.5 * np.log10(5 * bkg_noise)
                    target_catalog_data[ul3_key] = ul3
                    target_catalog_data[ul5_key] = ul5
                    matched_obj[ul3_key] = ul3
                    matched_obj[ul5_key] = ul5
                    update_kwargs[ul3_key] = (ul3, f"3-sigma depth for {mag_key}")
                    update_kwargs[ul5_key] = (ul5, f"5-sigma depth for {mag_key}")

            # When calculate_color_terms
            if calculate_color_terms:
                color_terms = [
                ('g', 'r'),
                ('g', 'i'),
                ('r', 'i'),
                ('B', 'V'),
                ('V', 'R'),
                ('R', 'I')
                ]
                for filt_ in ['g', 'r', 'i', 'B', 'V', 'R', 'I']:
                    color_terms.append((target_img.filter, filt_))
                
                if target_img.filter.startswith('m'):
                    target_filter_nm = float(re.findall(r'\d+', target_img.filter)[0])

                    all_filters_in_matched_ref = matched_ref.colnames
                    all_filters_nm_in_matched_ref = np.array([
                        float(re.findall(r'\d+', f)[0])
                        for f in all_filters_in_matched_ref
                        if f.startswith('m')
                    ])
                    
                    nearby_idx = np.where(
                        (np.abs(all_filters_nm_in_matched_ref - target_filter_nm) < 350) &
                        (np.abs(all_filters_nm_in_matched_ref - target_filter_nm) > 150)
                    )[0]

                    nearby_filters_nm = all_filters_nm_in_matched_ref[nearby_idx]

                    for filt_nm in nearby_filters_nm:
                        color_terms.append((target_img.filter, f'm{int(filt_nm)}'))
                                      
                reference_tbl = matched_ref[zp_cleaned_indices]
                for f1, f2 in color_terms:
                    key1 = f'{f1}_mag'
                    key2 = f'{f2}_mag'    
                    slope_key = mag_key.replace('MAG_', 'K_COLOR_') + f'_{f1}-{f2}'
                    intercept_key = mag_key.replace('MAG_', 'C_COLOR_') + f'_{f1}-{f2}'
                    if key1 in reference_tbl.colnames and key2 in reference_tbl.colnames:
                        color = reference_tbl[key1] - reference_tbl[key2]
                        try:
                            # Calculate residuals
                            zp_residual = masked_zp - zp_median
                            
                            # Fit (ZP - ZP_median) = a * color + b
                            popt, pcov = curve_fit(linear, color, zp_residual)
                            
                            color_term_info[slope_key] = {
                                'slope': popt[0],
                                'intercept': popt[1],
                                'filters': (f1, f2),
                            }

                            # Save slope and intercept with comments
                            update_kwargs[slope_key] = (round(popt[0],4), f"Slope a in color correction: mag offset = a*({f1}-{f2}) + b")
                            update_kwargs[intercept_key] = (round(popt[1],4), f"Intercept b in color correction: mag offset = a*({f1}-{f2}) + b")

                        except Exception as e:
                            self.helper.print(f"[WARN] [{mag_key}] Color term {f1}-{f2} fit failed: {e}", verbose)   
            
            if calculate_mag_terms:
                slope_key = mag_key.replace('MAG_', 'K_MAG_') 
                intercept_key = mag_key.replace('MAG_', 'C_MAG_') 
                zp_residual = masked_zp - zp_median
                mag = matched_obj[mag_key][zp_cleaned_indices] + zp_median
                magerr = matched_obj[magerr_key][zp_cleaned_indices]
                #mag = matched_ref[mag_key_ref][zp_cleaned_indices]
                #magerr = matched_ref[magerr_key_ref][zp_cleaned_indices]
                try:
                    # Fit (ZP - ZP_median) = a * mag + b
                    popt, pcov = curve_fit(linear, mag, zp_residual)
                    
                    mag_term_info[slope_key] = {
                        'slope': popt[0],
                        'intercept': popt[1]
                    }
                    
                    # Save slope and intercept with comments
                    update_kwargs[slope_key] = (round(popt[0],4), f"Slope a in magnitude correction: mag offset = a*m_sky + b")
                    update_kwargs[intercept_key] = (round(popt[1],4), f"Intercept b in magnitude correction: mag offset = a*m_sky + b")
                    
                except Exception as e:
                    self.helper.print(f"[WARN] [{mag_key}] Magnitude term fit failed: {e}", verbose)

            # Save spatial variation of the zeropoint (density map)
            if visualize or save_fig:
                if 'zp_maps' not in locals():
                    zp_maps = {}
                    zp_errs = {}
                    zp_extent = None

                zp_residual = zp - zp_median
                x = matched_obj['X_IMAGE']
                y = matched_obj['Y_IMAGE']
                zp_residual_cleaned = zp_residual[zp_cleaned_indices]
                x_cleaned = x[zp_cleaned_indices]
                y_cleaned = y[zp_cleaned_indices]

                bins = 100
                X, Y = np.meshgrid(
                    np.linspace(x_cleaned.min(), x_cleaned.max(), bins),
                    np.linspace(y_cleaned.min(), y_cleaned.max(), bins)
                )

                poly2d = models.Polynomial2D(degree=2)
                fitter = fitting.LinearLSQFitter()
                zp_model = fitter(poly2d, x_cleaned, y_cleaned, zp_residual_cleaned)
                Z = zp_model(X, Y)

                zp_maps[mag_key] = Z
                zp_errs[mag_key] = zp_err
                zp_extent = [x.min(), x.max(), y.min(), y.max()]

        # Visualization for 2D ZP map
        if (visualize or save_fig) and 'zp_maps' in locals():

            n = len(zp_maps)
            ncols = int(np.ceil(np.sqrt(n)))
            nrows = int(np.ceil(n / ncols))

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(4.5*ncols, 3.5*nrows),
                sharex=True,
                sharey=True,
                dpi=300
            )
            axes = np.atleast_2d(axes)

            vmin = min(np.nanmin(Z) for Z in zp_maps.values())
            vmax = max(np.nanmax(Z) for Z in zp_maps.values())
            
            # --- global colormap & norm ---
            cmap = plt.cm.viridis

            if vmin < 0 < vmax:
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            for ax, (mag_key, Z) in zip(axes.flat, zp_maps.items()):

                im = ax.imshow(
                    Z,
                    origin='lower',
                    extent=zp_extent,
                    cmap=cmap,
                    norm=norm
                )

                ax.set_title(mag_key, fontsize=14)
                ax.set_xlabel('X_IMAGE', fontsize=14)
                ax.set_ylabel('Y_IMAGE', fontsize=14)
                ax.tick_params(axis='both', labelsize=12)

                err = zp_errs[mag_key]
                if np.isfinite(err) and err > 0:
                    CS = ax.contour(
                        X, Y, Z,
                        levels=[-err, err],
                        colors='white',
                        linewidths=1
                    )
                    ax.clabel(
                        CS,
                        fmt=lambda v: f'{v:+.3f}',
                        fontsize=11,
                        inline=True
                    )

                # --- scatter: mag_key별 residual ---
                info = zp_info[mag_key]
                x = info['ZP_target']['X_IMAGE']
                y = info['ZP_target']['Y_IMAGE']
                zp_residual = info['ZP_all'] - info['ZP_median']

                ax.scatter(
                    x, y,
                    c=zp_residual,
                    cmap=cmap,
                    norm=norm,
                    s=15,
                    edgecolor='k',
                    linewidth=0.3,
                    alpha=0.7,
                    zorder=10
                )

            for ax in axes.flat[len(zp_maps):]:
                ax.axis('off')

            cbar = fig.colorbar(
                im,
                ax=axes.ravel().tolist(),
                orientation='horizontal',
                location='bottom',
                shrink=0.9,
                pad=0.08
            )

            cbar.set_label('ZP spatial residual (mag)', fontsize=18)
            cbar.ax.tick_params(labelsize=14)

            fig.suptitle('2D Zeropoint Spatial Variation (all apertures)', fontsize=20)
            fig.tight_layout(rect=[0, 0.3, 1, 0.985])

            if save_fig:
                fig.savefig(str(target_img.savepath.catalogpath) + '.zp_2d.png', dpi=300)

            if visualize:
                plt.show()
            plt.close(fig)
        
        # Visualization for ZP vs reference magnitude
        if visualize or save_fig:            

            n = len(zp_info)
            ncols = int(np.ceil(np.sqrt(n)))
            nrows = int(np.ceil(n / ncols))

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(4.5*ncols, 4*nrows),
                sharex=True,
                sharey=True,
                dpi=300
            )
            axes = np.atleast_2d(axes)

            for ax, (mag_key, info) in zip(axes.flat, zp_info.items()):
                ref_mag = info['ZP_reference'][mag_key_ref]
                zp = info['ZP_all']              # absolute ZP
                zp_med = info['ZP_median']
                zp_err = info['ZP_err']

                ax.scatter(ref_mag, zp, marker = 'o', facecolor = 'none', edgecolor = 'k', s=15, alpha=0.4)
                ax.axhline(zp_med, color='k', ls='--', lw=1)
                
                ax.fill_between(
                    [ref_mag.min(), ref_mag.max()],
                    zp_med - zp_err,
                    zp_med + zp_err,
                    color='k',
                    alpha=0.1
                )                
                
                ax.set_title(
                    f'{mag_key}\n'
                    f'ZP={zp_med:.3f} ± {zp_err:.3f}',
                    fontsize=14
                )
                ax.invert_xaxis()
                ax.tick_params(axis='both', labelsize=14)

                # optional magnitude-term fit
                k_key = mag_key.replace('MAG_', 'K_MAG_')
                if calculate_mag_terms and k_key in mag_term_info:
                    a = mag_term_info[k_key]['slope']
                    b = mag_term_info[k_key]['intercept']
                    xfit = np.linspace(ref_mag.min(), ref_mag.max(), 100)
                    ax.plot(xfit, a*xfit + b + zp_med, 'b--', lw=1)

            for ax in axes.flat[len(zp_info):]:
                ax.axis('off')

            fig.supxlabel('Reference magnitude', fontsize=20)
            fig.supylabel('Zeropoint', fontsize=20)
            fig.suptitle(f'ZP ({target_img.filter}) vs Reference Magnitude', fontsize=30)
            fig.tight_layout(rect=[0.03, 0, 0.97, 0.97])

            if save_fig:
                fig_path = str(target_img.savepath.catalogpath) + '.zp_mag.png'
                fig.savefig(fig_path, dpi=300)
                self.helper.print(f"[INFO] ZP vs mag plot saved to {fig_path}", verbose)

            if visualize:
                plt.show()
            plt.close(fig)
            
        # ============================================================
        # Visualization: ZP vs g-r color
        # ============================================================
        if visualize or save_fig:

            n = len(zp_info)
            ncols = int(np.ceil(np.sqrt(n)))
            nrows = int(np.ceil(n / ncols))

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(4.5*ncols, 4*nrows),
                sharex=True,
                sharey=True,
                dpi=300
            )
            axes = np.atleast_2d(axes)

            for ax, (mag_key, info) in zip(axes.flat, zp_info.items()):

                ref = info['ZP_reference']

                # --- g-r color from reference catalog ---
                if ('g_mag' not in ref.colnames) or ('r_mag' not in ref.colnames):
                    ax.text(0.5, 0.5, 'no g-r',
                            transform=ax.transAxes,
                            ha='center', va='center')
                    ax.axis('off')
                    continue

                color = ref['g_mag'] - ref['r_mag']

                zp = info['ZP_all']
                zp_med = info['ZP_median']
                zp_err = info['ZP_err']

                zp_res = zp - zp_med   # residual (recommended)

                # --- scatter ---
                ax.scatter(
                    color,
                    zp_res,
                    s=15,
                    facecolor='none',
                    edgecolor='k',
                    alpha=0.5
                )

                # --- zero line ---
                ax.axhline(0.0, color='k', ls='--', lw=1)

                # --- ±1σ band ---
                ax.fill_between(
                    [color.min(), color.max()],
                    -zp_err,
                    +zp_err,
                    color='k',
                    alpha=0.1
                )

                ax.set_title(
                    f'{mag_key}\n'
                    rf'$\sigma_{{ZP}}={zp_err:.3f}$',
                    fontsize=14
                )

                ax.tick_params(axis='both', labelsize=14)

                # --- optional color-term fit ---
                if calculate_color_terms:
                    # example: g-r
                    slope_key = mag_key.replace('MAG_', 'K_COLOR_') + '_g-r'
                    intercept_key = mag_key.replace('MAG_', 'C_COLOR_') + '_g-r'

                    if slope_key in color_term_info:
                        a = color_term_info[slope_key]['slope']
                        b = color_term_info[slope_key]['intercept']

                        xfit = np.linspace(color.min(), color.max(), 100)
                        yfit = a * xfit + b

                        ax.plot(xfit, yfit, 'r--', lw=1)

            for ax in axes.flat[len(zp_info):]:
                ax.axis('off')

            fig.supxlabel('Reference color (g − r)', fontsize=20)
            fig.supylabel('ZP residual (mag)', fontsize=20)
            fig.suptitle(
                f'ZP residual vs g−r color ({target_img.filter})',
                fontsize=30
            )
            fig.tight_layout(rect=[0.03, 0, 0.97, 0.97])

            if save_fig:
                fig_path = str(target_img.savepath.catalogpath) + '.zp_color.png'
                fig.savefig(fig_path, dpi=300)
                self.helper.print(f"[INFO] ZP vs color plot saved to {fig_path}", verbose)

            if visualize:
                plt.show()
            plt.close(fig)

        
        # update_kwargs = {
        #     'depth': target_img.depth,
        #     'seeing': target_img.seeing}
        
        # for key, value in update_kwargs.items():
        #     target_catalog.info.update(key, value)
        #     filtered_catalog.info.update(key, value)

        # Final: Update the header
        if update_header:
            for key, value in update_kwargs.items():
                if isinstance(value, tuple):
                    target_img.header[key] = value
                else:
                    target_img.header[key] = (value, "")
        
        # Update the target image
        target_img.update_status('ZPCALC')
        target_img.write(verbose = verbose) 

        if save:
            target_catalog.write(verbose = verbose)
            
        if save_refcat:
            filtered_catalog.write(verbose = verbose)
        return target_img, target_catalog, filtered_catalog, update_kwargs
    
    def apply_zp(self,
                target_img: Union[ScienceImage, ReferenceImage],
                target_catalog: Catalog,
                verbose: bool = True,
                save: bool = True) -> Table:
        """
        Apply photometric zeropoint corrections using values saved in the FITS header.
        Adds MAGSKY_*, ZP_*, ZPERR_*, UL3_*, UL5_* columns to target_catalog.
        
        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage
            The target image to apply photometric zeropoint corrections to.
        target_catalog : Catalog
            The catalog to apply photometric zeropoint corrections to.
        save : bool, optional
            Whether to save the catalog.
            
        Returns
        -------
        target_catalog: Catalog
            The updated catalog with photometric zeropoint corrections.
        """
        
        header = target_img.header
        target_catalog_data = target_catalog.data
        skysig = None
        if 'SKYSIG' in target_catalog_data.colnames:
            skysig = float(target_catalog_data['SKYSIG'][0])
        else:
            skysig = target_img.info.SKYSIG
        
        magsky_keys = [
            col for col in target_catalog_data.colnames
            if col.startswith('MAG_') and not np.all(target_catalog_data[col] == 0)
        ]

        for mag_key in magsky_keys:
            mag_key_sky = mag_key.replace('MAG_', 'MAGSKY_')
            zp_key = mag_key.replace('MAG_', 'ZP_')
            zperr_key = mag_key.replace('MAG_', 'ZPERR_')
            ul3_key = mag_key.replace('MAG_', 'UL3_')
            ul5_key = mag_key.replace('MAG_', 'UL5_')
            npix_key = mag_key.replace('MAG_', 'NPIX_')
            ul3_key_sky = ul3_key.replace('UL3_', 'UL3SKY_')
            ul5_key_sky = ul5_key.replace('UL5_', 'UL5SKY_')

            if zp_key not in header:
                self.helper.print(f"[WARNING] {zp_key} not in header. Skipping {mag_key}", True)
                continue

            zp = header[zp_key]
            target_catalog_data[mag_key_sky] = target_catalog_data[mag_key] + zp
            target_catalog_data[zp_key] = zp
            if ul3_key in target_catalog_data.colnames:
                target_catalog_data[ul3_key_sky] = target_catalog_data[ul3_key] + zp
            else:
                if skysig is not None:
                    npix_aperture = np.mean(target_catalog_data[npix_key])
                    bkg_noise = skysig * np.sqrt(npix_aperture)
                    ul3 = zp - 2.5 * np.log10(3 * bkg_noise)
                    target_catalog_data[ul3_key_sky] = ul3
                
            if ul5_key in target_catalog_data.colnames:
                target_catalog_data[ul5_key_sky] = target_catalog_data[ul5_key] + zp
            else:
                if skysig is not None:
                    npix_aperture = np.mean(target_catalog_data[npix_key])
                    bkg_noise = skysig * np.sqrt(npix_aperture)
                    ul5 = zp - 2.5 * np.log10(5 * bkg_noise)
                    target_catalog_data[ul5_key_sky] = ul5

            if zperr_key in header:
                target_catalog_data[zperr_key] = header[zperr_key]

        if save:
            target_catalog.write(verbose = verbose)
            
        return target_catalog
    
    def find_comparison_catalog(self,
                                target_img,
                                target_catalog,
                                max_day_offset=1,
                                verbose=False):
        """
        Automatically find a suitable comparison catalog for color-term correction.
        """
        
        def wavelength_score_gaussian(filt_nm, center=550.0, sigma=150.0):
            """
            filt_nm : float or array (nm)
            """
            return np.exp(-0.5 * ((filt_nm - center) / sigma)**2)

        from ezphot.utils import DataBrowser
        from astropy.time import Time

        dbrowser = DataBrowser('scidata')
        dbrowser.observatory = target_img.observatory
        dbrowser.objname = target_img.objname

        # --- find catalogs with same filename pattern ---
        catalog_filename = target_catalog.path.name
        prefix = catalog_filename.split('_')[0]
        suffix = catalog_filename.split('_')[-1]
        pattern = f'{prefix}*{suffix}'

        catalogset = dbrowser.search(pattern=pattern, return_type='catalog')

        # --- time filter ---
        t0 = Time(target_img.obsdate, format='isot')
        tmin = t0 - max_day_offset * u.day
        tmax = t0 + max_day_offset * u.day
        catalogset.select_catalogs(obs_start=tmin.isot, obs_end=tmax.isot)

        if len(catalogset.catalogs) == 0:
            raise RuntimeError("No comparison catalogs found within time window")

        # --- classify filters ---
        target_filter = target_img.filter
        target_nm = None
        if target_filter.startswith('m'):
            target_nm = float(re.findall(r'\d+', target_filter)[0])
        candidates = []

        for cat in catalogset.catalogs:
            filt = cat.info.filter

            # skip same filter
            if filt == target_filter:
                continue

            score = 0.0

            if target_filter.startswith('m'):
                if filt.startswith('m'):
                    nm = float(re.findall(r'\d+', filt)[0])
                    dnm = abs(nm - target_nm)

                    if not (150 < dnm < 350):
                        continue  # reject too close / too far

                    score += 3.0
                    score += wavelength_score_gaussian(nm)  # ⭐ 525nm preference

                else:
                    # broad band vs medium band
                    score += 2.0
                    score += 1.0  # neutral wavelength score

            else:
                # target is broad band
                if filt in ['g', 'r', 'i', 'B', 'V', 'R', 'I']:
                    score += 3.0
                else:
                    score += 1.0
                score += 1.0  # wavelength not meaningful

            # (optional) time preference could go here

            candidates.append((score, cat))

        if len(candidates) == 0:
            raise RuntimeError("No suitable comparison catalog found")

        # --- choose best ---
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_cat = candidates[0]

        if verbose:
            self.helper.print(f"[INFO] Comparison catalog selected: {best_cat.info.filter} ({best_cat.info.obsdate})",verbose)

        return best_cat
    
    
    def apply_color_terms(self, target_img: Union[ScienceImage, ReferenceImage], target_catalog: Catalog, 
                          comparison_catalog: Catalog = None,
                          magkey_suffix: Union[str, List[str]] = 'MAGSKY_',
                          max_distance_arcsec: float = 2.5, 
                          save: bool = True, 
                          verbose: bool = False):
        
        self._backup_magsky_once(target_catalog)

        def linear(x, a, b):
            return a * x + b
        
        def select_catalog_rows(merged_tbl, catalog_idx, data_keys):
            """
            Select rows where the given catalog has a detection.
            """
            idx_cols = [f"{k}_idx{catalog_idx}" for k in data_keys
                        if f"{k}_idx{catalog_idx}" in merged_tbl.colnames]

            if len(idx_cols) == 0:
                raise ValueError("No matching columns found for catalog_idx")

            mask = np.zeros(len(merged_tbl), dtype=bool)
            for col in idx_cols:
                mask |= np.isfinite(merged_tbl[col])

            return merged_tbl[mask]
        
        if comparison_catalog is None:
            comparison_catalog = self.find_comparison_catalog(
                target_img, target_catalog, max_day_offset=1, verbose=verbose
            )
            
        num_to_row = {
            int(n): i
            for i, n in enumerate(target_catalog.data['NUMBER'])
        }
        from ezphot.dataobjects import CatalogSet
        catalog_dataset = CatalogSet([target_catalog, comparison_catalog])
        if isinstance(magkey_suffix, str):
            magkey_suffix = [magkey_suffix]
        magsky_keys = []
        for magkey_suffix in magkey_suffix:
            magsky_keys.extend([c for c in target_catalog.data.colnames
                                if c.startswith(magkey_suffix)])

        merged_catalog, merged_metadata = catalog_dataset.merge_catalogs(
            max_distance_arcsec=max_distance_arcsec,
            join_type='outer',
            data_keys=magsky_keys + ['NUMBER']
        )

        target_catalog_data = select_catalog_rows(
            merged_catalog,
            catalog_idx=0,
            data_keys=magsky_keys
        )
        order = [
            num_to_row[int(n)]
            for n in target_catalog_data['NUMBER_idx0']
        ]

        order = np.array(order, dtype=int)        
        
        header = target_img.header
        tab = target_catalog.data[order]

        f1 = target_catalog.info.filter
        f2 = comparison_catalog.info.filter

        for k in magsky_keys:
            cterm_key = k.replace('MAGSKY_', 'CTERM_')
            if cterm_key in target_catalog.data.colnames:
                continue
            slope_key_12 = k.replace('MAGSKY_', 'K_COLOR_') + f'_{f1}-{f2}'
            intercept_key_12 = k.replace('MAGSKY_', 'C_COLOR_') + f'_{f1}-{f2}'
            slope_key_21 = k.replace('MAGSKY_', 'K_COLOR_') + f'_{f2}-{f1}'
            intercept_key_21 = k.replace('MAGSKY_', 'C_COLOR_') + f'_{f2}-{f1}'

            m1 = k + '_idx0'
            m2 = k + '_idx1'

            if slope_key_12 in header and intercept_key_12 in header:
                a = float(header[slope_key_12])
                b = float(header[intercept_key_12])
                color = np.array(target_catalog_data[m1] - target_catalog_data[m2], dtype=float)
            elif slope_key_21 in header and intercept_key_21 in header:
                a = float(header[slope_key_21])
                b = float(header[intercept_key_21])
                color = np.array(target_catalog_data[m2] - target_catalog_data[m1], dtype=float)
            else:
                self.helper.print(f"[WARN] No color-term coeffs for {k}", verbose)
                target_catalog.data[cterm_key] = np.zeros(len(target_catalog.data), dtype=float)
                continue

            cterm = linear(color, a, b)

            # NaN-safe
            cterm[~np.isfinite(cterm)] = 0.0
            tab[k] = target_catalog_data[m1] + cterm
            tab[cterm_key] = cterm
        
        target_catalog.data = tab
        if save:
            target_catalog.write(verbose=verbose)
        return target_catalog

    def apply_mag_terms(self, target_img: Union[ScienceImage, ReferenceImage], target_catalog: Catalog, 
                        magkey_suffix: Union[str, List[str]] = 'MAGSKY_', 
                        verbose: bool = False,
                        save: bool = True):

        self._backup_magsky_once(target_catalog)
        
        def linear(x, a, b):
            return a * x + b
        
        header = target_img.header
        tab = target_catalog.data.copy()
        
        if isinstance(magkey_suffix, str):
            magkey_suffix = [magkey_suffix]
        magsky_keys = []
        for magkey_suffix in magkey_suffix:
            magsky_keys.extend([c for c in tab.colnames
                                if c.startswith(magkey_suffix)])
            
        for k in magsky_keys:
            mterm_key = k.replace('MAGSKY_', 'MTERM_')
            if mterm_key in tab.colnames:
                continue
            slope_key = k.replace('MAGSKY_', 'K_MAG_')
            intercept_key = k.replace('MAGSKY_', 'C_MAG_')

            if slope_key not in header or intercept_key not in header:
                self.helper.print(f"[WARN] No mag-term coeffs for {k}", verbose)
                tab[mterm_key] = np.zeros(len(tab), dtype=float)
                continue

            a = float(header[slope_key])
            b = float(header[intercept_key])

            mag0 = np.array(tab[k], dtype=float)
            mterm = linear(mag0, a, b)
            
            mterm[~np.isfinite(mterm)] = 0.0
            tab[k] = mag0 + mterm
            tab[mterm_key] = mterm

        # do NOT overwrite MAGSKY here; let apply_all_terms handle it
        target_catalog.data = tab
        if save:
            target_catalog.write(verbose=verbose)
        return target_catalog

    def _backup_magsky_once(self, catalog):
        """
        Backup MAGSKY_* → ORIGIN_MAGSKY_* (only once)
        """
        for col in catalog.data.colnames:
            if col.startswith('MAGSKY_') and not col.startswith('ORIGIN_MAGSKY_'):
                orig_col = col.replace('MAGSKY_', 'ORIGIN_MAGSKY_')
                if orig_col not in catalog.data.colnames:
                    catalog.data[orig_col] = catalog.data[col].copy()




    # def apply_color_terms(self,
    #                       target_img: Union[ScienceImage, ReferenceImage],
    #                       target_catalog: Catalog,
    #                       comparison_catalog: Catalog = None,                          
    #                       max_distance_arcsec: float = 2.5,
    #                       save: bool = True,
    #                       verbose: bool = False
    #                       ):
    #     """
    #     Apply color term correction to target_catalog using compare_catalog_for_color.
    #     Color term equation: MAG_corrected = MAG + a*(color) + b
    #     where color = compare_catalog[filter_1] - compare_catalog[filter_2]
        
    #     Parameters
    #     ----------
    #     target_img : ScienceImage or ReferenceImage
    #         The target image to apply color term corrections to.
    #     target_catalog : Catalog
    #         The catalog to apply color term corrections to.
    #     comparison_catalog : Catalog
    #         The catalog to use for color term corrections.
    #     max_distance_second : float, optional
    #         The maximum distance in arcseconds to consider for color term corrections.
    #     save : bool, optional
    #         Whether to save the catalog.
    #     verbose : bool, optional
    #         Whether to print verbose output.
            
    #     Returns
    #     -------
    #     target_catalog: Catalog
    #         The updated catalog with color term corrections.
    #     """
    #     def linear(x, a, b):
    #         return a * x + b

    #     if comparison_catalog is None:
    #         comparison_catalog = self.find_comparison_catalog(target_img, target_catalog,
    #                                                           max_day_offset = 1,
    #                                                           verbose = verbose)
    #     # 0. Cross-match catalogs
    #     from ezphot.dataobjects import CatalogSet
    #     catalog_dataset = CatalogSet([target_catalog, comparison_catalog])
    #     magsky_key_all = [col for col in target_catalog.data.colnames if col.startswith('MAGSKY_')]
    #     mag_key_all = [col.replace('MAGSKY_','MAG_') for col in magsky_key_all]
    #     merged_catalog, merged_metadata = catalog_dataset.merge_catalogs(max_distance_arcsec = max_distance_arcsec, join_type = 'outer', data_keys = magsky_key_all)
    #     target_catalog_data = merged_catalog[:len(target_catalog.data)]
    #     filter_1_key = target_catalog.info.filter
    #     filter_2_key = comparison_catalog.info.filter
        
    #     header = target_img.header
        
    #     for magsky_key in magsky_key_all:
    #         slope_key_try1 = magsky_key.replace('MAGSKY_', 'K_COLOR_') + f'_{filter_1_key}-{filter_2_key}'
    #         intercept_key_try1 = magsky_key.replace('MAGSKY_', 'C_COLOR_') + f'_{filter_1_key}-{filter_2_key}'
    #         slope_key_try2 = magsky_key.replace('MAGSKY_', 'K_COLOR_') + f'_{filter_2_key}-{filter_1_key}'
    #         intercept_key_try2 = magsky_key.replace('MAGSKY_', 'C_COLOR_') + f'_{filter_2_key}-{filter_1_key}'
    #         magsky_filter_1_key = magsky_key + '_idx0'
    #         magsky_filter_2_key = magsky_key + '_idx1'
    #         # Calculate color term
    #         if slope_key_try1 in header and intercept_key_try1 in header:
    #             slope = header[slope_key_try1]
    #             intercept = header[intercept_key_try1]
    #             color = target_catalog_data[magsky_filter_1_key] - target_catalog_data[magsky_filter_2_key]
    #             color_key = f'{filter_1_key}-{filter_2_key}'
    #         elif slope_key_try2 in header and intercept_key_try2 in header:
    #             slope = header[slope_key_try2]
    #             intercept = header[intercept_key_try2]
    #             color = target_catalog_data[magsky_filter_2_key] - target_catalog_data[magsky_filter_1_key]
    #             color_key = f'{filter_2_key}-{filter_1_key}'
    #         else:
    #             self.helper.print(f"[WARNING] Color term keys '{slope_key_try1}' or '{intercept_key_try1}' not found in FITS header.", verbose)
    #             continue
    #         color_term = linear(color, slope, intercept)
    #         # Update target_catalog with color term
    #         corrected_magsky_key = magsky_key.replace('MAGSKY_', 'MAGSKY_CORR_')
    #         colorterm_key = magsky_key.replace('MAGSKY_', 'CTERM_')
    #         corrmag_key = magsky_key.replace('MAGSKY_', 'C_CORR_MAGSKY_')
    #         target_catalog.data[corrmag_key] = target_catalog.data[magsky_key] + color_term
    #         target_catalog.data[colorterm_key] = color_term
    #         target_catalog.data[color_key] = color
        
    #     if save:
    #         target_catalog.write(verbose = verbose)    

    #     return target_catalog
            
    # def apply_mag_terms(self,
    #                     target_img: Union[ScienceImage, ReferenceImage],
    #                     target_catalog: Catalog,
    #                     save: bool = True,
    #                     verbose: bool = False):
    #     """
    #     Apply magnitude term correction to target_catalog using values saved in the FITS header.
    #     Magnitude term equation: MAG_corrected = MAG + a*(MAG) + b
        
    #     Parameters
    #     ----------
    #     target_img : ScienceImage or ReferenceImage
    #         The target image to apply magnitude term corrections to.
    #     target_catalog : Catalog
    #         The catalog to apply magnitude term corrections to.
    #     save : bool, optional
    #         Whether to save the catalog.
    #     verbose : bool, optional
    #         Whether to print verbose output.
            
    #     Returns
    #     -------
    #     target_catalog: Catalog
    #         The updated catalog with magnitude term corrections.
    #     """
    #     def linear(x, a, b):
    #         return a * x + b
        
    #     target_catalog_data = target_catalog.data
    #     magsky_key_all = [
    #         col for col in target_catalog_data.colnames
    #         if col.startswith('MAGSKY_') and not np.all(target_catalog_data[col] == 0)
    #     ]
    #     header = target_img.header

    #     for magsky_key in magsky_key_all:
    #         slope_key = magsky_key.replace('MAGSKY_', 'K_MAG_')
    #         intercept_key = magsky_key.replace('MAGSKY_', 'C_MAG_')
    #         if slope_key not in header or intercept_key not in header:
    #             self.helper.print (f"[WARNING] Color term keys '{slope_key}' or '{intercept_key}' not found in FITS header.", verbose)
    #             continue
    #         # Calculate mag term
    #         slope = header[slope_key]
    #         intercept = header[intercept_key]
    #         mag = target_catalog_data[magsky_key]
    #         mag_term = linear(mag, slope, intercept)
    #         # Update target_catalog with mag term
    #         magterm_key = magsky_key.replace('MAGSKY_', 'MTERM_')
    #         corrmag_key = magsky_key.replace('MAGSKY_', 'M_CORR_MAGSKY_')
    #         if magsky_key not in target_catalog_data.colnames:
    #             self.helper.print (f"[WARNING] '{magsky_key}' not found in target catalog.", verbose)
    #             continue
    #         target_catalog_data[corrmag_key] = target_catalog_data[magsky_key] + mag_term
    #         target_catalog_data[magterm_key] = mag_term
            
    #     if save:
    #         target_catalog.write(verbose = verbose)
            
    #     return target_catalog

    def select_stars(self,
                     target_catalog: Catalog,
                     mag_lower: float = 12,
                     mag_upper: float = 16,
                     snr_lower: float = 10,
                     snr_upper: float = 300,
                     classstar_lower: float = 0.8,
                     elongation_upper: float = 1.5,
                     elongation_sigma: float = 5,
                     fwhm_lower_arcsec: float = 1,
                     fwhm_upper_arcsec: float = 15,
                     fwhm_sigma: float = 5,
                     flag_upper: int = 1,
                     maskflag_upper: int = 1,
                     ra_deg: float = None,
                     dec_deg: float = None,
                     radius_arcsec: float = None,
                     inner_fraction: float = 0.7, # Fraction of the images
                     isolation_radius_arcsec: float = 10.0,
                     
                     save: bool = False,
                     verbose: bool = True,
                     visualize: bool = True,
                     save_fig: bool = False,
                     
                     magnitude_key: str = 'MAG_AUTO',
                     magnitudeerr_key: str = 'MAGERR_AUTO',
                     fwhm_key: str = 'FWHM_WORLD',
                     ra_key: str = 'X_WORLD',
                     dec_key: str = 'Y_WORLD',
                     classstar_key: str = 'CLASS_STAR',
                     elongation_key: str = 'ELONGATION',
                     flag_key: str = 'FLAGS',
                     maskflag_key: str = 'IMAFLAGS_ISO',
                     ) -> Table:
        """
        Filter stars by selecting the top N non-saturated, isolated, round, appropriately bright
        sources from each image grid cell (or globally if num_grids is None or 0).

        Parameters
        ----------
        target_catalog : Catalog
            The catalog to filter stars from.
        mag_lower : float, optional
            Minimum magnitude to select stars from.
        mag_upper : float, optional
            Maximum magnitude to select stars from.
        classstar_lower : float, optional
            Minimum CLASS_STAR to select stars from.
        elongation_upper : float, optional
            Maximum elongation to select stars from.
        elongation_sigma : float, optional
            Sigma of the elongation to select stars from.
        fwhm_lower : float, optional
            Minimum FWHM to select stars from.
        fwhm_upper : float, optional
            Maximum FWHM to select stars from.
        fwhm_sigma : float, optional
            Sigma of the FWHM to select stars from.
        flag_upper : int, optional
            Maximum flag to select stars from.
        maskflag_upper : int, optional
            Maximum mask flag to select stars from.
        inner_fraction : float, optional
            Fraction of the image to select stars from.
        isolation_radius : float, optional
            Isolation radius to select stars from.
        save : bool, optional
            Whether to save the catalog.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the catalog.
        save_fig : bool, optional
            Whether to save the figure.
            
        Returns
        -------
        filtered_catalog: Catalog
            The filtered catalog with stars selected.
        """
        target_catalog_data = target_catalog.data
        if target_catalog.data is None:
            raise ValueError("target_catalog.data is None. Please provide a valid Catalog object with data.")
        
        if fwhm_key not in target_catalog_data.keys():
            visualize = False
            self.helper.print(f"Warning: '{fwhm_key}' not found in target_catalog. Visualization disabled.", verbose)
        if visualize or save_fig:
            plt.figure(dpi=300)
            plt.xlabel(magnitude_key)
            plt.ylabel(fwhm_key)
            plt.title("Star selection filtering")
            
        def _plot_if_visualize(x, y, color, label, alpha=0.4):
            if visualize or save_fig:  # or pass `visualize` as a parameter
                plt.scatter(x, y, c=color, alpha=alpha, label=label)
        _plot_if_visualize(target_catalog_data[magnitude_key], target_catalog_data[fwhm_key]*3600, 'k', label = 'All sources', alpha = 0.3)#, c = sources[x_key])
        filtered_catalog_data = target_catalog_data.copy()
        self.helper.print(f'Initial sources: {len(filtered_catalog_data)}', verbose)
        filter_info = {'initial': len(filtered_catalog_data)}

        # Step 0: FWHM cut: remove too small sources
        if fwhm_key not in filtered_catalog_data.keys():
            self.helper.print(f"Warning: '{fwhm_key}' not found in target_catalog.", verbose)
        else:
            abs_fwhm_mask = (filtered_catalog_data[fwhm_key] > fwhm_lower_arcsec/3600) & (filtered_catalog_data[fwhm_key] < fwhm_upper_arcsec/3600)
            filtered_catalog_data = filtered_catalog_data[abs_fwhm_mask]
            
            filter_info['after_fwhm_abs'] = len(filtered_catalog_data)
            self.helper.print(f"[FWHM ABS CUT]: {len(filtered_catalog_data)} sources passed {fwhm_lower_arcsec} < FWHM < {fwhm_upper_arcsec} ", verbose)
        filter_info['after_fwhm_abs'] = len(filtered_catalog_data)
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'm', label = 'FWHM(Asvolute) cut', alpha = 0.3)

        # Step 1: Inner region cut        
        if ra_key not in filtered_catalog_data.keys() or dec_key not in filtered_catalog_data.keys():
            self.helper.print(f"Warning: '{ra_key}' or '{dec_key}' not found in sources.", verbose)
        elif ra_deg is not None and dec_deg is not None and radius_arcsec is not None:
            # If ra_deg, dec_deg, radius_arcsec is given, use them to select the nearby sources
            ra_vals = filtered_catalog_data[ra_key]
            dec_vals = filtered_catalog_data[dec_key]
            catalog_coords = SkyCoord(ra = ra_vals, dec = dec_vals, unit = (u.deg, u.deg))
            input_coords = SkyCoord(ra = ra_deg, dec = dec_deg, unit = (u.deg, u.deg))
            distances = catalog_coords.separation(input_coords)
            nearby_mask = distances < radius_arcsec * u.arcsec
            filtered_catalog_data = filtered_catalog_data[nearby_mask]
            self.helper.print(f'[NEARBYREGION CUT] {len(filtered_catalog_data)} sources passed within {radius_arcsec} arcsec', verbose)
        else:        
            # If ra_deg, dec_deg, radius_arcsec is not given, use inner_fraction to select the inner region sources
            ra_vals = filtered_catalog_data[ra_key]
            dec_vals = filtered_catalog_data[dec_key]
            catalog_coords = SkyCoord(ra = ra_vals, dec = dec_vals, unit = (u.deg, u.deg))
            center = SkyCoord(np.median(catalog_coords.ra.deg)*u.deg, np.median(catalog_coords.dec.deg)*u.deg)
            
            # radius that corresponds to inner_fraction of the catalog footprint (simple heuristic)
            sep = catalog_coords.separation(center)
            rmax = np.nanpercentile(sep.arcsec, inner_fraction * 100)
            inner_mask = sep.arcsec <= rmax
            filtered_catalog_data = filtered_catalog_data[inner_mask]
            self.helper.print(f'[INNERREGION CUT] {len(filtered_catalog_data)} sources passed within {rmax} arcsec', verbose)
        
        filter_info['after_region'] = len(filtered_catalog_data)        
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'r', label = 'Region cut', alpha = 0.3)

        # Step 2: Isolation
        if ra_key not in filtered_catalog_data.keys() or dec_key not in filtered_catalog_data.keys():
            self.helper.print(f"Warning: '{ra_key}' or '{dec_key}' not found in sources.", verbose)
        else:
            coords = SkyCoord(ra = filtered_catalog_data[ra_key], dec = filtered_catalog_data[dec_key], unit = (u.deg, u.deg))
            idx1, idx2, sep2d, _ = coords.search_around_sky(coords, isolation_radius_arcsec * u.arcsec)
            # count neighbors for each object (exclude itself)
            neighbor_count = np.bincount(idx1, minlength=len(coords)) - 1

            isolated_mask = neighbor_count == 0
            filtered_catalog_data = filtered_catalog_data[isolated_mask]
            self.helper.print(f'[ISOLATION CUT] {len(filtered_catalog_data)} sources passed with isolation radius {isolation_radius_arcsec} arcsec', verbose)
        filter_info['after_isolation'] = len(filtered_catalog_data)

        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'g', label = 'Isolation cut', alpha = 0.3)

        # Step 3: MAG cut
        if mag_lower is not None and mag_upper is not None:
            if mag_lower is not None:
                filtered_catalog_data = filtered_catalog_data[(filtered_catalog_data[magnitude_key] > mag_lower)]
            if mag_upper is not None:
                filtered_catalog_data = filtered_catalog_data[(filtered_catalog_data[magnitude_key] < mag_upper)]                
            self.helper.print(f"[MAG CUT]: {len(filtered_catalog_data)} sources passed {mag_lower} < {magnitude_key} < {mag_upper}", verbose)
        else:
            snr = 1.085736/filtered_catalog_data[magnitudeerr_key]
            snr_mask = (snr > snr_lower) & (snr < snr_upper)
            filtered_catalog_data = filtered_catalog_data[snr_mask]
            self.helper.print(f"[SNR CUT]: {len(filtered_catalog_data)} sources passed {snr_lower} < SNR < {snr_upper}", verbose)
        
        filter_info['after_magcut'] = len(filtered_catalog_data)
            
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'b', label = 'MAG cut', alpha = 0.3)

        # Step 4: CLASS_STAR cut
        if classstar_key not in filtered_catalog_data.keys():
            self.helper.print(f"Warning: '{classstar_key}' not found in sources.", verbose)
        else:
            class_star_mask = filtered_catalog_data[classstar_key] > classstar_lower
            filtered_catalog_data = filtered_catalog_data[class_star_mask]
            self.helper.print(f"[CLASSSTAR CUT]: {len(filtered_catalog_data)} sources passed CLASS_STAR > {classstar_lower}", verbose)
        filter_info['after_classstar'] = len(filtered_catalog_data)
 
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'cyan', label = 'ClassStar cut', alpha = 0.3)

        # Step 5: FWHM absolute and relative cut
        if fwhm_key not in filtered_catalog_data.keys():
            self.helper.print(f"Warning: '{fwhm_key}' not found in sources.", verbose)
        else:
            # Step 5.1: Absolute cut: remove too small sources
            abs_fwhm_mask = (filtered_catalog_data[fwhm_key] > fwhm_lower_arcsec/3600) & (filtered_catalog_data[fwhm_key] < fwhm_upper_arcsec/3600)
            filtered_catalog_data = filtered_catalog_data[abs_fwhm_mask]
            
            # Step 5.2: Relative cut: sigma-clipped sources
            fwhm_values = filtered_catalog_data[fwhm_key]
            fwhm_mean, fwhm_median, fwhm_std = sigma_clipped_stats(fwhm_values, sigma=5.0, maxiters=10)
            clip_mask = np.abs(fwhm_values - fwhm_median) <= fwhm_sigma * fwhm_std
            filtered_catalog_data = filtered_catalog_data[clip_mask]
            filter_info['after_fwhm'] = len(filtered_catalog_data)
            self.helper.print(
                f"[FWHM CUT]: {len(filtered_catalog_data)} sources passed {fwhm_lower_arcsec} < FWHM < {fwhm_upper_arcsec} and within ±{fwhm_sigma} sigma"
                f"around median ({fwhm_median*3600:.2f} ± {fwhm_sigma* fwhm_std*3600:.2f}) arcsec",
                verbose
            ) 
            
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'orange', label = 'FWHM(Relative) cut', alpha = 0.3)

        # Step 6: Elongation cut
        if elongation_key not in filtered_catalog_data.keys():
            self.helper.print(f"Warning: '{elongation_key}' not found in sources.", verbose)
        else:
            # Step 6.1: Absolute limit
            elong_vals = filtered_catalog_data[elongation_key]
            abs_elong_mask = elong_vals < elongation_upper
            filtered_catalog_data = filtered_catalog_data[abs_elong_mask]
            filter_info['after_elong_abs'] = len(filtered_catalog_data)

            # Step 6.2: Sigma-clipping
            elong_vals = filtered_catalog_data[elongation_key]
            elong_mean, elong_median, elong_std = sigma_clipped_stats(elong_vals, sigma=5.0, maxiters=5)
            sigclip_mask = np.abs(elong_vals - elong_median) < elongation_sigma * elong_std
            filtered_catalog_data = filtered_catalog_data[sigclip_mask]
            filter_info['after_elong_sigclip'] = len(filtered_catalog_data)

            self.helper.print(f"[ELONGATION CUT]: {len(filtered_catalog_data)} passed elongation < {elongation_upper} and within ±{elongation_sigma} sigma of median ({elong_median:.2f} ± {elong_std:.2f})", verbose)

        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'purple', label = 'Elongation cut', alpha = 0.3)
        
        # Step 7: Flag cut
        if flag_key not in filtered_catalog_data.keys():
            self.helper.print(f"Warning: '{flag_key}' not found in sources.", verbose)
        else:
            flag_mask = filtered_catalog_data[flag_key] <= flag_upper
            filtered_catalog_data = filtered_catalog_data[flag_mask]
            self.helper.print(f"[FLAG CUT]: {len(filtered_catalog_data)} sources passed FLAGS <= {flag_upper}", verbose)
        filter_info['after_flag'] = len(filtered_catalog_data)
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'magenta', label = 'Flag cut', alpha = 0.3)
        
        # Step 8: Mask flag cut
        if maskflag_key not in filtered_catalog_data.keys():
            self.helper.print(f"Warning: '{maskflag_key}' not found in sources.", verbose)
        else:
            maskflag_mask = filtered_catalog_data[maskflag_key] <= maskflag_upper
            filtered_catalog_data = filtered_catalog_data[maskflag_mask]
            self.helper.print(f"[MASKFLAG CUT]: {len(filtered_catalog_data)} sources passed IMAFLAGS_ISO <= {maskflag_upper}", verbose)
        filter_info['after_maskflag'] = len(filtered_catalog_data)
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'brown', label = 'MaskFlag cut', alpha = 0.3)

        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'red', label = 'Final selected', alpha = 0.3)

        seeing = np.median(filtered_catalog_data[fwhm_key]) * 3600
        filtered_catalog = Catalog(target_catalog.savepath.refcatalogpath, catalog_type = 'reference', info = target_catalog.info.copy(), load = False)
        filtered_catalog.data = filtered_catalog_data
        filtered_catalog.info.path = str(target_catalog.savepath.refcatalogpath)
        
        if visualize or save_fig:
            plt.legend()
            plt.ylim(seeing - 2, seeing + 10)
            valid_mag = target_catalog_data[magnitude_key][~np.isnan(target_catalog_data[magnitude_key])]
            median_mag = np.median(valid_mag) if len(valid_mag) > 0 else 0
            
            if len(valid_mag) > 0:
                mag_min = max(median_mag - 9, np.min(valid_mag)-0.5)
                mag_max = min(median_mag + 3, np.max(valid_mag)+0.5)
                plt.xlim(mag_min, mag_max)
            else:
                # No valid data to set xlim
                self.helper.print("Warning: No valid magnitudes for setting xlim.", verbose)
                
            if save_fig:
                plt.savefig(str(filtered_catalog.savepath.savepath) + '.png', dpi=300)
                self.helper.print(f"[INFO] Star selection plot saved to {str(filtered_catalog.savepath.savepath) }", verbose)
            if visualize:
                plt.show()
            plt.close()
            
        if save:
            filtered_catalog.write(verbose = verbose)
            self.helper.print(f"[INFO] Filtered catalog saved to {filtered_catalog.savepath.savepath}", verbose)
    
        return filtered_catalog, filter_info, seeing

    def determine_reference_mag_range(self,
                                      target_catalog: Catalog,
                                      bin_width=0.5,
                                      sigma_clip=5,
                                      magnitude_key='MAG_AUTO',
                                      zp_key='ZP_AUTO',
                                      fwhm_key='FWHM_IMAGE',
                                      verbose=True,
                                      visualize=False,
                                      save_fig=False):
        """
        Determine the reference magnitude range for the target catalog.
        
        Parameters
        ----------
        target_catalog : Catalog
            The target catalog to determine the reference magnitude range for.
        bin_width : float, optional
            The width of the magnitude bins.
        sigma_clip : float, optional
            The sigma to use for the sigma-clipping.
        magnitude_key : str, optional
            The key to use for the magnitude.
        zp_key : str, optional
            The key to use for the zeropoint.
        fwhm_key : str, optional
            The key to use for the FWHM.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the reference magnitude range.
        save_fig : bool, optional
            Whether to save the figure.
            
        Returns
        -------
        mag_min : float
            The minimum magnitude of the reference magnitude range.
        mag_max : float
            The maximum magnitude of the reference magnitude range.
        zp : float
            The zeropoint of the reference magnitude range.
        zp_err : float
            The error of the zeropoint of the reference magnitude range.
        saturation_level : float
            The saturation level of the reference magnitude range.
        """
        
        # === ??? ?? ===
        catalog_data = target_catalog.data
        mags = np.array(catalog_data[magnitude_key])
        zps = np.array(catalog_data[zp_key])
        fwhms = np.array(catalog_data[fwhm_key])

        # === Bin data ===
        mag_bins = np.arange(np.floor(mags.min()), np.ceil(mags.max()) + bin_width, bin_width)
        bin_centers, med_zp, std_zp = [], [], []

        for i in range(len(mag_bins)-1):
            mask = (mags >= mag_bins[i]) & (mags < mag_bins[i+1])
            if np.sum(mask) > 2:
                clip_mean, clip_median, clip_std = sigma_clipped_stats(zps[mask], sigma=sigma_clip, maxiters=5)
                bin_centers.append((mag_bins[i] + mag_bins[i+1]) / 2)
                med_zp.append(clip_median)
                std_zp.append(clip_std)

        bin_centers = np.array(bin_centers)
        med_zp = np.array(med_zp)
        std_zp = np.array(std_zp)

        if len(bin_centers) < 5:
            raise ValueError("Not enough binned data to determine reference magnitude range.")

        # === LOWESS smoothing ===
        smooth_zp = med_zp#lowess(med_zp, bin_centers, frac=0.1, return_sorted=False)
        smooth_zp_err = std_zp#lowess(std_zp, bin_centers, frac=0.3, return_sorted=False)
        zp_slope = np.gradient(smooth_zp, bin_centers)

        # ???? ?? ?? ?? ?? (??? ??? ??)
        mid_mask = (bin_centers > np.percentile(bin_centers, 30)) & (bin_centers < np.percentile(bin_centers, 70))
        stable_slopes = zp_slope[mid_mask]

        # sigma-clipped stats? ??? ?? ??
        _, _, slope_std = sigma_clipped_stats(stable_slopes, sigma=2, maxiters=5)

        # slope_threshold ?? ?? (?: 3? ??)
        slope_threshold = 3 * slope_std
        
        _, zp_value_first, _ = sigma_clipped_stats(med_zp, sigma=1, maxiters=5)
        start_idx = np.argmin(np.abs(smooth_zp - zp_value_first))
        
        # === mag_min ??: ZP slope ===
        mag_min = bin_centers[0]
        consecutive = 0
        first_idx = None

        # faint ? bright ???? ??
        saturation_level = 60000
        for i in range(start_idx, -1, -1):  # start_idx?? ?? ??? ??
            if abs(zp_slope[i]) > slope_threshold:
                consecutive += 1
                if first_idx is None:  
                    first_idx = i  # ? ??? ??? ?? ?? ??
                if consecutive >= 3:
                    mag_min = bin_centers[first_idx]  # ?? ??? ? ?? idx? mag_min ??
                    saturated_sources = catalog_data[catalog_data[magnitude_key] <= mag_min]
                    saturated_sources_count = len(saturated_sources)
                    if saturated_sources_count > 0:
                        saturation_level = np.median(saturated_sources['FLUX_MAX'])
                        self.helper.print(f"Saturation level: {saturation_level:.2f}", verbose)
                    self.helper.print(f"Saturation detected (stable): mag_min = {mag_min:.2f}", verbose)
                    break
            else:
                consecutive = 0
                first_idx = None  # ?? ?? ???

        # === mag_max ??: ZP_err ? ?? ===
        # ?? ???? baseline? ???? ??
        mid_mask = (bin_centers < np.percentile(bin_centers, 50)) & (bin_centers > mag_min)

        # Sigma clipping to reject outliers
        sigmaclip = SigmaClip(sigma=2, maxiters=5)
        clipped_zp_err = sigmaclip(smooth_zp_err[mid_mask])  # Masked array after sigma clipping

        # Remove masked (outlier) values
        valid_zp_err = clipped_zp_err[~clipped_zp_err.mask]
        zp_err = np.percentile(valid_zp_err, 10)
        err_threshold = zp_err + 1 * np.std(valid_zp_err)

        # ?? ????? faint ???? ??
        start_idx = np.argmax(mid_mask)  # mid_mask?? ? True ??? (?? ???)
        mag_max = bin_centers[-1]  # ???: ?? faint
        consecutive = 0
        first_idx = None

        for i in range(start_idx, len(bin_centers)):  # mid ? faint
            if smooth_zp_err[i] > err_threshold:
                consecutive += 1
                if first_idx is None:
                    first_idx = i
                if consecutive >= 2:  # ?? 2? ?? ? mag_max ??
                    mag_max = bin_centers[first_idx]
                    break
            else:
                consecutive = 0
                first_idx = None


        # === ZP, ZP_err ?? ===
        valid_mask = (bin_centers >= mag_min) & (bin_centers <= mag_max)
        zp = np.median(med_zp[valid_mask])
        zp_err = np.median(std_zp[valid_mask])

        if verbose:
            self.helper.print(f"[INFO] mag_min={mag_min:.2f}, mag_max={mag_max:.2f}, ZP={zp:.3f} ± {zp_err:.3f}", verbose)

        # === Visualization ===
        if visualize or save_fig:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

            # ZP plot
            ax1.scatter(mags, zps, c=fwhms, cmap='plasma', alpha=0.5, s=8)
            ax1.plot(bin_centers, smooth_zp, 'o-', color='orange', label="Smoothed ZP")
            ax1.axvline(mag_min, color='red', ls='--', label=f"mag_min={mag_min:.2f}")
            ax1.axvline(mag_max, color='blue', ls='--', label=f"mag_max={mag_max:.2f}")
            ax1.set_ylabel("Zeropoint")
            ax1.set_ylim(zp - 2, zp + 1)
            ax1.legend()
            ax1.invert_xaxis()

            # ZP_err plot
            ax2.plot(bin_centers, smooth_zp_err, 'o-', color='purple', label="ZP Std (Err)")
            ax2.axhline(err_threshold, color='gray', ls='--', label="ZP_err Threshold")
            ax2.axvline(mag_min, color='red', ls='--')
            ax2.axvline(mag_max, color='blue', ls='--')
            ax2.set_xlabel("Magnitude")
            ax2.set_ylabel("ZP Error")
            ax2.set_ylim(zp_err-0.1, zp_err+ 0.2)
            ax2.legend()

            if save_fig:
                plt.savefig(str(target_catalog.savepath.savepath) + '.reference_mag.png', dpi=300)
            if visualize:
                plt.show()
            plt.close()

        return mag_min, mag_max, zp, zp_err, saturation_level
    
    def stable_zp_indices(self, zp_list, zp_err_list, mode='self'):
        """
        Find indices where adjacent ZP differences are smaller than ZP error.

        Parameters
        ----------
        zp : array-like
            ZP values (length N)
        zp_err : array-like
            ZP errors (length N)
        mode : {'self', 'next', 'combined'}
            self     : |ZP[i+1] - ZP[i]| < err[i]
            next     : |ZP[i+1] - ZP[i]| < err[i+1]
            combined : |ZP[i+1] - ZP[i]| < sqrt(err[i]^2 + err[i+1]^2)

        Returns
        -------
        idx : ndarray
            Indices i where condition holds (refers to ZP[i])
        """
        zp = np.asarray(zp_list)
        zp_err = np.asarray(zp_err_list)

        dzp = np.abs(zp[1:] - zp[:-1])

        if mode == 'self':
            sigma = zp_err[:-1]
        elif mode == 'next':
            sigma = zp_err[1:]
        elif mode == 'combined':
            sigma = np.sqrt(zp_err[:-1]**2 + zp_err[1:]**2)
        else:
            raise ValueError("mode must be 'self', 'next', or 'combined'")

        idx = np.where(dzp < sigma)[0]
        return idx, dzp, sigma
# %%
