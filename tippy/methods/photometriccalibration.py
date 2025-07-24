#%%
from typing import Union, Optional
from astropy.table import Table, vstack
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import SigmaClip
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit

from tippy.configuration import TIPConfig
from tippy.helper import Helper
from tippy.imageojbects import ScienceImage, ReferenceImage, CalibrationImage
from tippy.catalog import TIPCatalog
from tippy.catalog.utils import *
#%%

class TIPPhotometricCalibration(Helper):
    
    def __init__(self):
        super().__init__()

    def photometric_calibration(self,
                                target_img: Union[ScienceImage, ReferenceImage, CalibrationImage],
                                target_catalog: TIPCatalog,
                                catalog_type: str = 'GAIAXP',
                                max_distance_second: float = 1.0,
                                calculate_color_terms: bool = True,
                                calculate_mag_terms: bool = True,
                                
                                # Selection parameters
                                mag_lower: float = None,
                                mag_upper: float = None,
                                snr_lower: float = 20,
                                snr_upper: float = 300,
                                classstar_lower: float = 0.8,
                                elongation_upper: float = 1.7,
                                elongation_sigma: float = 5,
                                fwhm_lower: float = 2,
                                fwhm_upper: float = 15,
                                fwhm_sigma: float = 5,
                                flag_upper: int = 1,
                                maskflag_upper: int = 1,
                                inner_fraction: float = 0.7, # Fraction of the images
                                isolation_radius: float = 10.0,
                                magnitude_key: str = 'MAG_AUTO',
                                flux_key: str = 'FLUX_AUTO',
                                fluxerr_key: str = 'FLUXERR_AUTO',
                                fwhm_key: str = 'FWHM_IMAGE',
                                x_key: str = 'X_IMAGE',
                                y_key: str = 'Y_IMAGE',
                                classstar_key: str = 'CLASS_STAR',
                                elongation_key: str = 'ELONGATION',
                                flag_key: str = 'FLAGS',
                                maskflag_key: str = 'IMAFLAGS_ISO',

                                # Other parameters
                                save: bool = True,
                                verbose: bool = True,
                                visualize: bool = True,
                                save_fig: bool = False,
                                save_refcat: bool = True,
                                **kwargs):

        
        """_summary_

        target_catalog: Table = objects_sex
        catalog_type: str = 'GAIAXP'
        max_distance_second: float = 1.0
        
        visualize: bool = True
        save: bool = True
        """
        catalogs = get_catalogs_coord(
            ra = target_img.ra,
            dec = target_img.dec,
            fov_ra = target_img.fovx,
            fov_dec = target_img.fovy,
            catalog_type = catalog_type
        )
        if len(catalogs) == 0:
            raise ValueError("No catalogs found in the given coordinates.")
        else:
            all_references = Table()
            for catalog in catalogs:
                data = select_reference_sources(catalog = catalog)[0]
                #data = catalog.data
                all_references = vstack([all_references, data])
        
        # Filter out reference sources
        filtered_catalog, _, target_seeing = self.select_stars(
            target_catalog = target_catalog,
            verbose = verbose,
            visualize = visualize,
            save = False,
            save_fig = save_fig, 
            mag_lower = mag_lower,
            mag_upper = mag_upper,
            snr_lower = snr_lower,
            snr_upper = snr_upper,
            classstar_lower = classstar_lower,
            elongation_upper = elongation_upper,
            elongation_sigma = elongation_sigma,
            fwhm_lower = fwhm_lower,
            fwhm_sigma = fwhm_sigma,
            flag_upper = flag_upper,
            maskflag_upper = maskflag_upper,
            magnitude_key = magnitude_key,
            flux_key = flux_key,
            fluxerr_key = fluxerr_key,
            fwhm_key = fwhm_key,
            x_key = x_key,
            y_key = y_key,
            classstar_key = classstar_key,
            elongation_key = elongation_key,
            flag_key = flag_key,
            maskflag_key = maskflag_key,
            inner_fraction = inner_fraction,
            isolation_radius = isolation_radius
            )
        
        filtered_catalog_data = filtered_catalog.data
        catalog_coord = SkyCoord(filtered_catalog_data['X_WORLD'], filtered_catalog_data['Y_WORLD'], unit='deg')
        reference_coord = SkyCoord(all_references['ra'], all_references['dec'], unit='deg')
        obj_indices, ref_indices, unmatched_obj_indices = self.cross_match(catalog_coord, reference_coord, max_distance_second = max_distance_second)
        matched_obj = filtered_catalog_data[obj_indices]
        matched_ref = all_references[ref_indices]
        filtered_catalog.data = matched_obj
        if save_refcat:
            filtered_catalog.write()

        # Update the target image header
        update_kwargs = dict()
        update_kwargs['PEEING'] = (target_seeing, "Seeing FWHM in pixel")
        update_kwargs['SEEING'] = (target_seeing * np.mean(target_img.pixelscale), "Seeing FWHM in arcsec")

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

        magerr_key_all = [
            col for col in matched_obj.colnames
            if col.startswith('MAGERR_') and not np.all(matched_obj[col] == 0)
        ]

        def linear(x, a, b):
            return a * x + b

        zp_info = dict()
        color_term_info = dict()
        mag_term_info = dict()
        target_catalog_data = target_catalog.data
        # Update magnitude related keys
        for mag_key, magerr_key in zip(mag_key_all, magerr_key_all):
            zp_key = mag_key.replace('MAG_', 'ZP_')
            zperr_key = magerr_key.replace('MAGERR_', 'ZPERR_')
            mag_key_sky = mag_key.replace('MAG_', 'MAGSKY_')
            npix_key = mag_key.replace('MAG_', 'NPIX_')
            ul3_key = mag_key.replace('MAG_', 'UL3_')
            ul5_key = mag_key.replace('MAG_', 'UL5_')
            
            # Calculate zero point
            zp = matched_ref[mag_key_ref] - matched_obj[mag_key]
            #if magerr_key_ref in matched_ref.colnames:
            #    zperr = np.sqrt(matched_ref[magerr_key_ref]**2 + matched_objf[magerr_key]**2)
            sc = SigmaClip(sigma=5.0, maxiters=3)
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
                ('R', 'I'),
                ('m475', 'm625'), # g-r
                ('m625', 'm750'), # r-i
                ('m450', 'm550'), # B-V
                ('m550', 'm650'), # V-R
                ('m650', 'm800') # R-I
                ]
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
                            self.print(f"[WARN] [{mag_key}] Color term {f1}-{f2} fit failed: {e}", verbose)   
            
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
                    self.print(f"[WARN] [{mag_key}] Magnitude term fit failed: {e}", verbose)
        
        
        # Final: Update the header
        for key, value in update_kwargs.items():
            if isinstance(value, tuple):
                target_img.header[key] = value
            else:
                target_img.header[key] = (value, "")
        
        # Update the target image status
        target_img.update_status('ZPCALC')       
        
        # Write the target image 
        target_img.write() 
        
        if visualize or save_fig:
            catalog_coord_all = SkyCoord(target_catalog_data['X_WORLD'], target_catalog_data['Y_WORLD'], unit='deg')
            reference_coord_all = SkyCoord(all_references['ra'], all_references['dec'], unit='deg')
            obj_indices , ref_indices, unmatched_obj_indices = self.cross_match(catalog_coord_all, reference_coord_all, max_distance_second = max_distance_second)
            matched_obj_all = target_catalog_data[obj_indices]
            matched_ref_all = all_references[ref_indices]
            
            magerr_key = magnitude_key.replace('MAG_','MAGERR_')#'MAGERR_AUTO'
            zp_key = magnitude_key.replace('MAG_', 'ZP_')
            zp_all = matched_ref_all[mag_key_ref] - matched_obj_all[magnitude_key]
            zp_median = zp_info[magnitude_key]['ZP_median']
            
            plt.figure(dpi = 300)
            plt.title(f'{zp_key} calculation for {target_img.filter} band')
            plt.xlabel(f'Photometric reference ({target_img.filter})')
            plt.ylabel(f'{zp_key}')
            
            plt.scatter(matched_ref_all[mag_key_ref], zp_all, c = 'k', alpha = 0.15, label = 'All targets')
            plt.scatter(zp_info[magnitude_key]['ZP_reference'][mag_key_ref], zp_info[magnitude_key]['ZP_all'], c = 'r', alpha = 0.5, label = 'Selected targets')
            plt.errorbar(zp_info[magnitude_key]['ZP_reference'][mag_key_ref], zp_info[magnitude_key]['ZP_all'], yerr = np.sqrt(zp_info[magnitude_key]['ZP_target'][magerr_key]**2 + zp_info[magnitude_key]['ZP_err']**2), fmt='None', c = 'r', alpha=0.5)
            plt.axhline(zp_median, color='k', linestyle='--', label = 'ZP = %.3f +/- %.3f'%(zp_median, zp_info[mag_key]['ZP_err']))
            
            xmin = max(np.min(matched_ref_all[mag_key_ref]) -1, 9)
            xmax = min(np.max(matched_ref_all[mag_key_ref]) + 1, 20)
            plt.xlim(xmin, xmax)
            plt.ylim(zp_median - 0.5, zp_median + 1)
            plt.legend()
            
            if calculate_mag_terms:
                popt = list(mag_term_info[f'K_{magnitude_key}'].values())
                x_fit = np.linspace(xmin, xmax, 100)
                fit_result = linear(x_fit, *popt) + zp_median
                plt.plot(x_fit, fit_result, color='b', linestyle='--', label=f'Fit: {popt[0]:.3f}x+{popt[1]:.3f}, [{np.min(fit_result):.3f}~{np.max(fit_result):.3f}]')
            
            if save_fig:
                fig_path = str(target_img.savepath.catalogpath) + '.zp.png'
                plt.savefig(fig_path, dpi=300)
                self.print(f"[INFO] ZP calibration plot saved to {fig_path}", verbose)
            
            if visualize:
                plt.show()
            plt.close()

            if calculate_color_terms:

                # Automatically get all keys from zp_info
                photometry_keys = list(zp_info.keys())

                # Set up figure
                plt.figure(figsize=(8,6))

                # Optional: assign different colors and markers
                colors = ['k', 'r', 'g', 'b', 'm', 'c', 'y']
                markers = ['o', 's', '^', 'd', 'P', '*', 'v']

                # Setup cycling through colors and markers if many keys
                from itertools import cycle
                color_cycle = cycle(colors)
                marker_cycle = cycle(markers)

                for mag_key in photometry_keys:
                    ref = zp_info[mag_key]['ZP_reference']
                    target = zp_info[mag_key]['ZP_target']
                    zp_all = zp_info[mag_key]['ZP_all']
                    zp_median = zp_info[mag_key]['ZP_median']
                    zp_err = zp_info[mag_key]['ZP_err']

                    # g - r color
                    color = ref['g_mag'] - ref['r_mag']

                    # Fit linear model
                    try:
                        popt, pcov = curve_fit(linear, color, zp_all)
                    except Exception as e:
                        self.print(f"[WARN] Fitting failed for {mag_key}: {e}", verbose)
                        continue

                    # Plot scatter
                    color_ = next(color_cycle)
                    marker_ = next(marker_cycle)
                    plt.scatter(color, zp_all, color=color_, alpha=0.5, marker=marker_, label=f'{mag_key} ({zp_median:.3f} +/- {zp_err:.3f})')

                    # Plot fit line
                    x_fit = np.linspace(np.min(color), np.max(color), 100)
                    fit_result = linear(x_fit, *popt)
                    
                    plt.plot(x_fit, fit_result, color=color_, linestyle='--', label=f'Fit {mag_key}: {popt[0]:.3f}x+{popt[1]:.3f}, [{np.min(fit_result):.3f}~{np.max(fit_result):.3f}]')

                # --- Final plot settings ---
                plt.axhline(0, color='gray', linestyle=':')

                plt.xlabel('g - r color from reference catalog (mag)')
                plt.ylabel('Zero point residual (mag)')
                plt.title('ZP Residual vs Color')

                # Correct ylim setting: based on min and max of all zp_median
                if photometry_keys:
                    all_zp_medians = [zp_info[key]['ZP_median'] for key in photometry_keys]
                    zp_median_min = np.min(all_zp_medians)
                    zp_median_max = np.max(all_zp_medians)
                    plt.ylim(zp_median_min - 1, zp_median_max + 1.5)

                # Make legend smaller
                plt.legend(fontsize=8, loc='best', frameon=True, ncols=2)
                plt.grid(True)
                plt.tight_layout()
                if save_fig:
                    fig_path = str(target_img.savepath.catalogpath) + '.zp_color.png'
                    plt.savefig(fig_path, dpi=300)
                    self.print(f"[INFO] ZP calibration plot saved to {fig_path}", verbose)
                
                if visualize:
                    plt.show()
                
                plt.close()
        
        if save:
            target_catalog.write()
        return target_img, target_catalog, filtered_catalog
    
    def apply_zp(self,
                target_img: Union[ScienceImage, ReferenceImage],
                target_catalog: TIPCatalog,
                save: bool = True) -> Table:
        """
        Apply photometric zeropoint corrections using values saved in the FITS header.
        Adds MAGSKY_*, ZP_*, ZPERR_*, UL3_*, UL5_* columns to target_catalog.
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

            if zp_key not in header:
                print(f"[WARNING] {zp_key} not in header. Skipping {mag_key}")
                continue

            zp = header[zp_key]
            target_catalog_data[mag_key_sky] = target_catalog_data[mag_key] + zp
            target_catalog_data[zp_key] = zp
            if zperr_key in header:
                target_catalog_data[zperr_key] = header[zperr_key]

            if skysig is not None:
                npix_aperture = np.mean(target_catalog_data[npix_key])
                bkg_noise = skysig * np.sqrt(npix_aperture)
                ul3 = zp - 2.5 * np.log10(3 * bkg_noise)
                ul5 = zp - 2.5 * np.log10(5 * bkg_noise)
                target_catalog_data[ul3_key] = ul3
                target_catalog_data[ul5_key] = ul5

        if save:
            target_catalog.write()
            
        return target_catalog
    
    def apply_color_terms(self,
                          target_img: Union[ScienceImage, ReferenceImage],
                          target_catalog: TIPCatalog,
                          comparison_catalog: TIPCatalog,                          
                          max_distance_second: float = 1.0,
                          save: bool = True,
                          verbose: bool = False
                          ):
        """
        Apply color term correction to target_catalog using compare_catalog_for_color.
        Color term equation: MAG_corrected = MAG + a*(color) + b
        where color = compare_catalog[filter_1] - compare_catalog[filter_2]
        """
        def linear(x, a, b):
            return a * x + b
        
        # 0. Cross-match catalogs
        from tippy.catalog import TIPCatalogDataset
        catalog_dataset = TIPCatalogDataset([target_catalog, comparison_catalog])
        magsky_key_all = [col for col in target_catalog.data.colnames if col.startswith('MAGSKY_')]
        mag_key_all = [col.replace('MAGSKY_','MAG_') for col in magsky_key_all]
        merged_catalog, merged_metadata = catalog_dataset.merge_catalogs(max_distance_second = max_distance_second, join_type = 'outer', data_keys = magsky_key_all)
        target_catalog_data = merged_catalog[:len(target_catalog.data)]
        filter_1_key = target_catalog.info.filter
        filter_2_key = comparison_catalog.info.filter
        
        header = target_img.header
        
        for magsky_key in magsky_key_all:
            slope_key_try1 = magsky_key.replace('MAGSKY_', 'K_COLOR_') + f'_{filter_1_key}-{filter_2_key}'
            intercept_key_try1 = magsky_key.replace('MAGSKY_', 'C_COLOR_') + f'_{filter_1_key}-{filter_2_key}'
            slope_key_try2 = magsky_key.replace('MAGSKY_', 'K_COLOR_') + f'_{filter_2_key}-{filter_1_key}'
            intercept_key_try2 = magsky_key.replace('MAGSKY_', 'C_COLOR_') + f'_{filter_2_key}-{filter_1_key}'
            magsky_filter_1_key = magsky_key + '_idx0'
            magsky_filter_2_key = magsky_key + '_idx1'
            # Calculate color term
            if slope_key_try1 in header and intercept_key_try1 in header:
                slope = header[slope_key_try1]
                intercept = header[intercept_key_try]
                color = target_catalog_data[magsky_filter_1_key] - target_catalog_data[magsky_filter_2_key]
                color_key = f'{filter_1_key}-{filter_2_key}'
            elif slope_key_try2 in header and intercept_key_try2 in header:
                slope = header[slope_key_try2]
                intercept = header[intercept_key_try2]
                color = target_catalog_data[magsky_filter_2_key] - target_catalog_data[magsky_filter_1_key]
                color_key = f'{filter_2_key}-{filter_1_key}'
            else:
                self.print(f"[WARNING] Color term keys '{slope_key_try1}' or '{intercept_key_try1}' not found in FITS header.", verbose)
                continue
            color_term = linear(color, slope, intercept)
            # Update target_catalog with color term
            colorterm_key = magsky_key.replace('MAGSKY_', 'CTERM_')
            corrmag_key = magsky_key.replace('MAGSKY_', 'C_CORR_MAGSKY_')
            target_catalog.data[corrmag_key] = target_catalog.data[magsky_key] + color_term
            target_catalog.data[colorterm_key] = color_term
            target_catalog.data[color_key] = color
        
        if save:
            target_catalog.write()    

        return target_catalog
            
    def apply_mag_terms(self,
                        target_img: Union[ScienceImage, ReferenceImage],
                        target_catalog: TIPCatalog,
                        save: bool = True,
                        verbose: bool = False):
        def linear(x, a, b):
            return a * x + b
        
        target_catalog_data = target_catalog.data
        magsky_key_all = [
            col for col in target_catalog_data.colnames
            if col.startswith('MAGSKY_') and not np.all(target_catalog_data[col] == 0)
        ]
        header = target_img.header

        for magsky_key in magsky_key_all:
            slope_key = magsky_key.replace('MAGSKY_', 'K_MAG_')
            intercept_key = magsky_key.replace('MAGSKY_', 'C_MAG_')
            if slope_key not in header or intercept_key not in header:
                self.print (f"[WARNING] Color term keys '{slope_key}' or '{intercept_key}' not found in FITS header.", verbose)
                continue
            # Calculate mag term
            slope = header[slope_key]
            intercept = header[intercept_key]
            mag = target_catalog_data[magsky_key]
            mag_term = linear(mag, slope, intercept)
            # Update target_catalog with mag term
            magterm_key = magsky_key.replace('MAGSKY_', 'MTERM_')
            corrmag_key = magsky_key.replace('MAGSKY_', 'M_CORR_MAGSKY_')
            if magsky_key not in target_catalog_data.colnames:
                self.print (f"[WARNING] '{magsky_key}' not found in target catalog.", verbose)
                continue
            target_catalog_data[corrmag_key] = target_catalog_data[magsky_key] + mag_term
            target_catalog_data[magterm_key] = mag_term
            
        if save:
            target_catalog.write()
            
        return target_catalog

    def select_stars(self,
                     target_catalog: TIPCatalog,
                     mag_lower: float = None,
                     mag_upper: float = None,
                     snr_lower: float = 10,
                     snr_upper: float = 300,
                     classstar_lower: float = 0.8,
                     elongation_upper: float = 1.5,
                     elongation_sigma: float = 5,
                     fwhm_lower: float = 2,
                     fwhm_upper: float = 15,
                     fwhm_sigma: float = 5,
                     flag_upper: int = 1,
                     maskflag_upper: int = 1,
                     inner_fraction: float = 0.7, # Fraction of the images
                     isolation_radius: float = 5.0,
                     
                     save: bool = False,
                     verbose: bool = True,
                     visualize: bool = True,
                     save_fig: bool = False,
                     
                     magnitude_key: str = 'MAG_AUTO',
                     flux_key: str = 'FLUX_AUTO',
                     fluxerr_key: str = 'FLUXERR_AUTO',
                     fwhm_key: str = 'FWHM_IMAGE',
                     x_key: str = 'X_IMAGE',
                     y_key: str = 'Y_IMAGE',
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
        num_grids : int or None
            If None or 0, apply selection globally instead of grid-by-grid.
        eccentricity_percentile : float
            Keep sources below this percentile in eccentricity (rounder stars).
        min_brightness : float
            Minimum acceptable brightness (in segment_flux or sort_key units).
        max_brightness : float
            Maximum acceptable brightness (used to exclude cosmic rays, very bright saturated stars).
        
        """
        target_catalog_data = target_catalog.data
        if target_catalog.data is None:
            raise ValueError("target_catalog.data is None. Please provide a valid TIPCatalog object with data.")
        
        if fwhm_key not in target_catalog_data.keys():
            visualize = False
            self.print(f"Warning: '{fwhm_key}' not found in target_catalog. Visualization disabled.", verbose)
        if visualize or save_fig:
            plt.figure(dpi=300)
            plt.xlabel(magnitude_key)
            plt.ylabel(fwhm_key)
            plt.title("Star selection filtering")
            
        def _plot_if_visualize(x, y, color, label, alpha=0.4):
            if visualize or save_fig:  # or pass `visualize` as a parameter
                plt.scatter(x, y, c=color, alpha=alpha, label=label)
        _plot_if_visualize(target_catalog_data[magnitude_key], target_catalog_data[fwhm_key], 'k', label = 'All sources', alpha = 0.3)#, c = sources[x_key])
        filtered_catalog_data = target_catalog_data.copy()
        self.print(f'Initial sources: {len(filtered_catalog_data)}')
        filter_info = {'initial': len(filtered_catalog_data)}

        # Step 0: FWHM cut: remove too small sources
        if fwhm_key not in filtered_catalog_data.keys():
            self.print(f"Warning: '{fwhm_key}' not found in target_catalog.", verbose)
        else:
            abs_fwhm_mask = (filtered_catalog_data[fwhm_key] > fwhm_lower) & (filtered_catalog_data[fwhm_key] < fwhm_upper)
            filtered_catalog_data = filtered_catalog_data[abs_fwhm_mask]
            
            filter_info['after_fwhm_abs'] = len(filtered_catalog_data)
            self.print(f"[FWHM ABS CUT]: {len(filtered_catalog_data)} sources passed {fwhm_lower} < FWHM < {fwhm_upper} ", verbose)
        filter_info['after_fwhm_abs'] = len(filtered_catalog_data)
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key], 'm', label = 'FWHM(Asvolute) cut', alpha = 0.3)

        # Step 1: Inner region cut
        if x_key not in filtered_catalog_data.keys() or y_key not in filtered_catalog_data.keys():
            self.print(f"Warning: '{x_key}' or '{y_key}' not found in target_catalog.", verbose)
        else:
            x_vals = filtered_catalog_data[x_key]
            y_vals = filtered_catalog_data[y_key]

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
            filtered_catalog_data = filtered_catalog_data[inner_mask]
            self.print(f'[INNERREGION CUT] {len(filtered_catalog_data)} sources passed within X = [{x_inner_min},{x_inner_max}], Y = [{y_inner_min},{y_inner_max}]', verbose)
        filter_info['after_innerregion'] = len(filtered_catalog_data)
        
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key], 'r', label = 'InnerRegion cut', alpha = 0.3)

        # Step 2: Isolation
        if x_key not in filtered_catalog_data.keys() or y_key not in filtered_catalog_data.keys():
            self.print(f"Warning: '{x_key}' or '{y_key}' not found in sources.", verbose)
        else:
            # Step 1.1: Build KD-tree
            positions = np.vstack([filtered_catalog_data[x_key].value, filtered_catalog_data[y_key].value]).T
            tree = cKDTree(positions)
            neighbors = tree.query_ball_tree(tree, r=isolation_radius)

            # Step 1.2: Keep only isolated sources
            isolated_mask = np.array([len(nbrs) == 1 for nbrs in neighbors])
            filtered_catalog_data = filtered_catalog_data[isolated_mask]
            self.print(f'[ISOLATION CUT] {len(filtered_catalog_data)} sources passed with isolation radius {isolation_radius} pixels', verbose)
        filter_info['after_isolation'] = len(filtered_catalog_data)

        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key], 'g', label = 'Isolation cut', alpha = 0.3)

        # Step 3: SNR cut
        if flux_key not in filtered_catalog_data.keys():
            self.print(f"Warning: '{flux_key}' not found in sources.", verbose)
        else:
            snr_key = flux_key.replace('FLUX_', 'SNR_')
            filtered_catalog_data[snr_key] = filtered_catalog_data[flux_key] / filtered_catalog_data[fluxerr_key]
            if snr_lower is not None:
                filtered_catalog_data = filtered_catalog_data[(filtered_catalog_data[snr_key] > snr_lower)]
            if snr_upper is not None:
                filtered_catalog_data = filtered_catalog_data[(filtered_catalog_data[snr_key] < snr_upper)]
            if snr_lower is not None and snr_upper is not None:
                self.print(f"[SNR CUT]: {len(filtered_catalog_data)} sources passed {snr_lower} < SNR < {snr_upper}", verbose)
        filter_info['after_snrcut'] = len(filtered_catalog_data)
            
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key], 'b', label = 'SNR cut', alpha = 0.3)

        # Step 3: MAG cut
        if magnitude_key not in filtered_catalog_data.keys():
            self.print(f"Warning: '{magnitude_key}' not found in sources.", verbose)
        else:
            if mag_lower is not None:
                filtered_catalog_data = filtered_catalog_data[(filtered_catalog_data[magnitude_key] > mag_lower)]
            if mag_upper is not None:
                filtered_catalog_data = filtered_catalog_data[(filtered_catalog_data[magnitude_key] < mag_upper)]
            if mag_lower is not None and mag_upper is not None:
                self.print(f"[MAG CUT]: {len(filtered_catalog_data)} sources passed {mag_lower} < {magnitude_key} < {mag_upper}", verbose)
        filter_info['after_magcut'] = len(filtered_catalog_data)
            
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key], 'b', label = 'MAG cut', alpha = 0.3)

        # Step 4: CLASS_STAR cut
        if classstar_key not in filtered_catalog_data.keys():
            self.print(f"Warning: '{classstar_key}' not found in sources.", verbose)
        else:
            class_star_mask = filtered_catalog_data[classstar_key] > classstar_lower
            filtered_catalog_data = filtered_catalog_data[class_star_mask]
            self.print(f"[CLASSSTAR CUT]: {len(filtered_catalog_data)} sources passed CLASS_STAR > {classstar_lower}", verbose)
        filter_info['after_classstar'] = len(filtered_catalog_data)
 
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key], 'cyan', label = 'ClassStar cut', alpha = 0.3)

        # Step 5: FWHM absolute and relative cut
        if fwhm_key not in filtered_catalog_data.keys():
            self.print(f"Warning: '{fwhm_key}' not found in sources.", verbose)
        else:
            # Stel 5.2: Relative cut: sigma-clipped sources
            fwhm_values = filtered_catalog_data[fwhm_key]
            fwhm_mean, fwhm_median, fwhm_std = sigma_clipped_stats(fwhm_values, sigma=5.0, maxiters=10)
            clip_mask = np.abs(fwhm_values - fwhm_median) <= fwhm_sigma * fwhm_std
            filtered_catalog_data = filtered_catalog_data[clip_mask]
            filter_info['after_fwhm_percentile'] = len(filtered_catalog_data)
            self.print(
                f"[FWHM CUT]: {len(filtered_catalog_data)} sources passed within ±{fwhm_sigma} sigma"
                f"around median ({fwhm_median:.2f} ± {fwhm_std:.2f})",
                verbose
            ) 
            
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key], 'orange', label = 'FWHM(Relative) cut', alpha = 0.3)

        # Step 6: Elongation cut
        if elongation_key not in filtered_catalog_data.keys():
            self.print(f"Warning: '{elongation_key}' not found in sources.", verbose)
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

            self.print(f"[ELONGATION CUT]: {len(filtered_catalog_data)} passed elongation < {elongation_upper} and within ±{elongation_sigma} sigma of median ({elong_median:.2f} ± {elong_std:.2f})", verbose)

        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key], 'purple', label = 'Elongation cut', alpha = 0.3)
        
        # Step 7: Flag cut
        if flag_key not in filtered_catalog_data.keys():
            self.print(f"Warning: '{flag_key}' not found in sources.", verbose)
        else:
            flag_mask = filtered_catalog_data[flag_key] <= flag_upper
            filtered_catalog_data = filtered_catalog_data[flag_mask]
            self.print(f"[FLAG CUT]: {len(filtered_catalog_data)} sources passed FLAGS <= {flag_upper}", verbose)
        filter_info['after_flag'] = len(filtered_catalog_data)
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key], 'magenta', label = 'Flag cut', alpha = 0.3)
        
        # Step 8: Mask flag cut
        if maskflag_key not in filtered_catalog_data.keys():
            self.print(f"Warning: '{maskflag_key}' not found in sources.", verbose)
        else:
            maskflag_mask = filtered_catalog_data[maskflag_key] <= maskflag_upper
            filtered_catalog_data = filtered_catalog_data[maskflag_mask]
            self.print(f"[MASKFLAG CUT]: {len(filtered_catalog_data)} sources passed IMAFLAGS_ISO <= {maskflag_upper}", verbose)
        filter_info['after_maskflag'] = len(filtered_catalog_data)
        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key], 'brown', label = 'MaskFlag cut', alpha = 0.3)

        _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key], 'red', label = 'Final selected', alpha = 0.3)

        seeing = np.median(filtered_catalog_data[fwhm_key])
        filtered_catalog = TIPCatalog(target_catalog.savepath.refcatalogpath, catalog_type = 'reference', info = target_catalog.info, load = False)
        filtered_catalog.data = filtered_catalog_data
        filtered_catalog.load_target_img(target_img = target_catalog.target_img)
        
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
                self.print("Warning: No valid magnitudes for setting xlim.", verbose)
                
            if save_fig:
                plt.savefig(str(filtered_catalog.savepath.savepath) + '.png', dpi=300)
                self.print(f"[INFO] Star selection plot saved to {str(filtered_catalog.savepath.savepath) }", verbose)

            if visualize:
                plt.show()
            plt.close()
            
        if save:
            filtered_catalog.write()
            self.print(f"[INFO] Filtered catalog saved to {filtered_catalog.savepath.savepath}", verbose)
    
        return filtered_catalog, filter_info, seeing
        
# %%
if __name__ == "__main__":
    from tippy.configuration import TIPConfig
    from tippy.imageojbects import ScienceImage, Mask, Background, Errormap
    target_path = '/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T01462/7DT02/g/calib_7DT02_T01462_20250211_041624_g_100.fits'
    self = TIPPhotometricCalibration()
    target_img  = ScienceImage(target_path, telinfo = self.get_telinfo('7DT', 'C361K', 'HIGH', 1), load = True)
    target_path = '/data/data1/factory_hhchoi/data/scidata/7DT/7DT_C361K_HIGH_1x1/T01462/7DT15/i/calib_7DT15_T01462_20250212_051656_i_100.com.fits'
    target_img  = ScienceImage(target_path, telinfo = self.get_telinfo('7DT', 'C361K', 'HIGH', 1), load = True)
    target_catalog = TIPCatalog(target_img.savepath.catalogpath, catalog_type = 'all', load = True)
    catalog_type: str = 'GAIAXP'
    max_distance_second: float = 1.0
    calculate_color_terms: bool = True
    calculate_mag_terms: bool = True
    target_catalog = TIPCatalog(target_img.savepath.catalogpath, catalog_type = 'all', load = True)
    #target_img = ScienceImage('/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T01462/7DT02/g/calib_7DT02_T01462_20250211_041624_g_100.com.fits', telinfo = target_img.telinfo, load = True)
    comparison_catalog = TIPCatalog('/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T01462/7DT02/g/calib_7DT02_T01462_20250211_041624_g_100.com.fits.cat', catalog_type = 'reference', load = True)
    # Selection parameters
    mag_lower: float = None
    mag_upper: float = None
    snr_lower: float = 20
    snr_upper: float = 300
    classstar_lower: float = 0.8
    elongation_upper: float = 30
    elongation_sigma: float = 5
    fwhm_lower: float = 2
    fwhm_upper: float = 15
    fwhm_sigma: float = 5
    flag_upper: int = 1
    maskflag_upper: int = 1
    inner_fraction: float = 0.7  # Fraction of the images
    isolation_radius: float = 5.0
    magnitude_key: str = 'MAG_AUTO'
    flux_key: str = 'FLUX_AUTO'
    fluxerr_key: str = 'FLUXERR_AUTO'
    fwhm_key: str = 'FWHM_IMAGE'
    x_key: str = 'X_IMAGE'
    y_key: str = 'Y_IMAGE'
    classstar_key: str = 'CLASS_STAR'
    elongation_key: str = 'ELONGATION'
    flag_key: str = 'FLAGS'
    maskflag_key: str = 'IMAFLAGS_ISO'

    # Other parameters
    save: bool = True
    verbose: bool = True
    visualize: bool = True
    save_fig: bool = False
    save_refcat: bool = True

    #target_sourcemask = Mask(path = target_img.savepath.maskpath, load = True)
    #target_bkg = Background(path = target_img.savepath.bkgpath, load = True)    
    #target_bkgrms =  Errormap(target_img.savepath.errormappath, load = True)
    #target_mask: Optional[Mask] = None # For masking certain source (such as hot pixels)
    #catalog = Table.read(target_img.savepath.catalogpath, format = 'ascii')
    # cat = self.photometric_calibration(
    #     target_img = target_img,
    #     target_catalog = target_catalog,
    #     max_distance_second = 1.0, 
    #     calculate_color_terms = True,
    #     calculate_mag_terms = True,
    #     visualize_mag_key = 'MAG_APER')
# %%
