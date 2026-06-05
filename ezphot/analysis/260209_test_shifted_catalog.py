#%% 
#============================================================
# STAR SELECTION
#============================================================
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clipped_stats

def select_stars(target_catalog_data : Table,
                mag_lower: float = 10,
                mag_upper: float = 14,
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
                inner_fraction: float = 0.9, # Fraction of the images
                isolation_radius_arcsec: float = 15.0,
                
                verbose: bool = True,
                visualize: bool = True,
                
                magnitude_key: str = 'MAGSKY_AUTO',
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
    if fwhm_key not in target_catalog_data.keys():
        visualize = False
        helper.print(f"Warning: '{fwhm_key}' not found in target_catalog. Visualization disabled.", verbose)
    if visualize:
        plt.figure(dpi=300)
        plt.xlabel(magnitude_key)
        plt.ylabel(fwhm_key)
        plt.title("Star selection filtering")
        
    def _plot_if_visualize(x, y, color, label, alpha=0.4):
        if visualize:  # or pass `visualize` as a parameter
            plt.scatter(x, y, c=color, alpha=alpha, label=label)
    _plot_if_visualize(target_catalog_data[magnitude_key], target_catalog_data[fwhm_key]*3600, 'k', label = 'All sources', alpha = 0.3)#, c = sources[x_key])
    filtered_catalog_data = target_catalog_data.copy()
    helper.print(f'Initial sources: {len(filtered_catalog_data)}', verbose)
    filter_info = {'initial': len(filtered_catalog_data)}

    # Step 0: FWHM cut: remove too small sources
    if fwhm_key not in filtered_catalog_data.keys():
        helper.print(f"Warning: '{fwhm_key}' not found in target_catalog.", verbose)
    else:
        abs_fwhm_mask = (filtered_catalog_data[fwhm_key] > fwhm_lower_arcsec/3600) & (filtered_catalog_data[fwhm_key] < fwhm_upper_arcsec/3600)
        filtered_catalog_data = filtered_catalog_data[abs_fwhm_mask]
        
        filter_info['after_fwhm_abs'] = len(filtered_catalog_data)
        helper.print(f"[FWHM ABS CUT]: {len(filtered_catalog_data)} sources passed {fwhm_lower_arcsec} < FWHM < {fwhm_upper_arcsec} ", verbose)
    filter_info['after_fwhm_abs'] = len(filtered_catalog_data)
    _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'm', label = 'FWHM(Asvolute) cut', alpha = 0.3)

    # Step 1: Inner region cut        
    if ra_key not in filtered_catalog_data.keys() or dec_key not in filtered_catalog_data.keys():
        helper.print(f"Warning: '{ra_key}' or '{dec_key}' not found in sources.", verbose)
    elif ra_deg is not None and dec_deg is not None and radius_arcsec is not None:
        # If ra_deg, dec_deg, radius_arcsec is given, use them to select the nearby sources
        ra_vals = filtered_catalog_data[ra_key]
        dec_vals = filtered_catalog_data[dec_key]
        catalog_coords = SkyCoord(ra = ra_vals, dec = dec_vals, unit = (u.deg, u.deg))
        input_coords = SkyCoord(ra = ra_deg, dec = dec_deg, unit = (u.deg, u.deg))
        distances = catalog_coords.separation(input_coords)
        nearby_mask = distances < radius_arcsec * u.arcsec
        filtered_catalog_data = filtered_catalog_data[nearby_mask]
        helper.print(f'[NEARBYREGION CUT] {len(filtered_catalog_data)} sources passed within {radius_arcsec} arcsec', verbose)
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
        helper.print(f'[INNERREGION CUT] {len(filtered_catalog_data)} sources passed within {rmax} arcsec', verbose)
    
    filter_info['after_region'] = len(filtered_catalog_data)        
    _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'r', label = 'Region cut', alpha = 0.3)

    # Step 2: Isolation
    if ra_key not in filtered_catalog_data.keys() or dec_key not in filtered_catalog_data.keys():
        helper.print(f"Warning: '{ra_key}' or '{dec_key}' not found in sources.", verbose)
    else:
        coords = SkyCoord(ra = filtered_catalog_data[ra_key], dec = filtered_catalog_data[dec_key], unit = (u.deg, u.deg))
        idx1, idx2, sep2d, _ = coords.search_around_sky(coords, isolation_radius_arcsec * u.arcsec)
        # count neighbors for each object (exclude itself)
        neighbor_count = np.bincount(idx1, minlength=len(coords)) - 1

        isolated_mask = neighbor_count == 0
        filtered_catalog_data = filtered_catalog_data[isolated_mask]
        helper.print(f'[ISOLATION CUT] {len(filtered_catalog_data)} sources passed with isolation radius {isolation_radius_arcsec} arcsec', verbose)
    filter_info['after_isolation'] = len(filtered_catalog_data)

    _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'g', label = 'Isolation cut', alpha = 0.3)

    # Step 3: MAG cut
    if mag_lower is not None and mag_upper is not None:
        if mag_lower is not None:
            filtered_catalog_data = filtered_catalog_data[(filtered_catalog_data[magnitude_key] > mag_lower)]
        if mag_upper is not None:
            filtered_catalog_data = filtered_catalog_data[(filtered_catalog_data[magnitude_key] < mag_upper)]                
        helper.print(f"[MAG CUT]: {len(filtered_catalog_data)} sources passed {mag_lower} < {magnitude_key} < {mag_upper}", verbose)
    else:
        snr = 1.085736/filtered_catalog_data[magnitudeerr_key]
        snr_mask = (snr > snr_lower) & (snr < snr_upper)
        filtered_catalog_data = filtered_catalog_data[snr_mask]
        helper.print(f"[SNR CUT]: {len(filtered_catalog_data)} sources passed {snr_lower} < SNR < {snr_upper}", verbose)
    
    filter_info['after_magcut'] = len(filtered_catalog_data)
        
    _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'b', label = 'MAG cut', alpha = 0.3)

    # Step 4: CLASS_STAR cut
    if classstar_key not in filtered_catalog_data.keys():
        helper.print(f"Warning: '{classstar_key}' not found in sources.", verbose)
    else:
        class_star_mask = filtered_catalog_data[classstar_key] > classstar_lower
        filtered_catalog_data = filtered_catalog_data[class_star_mask]
        helper.print(f"[CLASSSTAR CUT]: {len(filtered_catalog_data)} sources passed CLASS_STAR > {classstar_lower}", verbose)
    filter_info['after_classstar'] = len(filtered_catalog_data)

    _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'cyan', label = 'ClassStar cut', alpha = 0.3)

    # Step 5: FWHM absolute and relative cut
    if fwhm_key not in filtered_catalog_data.keys():
        helper.print(f"Warning: '{fwhm_key}' not found in sources.", verbose)
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
        helper.print(
            f"[FWHM CUT]: {len(filtered_catalog_data)} sources passed {fwhm_lower_arcsec} < FWHM < {fwhm_upper_arcsec} and within ±{fwhm_sigma} sigma"
            f"around median ({fwhm_median*3600:.2f} ± {fwhm_sigma* fwhm_std*3600:.2f}) arcsec",
            verbose
        ) 
        
    _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'orange', label = 'FWHM(Relative) cut', alpha = 0.3)

    # Step 6: Elongation cut
    if elongation_key not in filtered_catalog_data.keys():
        helper.print(f"Warning: '{elongation_key}' not found in sources.", verbose)
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

        helper.print(f"[ELONGATION CUT]: {len(filtered_catalog_data)} passed elongation < {elongation_upper} and within ±{elongation_sigma} sigma of median ({elong_median:.2f} ± {elong_std:.2f})", verbose)

    _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'purple', label = 'Elongation cut', alpha = 0.3)
    
    # Step 7: Flag cut
    if flag_key not in filtered_catalog_data.keys():
        helper.print(f"Warning: '{flag_key}' not found in sources.", verbose)
    else:
        flag_mask = filtered_catalog_data[flag_key] <= flag_upper
        filtered_catalog_data = filtered_catalog_data[flag_mask]
        helper.print(f"[FLAG CUT]: {len(filtered_catalog_data)} sources passed FLAGS <= {flag_upper}", verbose)
    filter_info['after_flag'] = len(filtered_catalog_data)
    _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'magenta', label = 'Flag cut', alpha = 0.3)
    
    # Step 8: Mask flag cut
    if maskflag_key not in filtered_catalog_data.keys():
        helper.print(f"Warning: '{maskflag_key}' not found in sources.", verbose)
    else:
        maskflag_mask = filtered_catalog_data[maskflag_key] <= maskflag_upper
        filtered_catalog_data = filtered_catalog_data[maskflag_mask]
        helper.print(f"[MASKFLAG CUT]: {len(filtered_catalog_data)} sources passed IMAFLAGS_ISO <= {maskflag_upper}", verbose)
    filter_info['after_maskflag'] = len(filtered_catalog_data)
    _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'brown', label = 'MaskFlag cut', alpha = 0.3)

    _plot_if_visualize(filtered_catalog_data[magnitude_key], filtered_catalog_data[fwhm_key]*3600, 'red', label = 'Final selected', alpha = 0.3)

    seeing = np.median(filtered_catalog_data[fwhm_key]) * 3600
    
    if visualize:
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
            helper.print("Warning: No valid magnitudes for setting xlim.", verbose)
            
        if visualize:
            plt.show()
        plt.close()
    return filtered_catalog_data, filter_info, seeing

#%%
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
from scipy.optimize import curve_fit
from pathlib import Path

from ezphot.skycatalog import SkyCatalog
from ezphot.dataobjects import Spectrum
import numpy as np
from ezphot.helper import Helper
from astropy.coordinates import SkyCoord
from astropy.table import Table
from pyphot import unit
from pyphot.phot import Filter
import matplotlib.pyplot as plt

helper = Helper()
spec_tmp = Spectrum(wavelength = np.arange(3360, 10220, 20), flux = np.ones(343), fluxerr = np.ones(343), wavelength_unit = 'AA', flux_unit = 'flamb')
_, pyphot_filters, _, _, _ = spec_tmp.synphot(filterset = ['medium', 'u', 'g', 'r', 'i', 'z'], visualize = False, visualize_transmission = False)
#%%
plt.figure(figsize = (14, 10))
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
cmap = plt.cm.jet
norm = plt.Normalize(vmin = -50, vmax = 50)

shift_map = [-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
slope_map = [-2, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2]
shifted_pyphot_filters_list = dict()
wl_grid = np.arange(3000, 11000 + 1, 1) * unit['AA']
for slope in slope_map:
    
    for shift in shift_map:
        shifted_pyphot_filters = dict()

        for filter_name, pyphot_filter in pyphot_filters.items():

            # 원래 filter
            wl_orig = pyphot_filter.wavelength.value
            trans_orig = pyphot_filter.transmit

            # shift 적용된 wavelength
            wl_shifted = wl_orig + shift

            # interpolation (범위 밖은 0)
            interp = interp1d(
                wl_shifted,
                trans_orig,
                kind='linear',
                bounds_error=False,
                fill_value=0.0
            )

            # 고정 grid에서 재샘플링
            trans_resampled = interp(wl_grid.value)

            # 새 Filter (항상 3000–11000 Å)
            filt_pyphot = Filter(
                wl_grid,
                trans_resampled,
                name=f"{filter_name}_shift{shift:+d}"
            )
            shifted_pyphot_filters[filter_name] = filt_pyphot
            
            
            
            # 시각화 예시
            if filter_name == 'm400':
                plt.plot(
                    wl_grid.value,
                    trans_resampled,
                    label=f'{filter_name} + {shift} Å',
                    color=cmap(norm(shift))
                )

        shifted_pyphot_filters_list[shift] = shifted_pyphot_filters

plt.xlim(3750, 4250)
plt.xlabel('Wavelength (AA)')
plt.ylabel('Transmission')
plt.legend(ncols =2, fontsize = 15, loc = 'lower center')
plt.show()

#%%

plt.figure(figsize=(14, 10))

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from pyphot.phot import Filter

cmap = plt.cm.jet
norm = plt.Normalize(vmin=-3, vmax=3)
shift_map = np.arange(-50, 50.01, 5).astype(int)
# slope_map = np.arange(-10, 10.01, 10.0).astype(int)
slope_map = [-30, -20, -10, 0, 10, 20, 30]
width_map = np.arange(-20, 20.01, 5.0).astype(int)   # percent

#%%
wl_grid = np.arange(3000, 11000 + 1, 1) * unit['AA']

shifted_pyphot_filters_list = dict()

for width in width_map:
    w = width / 100.0
    shifted_pyphot_filters_list[width] = dict()

    for slope in slope_map:
        shifted_pyphot_filters_list[width][slope] = dict()

        for shift in shift_map:
            shifted_pyphot_filters = dict()

            for filter_name, pyphot_filter in pyphot_filters.items():

                wl_orig = pyphot_filter.wavelength.value
                trans_orig = pyphot_filter.transmit
                wl_pivot = pyphot_filter.lpivot.value

                # ------------------------------------
                # 1. WIDTH: stretch around pivot
                # ------------------------------------
                wl_width = wl_pivot + (1.0 + w) * (wl_orig - wl_pivot)

                # ------------------------------------
                # 2. SHIFT
                # ------------------------------------
                wl_shifted = wl_width + shift

                # ------------------------------------
                # 3. RESAMPLE
                # ------------------------------------
                interp = interp1d(
                    wl_shifted,
                    trans_orig,
                    kind='linear',
                    bounds_error=False,
                    fill_value=0.0
                )
                trans_resampled = interp(wl_grid.value)

                # ------------------------------------
                # 4. SLOPE (tilt)
                # ------------------------------------
                slope_factor = (
                    1.0
                    + slope * (wl_grid.value - wl_pivot) / wl_pivot
                )
                trans_tilted = trans_resampled * slope_factor

                # no negative transmission
                trans_tilted = np.clip(trans_tilted, 0.0, None)

                # ------------------------------------
                # 5. CREATE FILTER
                # ------------------------------------
                filt_pyphot = Filter(
                    wl_grid,
                    trans_tilted,
                    name=(
                        f"{filter_name}"
                        f"_w{width:+.1f}"
                        f"_s{shift:+.1f}"
                        f"_sl{slope:+.1f}"
                    )
                )

                shifted_pyphot_filters[filter_name] = filt_pyphot

                # visualization example
                if filter_name == 'm400' and shift == 0 and width == 0:
                    plt.plot(
                        wl_grid.value,
                        trans_tilted,
                        label=f'slope={slope:+.0f}',
                        color=cmap(norm(slope))
                    )

            shifted_pyphot_filters_list[width][slope][shift] = shifted_pyphot_filters

plt.xlim(3750, 4250)
plt.xlabel('Wavelength (Å)')
plt.ylabel('Transmission')
plt.legend(ncols=3, fontsize=12, loc='lower center')
plt.grid(alpha=0.3)
plt.show()



#%%
# --- Main Execution ---
from astropy.io import ascii
calspec_catalog = ascii.read('./calspec.sacii_fixed_width', format = 'fixed_width')
calspec_catalog_observed = calspec_catalog[calspec_catalog['is_observed'].astype(str) == 'True']
target = calspec_catalog_observed[1]
tile_id = target['tileid']
#%%
tile_id = 'T22956'
#%%\
def plot_color_term(
    target_img, ax,
    reference_catalog_tbl, 
    reference_catalog_shifted_tbl,
    catalog_star_coords=None,
    remove_mag_term: bool = False,   # 👈 ADD
    label_original='Original',
    label_shifted='Corrected',
    plot_shifted=True,
    mag_key='MAG_APER_2',
    magerr_key='MAGERR_APER_2'):
    target_catalog = target_img.catalog
    filter = target_img.filter

    # ------------------------------------------------------------
    # Load the reference catalog
    # ------------------------------------------------------------

    # shift = 0
    # catalog_version_shifted = f'v{shift}'
    # reference_catalog = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = 'v0')
    # reference_catalog = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP', catalog_version = 'v1')
    # reference_catalog_shifted = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = catalog_version_shifted)

    # ------------------------------------------------------------
    # Select common reference sources
    # ------------------------------------------------------------
    # build coords for full catalogs (always needed)
    reference_catalog_coord = SkyCoord(
        reference_catalog_tbl['ra'],
        reference_catalog_tbl['dec'],
        unit='deg'
    )
    reference_catalog_shifted_coord = SkyCoord(
        reference_catalog_shifted_tbl['ra'],
        reference_catalog_shifted_tbl['dec'],
        unit='deg'
    )
    reference_inputted = True

    if catalog_star_coords is None:
        # --------------------------------------------------
        # Case 1: NO input reference stars → use common stars
        # --------------------------------------------------
        reference_inputted = False
        catalog_star_coords = reference_catalog_coord

        ref_idx, ref_shift_idx, _ = helper.cross_match(
            catalog_star_coords,
            reference_catalog_shifted_coord,
            max_distance_second=10
        )

        reference_catalog_tbl = reference_catalog_tbl[ref_idx]
        reference_catalog_shifted_tbl = reference_catalog_shifted_tbl[ref_shift_idx]

        # print('Matched objects in the original catalog: ', len(ref_idx))
        # print('Matched objects in the corrected catalog: ', len(ref_shift_idx))

    else:
        # --------------------------------------------------
        # Case 2: user-input reference star coordinates -> use common stars
        # --------------------------------------------------
        _, ref_idx, _ = helper.cross_match(
            catalog_star_coords,
            reference_catalog_coord,
            max_distance_second=10
        )

        _, ref_shift_idx, _ = helper.cross_match(
            catalog_star_coords,
            reference_catalog_shifted_coord,
            max_distance_second=10
        )

        reference_catalog_tbl = reference_catalog_tbl[ref_idx]
        reference_catalog_shifted_tbl = reference_catalog_shifted_tbl[ref_shift_idx]
        reference_catalog_coord = reference_catalog_coord[ref_idx]
        reference_catalog_shifted_coord = reference_catalog_shifted_coord[ref_shift_idx]
        
        ref_idx, ref_shift_idx, _ = helper.cross_match(
            reference_catalog_coord,
            reference_catalog_shifted_coord,
            max_distance_second=10
        )
        
        reference_catalog_tbl = reference_catalog_tbl[ref_idx]
        reference_catalog_shifted_tbl = reference_catalog_shifted_tbl[ref_shift_idx]

        # print('Matched objects in the original catalog: ', len(ref_idx))
        # print('Matched objects in the corrected catalog: ', len(ref_shift_idx))
        
    # ------------------------------------------------------------
    # Calculation of the ZP (rough calculationm)
    # ------------------------------------------------------------

    magref_key = f'{target_catalog.info.filter}_mag'
    target_catalog_tbl = target_catalog.data
    target_catalog_coord = SkyCoord(target_catalog_tbl['X_WORLD'], target_catalog_tbl['Y_WORLD'], unit = 'deg')
    reference_catalog_coord = SkyCoord(reference_catalog_tbl['ra'], reference_catalog_tbl['dec'], unit = 'deg')
    reference_catalog_shifted_coord = SkyCoord(reference_catalog_shifted_tbl['ra'], reference_catalog_shifted_tbl['dec'], unit = 'deg')
    matched_obj_indices, matched_ref_indices, unmatched_obj_indices = helper.cross_match(target_catalog_coord, reference_catalog_coord, max_distance_second = 10)
    matched_obj_indices_corrected, matched_ref_indices_corrected, unmatched_obj_indices_corrected = helper.cross_match(target_catalog_coord, reference_catalog_shifted_coord, max_distance_second = 10)
    # print('Matched objects in the original catalog: ', len(matched_obj_indices))
    # print('Matched objects in the corrected catalog: ', len(matched_obj_indices_corrected))
    matched_target_catalog_tbl = target_catalog_tbl[matched_obj_indices]
    matched_reference_catalog_tbl = reference_catalog_tbl[matched_ref_indices]
    matched_target_catalog_corrected_tbl = target_catalog_tbl[matched_obj_indices_corrected]
    matched_reference_catalog_shifted_tbl = reference_catalog_shifted_tbl[matched_ref_indices_corrected]

    zp = matched_reference_catalog_tbl[magref_key] - matched_target_catalog_tbl[mag_key]
    zp_corrected = matched_reference_catalog_shifted_tbl[magref_key] - matched_target_catalog_corrected_tbl[mag_key]
    zp_median = np.nanmedian(zp)
    zp_corrected_median = np.nanmedian(zp_corrected)
    zp_std = np.nanstd(zp)
    zp_corrected_std = np.nanstd(zp_corrected)
    # print('ZP: ', zp_median, '±', zp_std)
    # print('ZP corrected: ', zp_corrected_median, '±', zp_corrected_std)
    target_catalog_tbl['MAGSKY_AUTO'] = target_catalog_tbl['MAG_AUTO'] + zp_median
    target_catalog_tbl['MAGSKY_AUTO_CORR'] = target_catalog_tbl['MAG_AUTO'] + zp_corrected_median

    # ------------------------------------------------------------
    # 1. Select stars
    # ------------------------------------------------------------
    if filter in ['g', 'r']:
        mag_lower = 13
        mag_upper = 16.5
    elif filter in ['i']:
        mag_lower = 12
        mag_upper = 16
    elif filter in ['m425', 'm450', 'm475', 'm500', 'm525', 'm550', 'm575', 'm600', 'm625', 'm650']:
        mag_lower = 11.5
        mag_upper = 15.5
    else:
        mag_lower = 10.5
        mag_upper = 14.5
    
    if not reference_inputted:
        filtered_catalog_tbl, filter_info, seeing = select_stars(
            target_catalog_tbl,
            mag_lower=mag_lower,
            mag_upper=mag_upper,
            visualize = False
        )
    else:
        filtered_catalog_tbl = target_catalog_tbl
        seeing = target_img.seeing

    filtered_catalog_coord = SkyCoord(
        filtered_catalog_tbl['X_WORLD'],
        filtered_catalog_tbl['Y_WORLD'],
        unit='deg'
    )

    # ------------------------------------------------------------
    # 2. Cross-match (independent; assumes common reference already ensured)
    # ------------------------------------------------------------
    matched_obj_idx, matched_ref_idx, _ = helper.cross_match(
        filtered_catalog_coord,
        reference_catalog_coord,
        max_distance_second=10
    )

    matched_obj_idx_corr, matched_ref_idx_corr, _ = helper.cross_match(
        filtered_catalog_coord,
        reference_catalog_shifted_coord,
        max_distance_second=10
    )

    matched_filtered_catalog_tbl = filtered_catalog_tbl[matched_obj_idx]
    matched_reference_catalog_tbl = reference_catalog_tbl[matched_ref_idx]

    matched_filtered_catalog_corr_tbl = filtered_catalog_tbl[matched_obj_idx_corr]
    matched_reference_catalog_corr_tbl = reference_catalog_shifted_tbl[matched_ref_idx_corr]

    # print(f"Matched (original):  {len(matched_obj_idx)}")
    # print(f"Matched (corrected): {len(matched_obj_idx_corr)}")

    # ------------------------------------------------------------
    # 3. Zeropoint calculation
    # ------------------------------------------------------------
    magref_key = f'{target_catalog.info.filter}_mag'
    magerrref_key = f'{target_catalog.info.filter}_magerr'

    zp = (
        matched_reference_catalog_tbl[magref_key]
        - matched_filtered_catalog_tbl[mag_key]
    )

    zp_corr = (
        matched_reference_catalog_corr_tbl[magref_key]
        - matched_filtered_catalog_corr_tbl[mag_key]
    )
    
    # ------------------------------------------------------------
    # 4. Sigma clipping
    # ------------------------------------------------------------
    sc = SigmaClip(sigma=3, maxiters=0)

    zp_clean = sc(zp)
    if magerrref_key in matched_reference_catalog_tbl.keys():
        zp_clean_err = np.sqrt(matched_reference_catalog_tbl[magerrref_key]**2 + matched_filtered_catalog_tbl[magerr_key]**2)
    else:
        zp_clean_err = np.sqrt(matched_filtered_catalog_tbl[magerr_key]**2)
    zp_corr_clean = sc(zp_corr)
    if magerrref_key in matched_reference_catalog_corr_tbl.keys():
        zp_corr_clean_err = np.sqrt(matched_reference_catalog_corr_tbl[magerrref_key]**2 + matched_filtered_catalog_corr_tbl[magerr_key]**2)
    else:
        zp_corr_clean_err = np.sqrt(matched_filtered_catalog_corr_tbl[magerr_key]**2)

    bp_rp = matched_reference_catalog_tbl['bp-rp']
    reference_catalog_used_tbl = matched_reference_catalog_tbl

    mask = (
        ~zp_clean.mask
        & ~zp_corr_clean.mask
        & np.isfinite(bp_rp)
    )

    zp_clean = zp_clean.data[mask]
    zp_corr_clean = zp_corr_clean.data[mask]
    zp_clean_err = zp_clean_err[mask]
    zp_corr_clean_err = zp_corr_clean_err[mask]
    bp_rp = bp_rp[mask]
    mag_ref = matched_reference_catalog_tbl[magref_key].astype(float)[mask]
    mag_ref_ref = np.nanmedian(mag_ref)

    reference_catalog_used_tbl_clean = reference_catalog_used_tbl[mask]
    reference_catalog_used_coord_clean = SkyCoord(reference_catalog_used_tbl_clean['ra'], reference_catalog_used_tbl_clean['dec'], unit = 'deg')
    # ------------------------------------------------------------
    # 5. Linear model for color term
    # ------------------------------------------------------------
    def linear(x, a, b):
        return a * x + b
    
    popt_mag = None
    popt_magc = None
    if remove_mag_term:
        # fit magnitude term first
        popt_mag, _ = curve_fit(
            linear,
            mag_ref,
            zp_clean,
            sigma=zp_clean_err,
            absolute_sigma=True
        )
        popt_magc, _ = curve_fit(
            linear,
            mag_ref,
            zp_corr_clean,
            sigma=zp_corr_clean_err,
            absolute_sigma=True
        )

        a_mag, b_mag = popt_mag
        a_magc, b_magc = popt_magc

        # subtract magnitude term per star
        zp_for_color = (
            zp_clean
            - a_mag * (mag_ref - mag_ref_ref)
        )
        zp_corr_for_color = (
            zp_corr_clean
            - a_magc * (mag_ref - mag_ref_ref)
        )
    else:
        zp_for_color = zp_clean
        zp_corr_for_color = zp_corr_clean

    popt_clean, pcov_clean = curve_fit(
        linear,
        bp_rp,
        zp_for_color,
        sigma=zp_clean_err,
        absolute_sigma=True
    )

    popt_corr, pcov_corr = curve_fit(
        linear,
        bp_rp,
        zp_corr_for_color,
        sigma=zp_corr_clean_err,
        absolute_sigma=True
    )
    
    zp_orig_median = np.nanmedian(zp_for_color)
    zp_orig_std = np.nanstd(zp_for_color)
    zp_corr_median = np.nanmedian(zp_corr_for_color)
    zp_corr_std = np.nanstd(zp_corr_for_color)

    a_clean, b_clean = popt_clean
    a_corr, b_corr = popt_corr

    bp_rp_ref = np.nanmedian(bp_rp)

    ZP_clean_ref = b_clean + a_clean * bp_rp_ref
    ZP_corr_ref = b_corr + a_corr * bp_rp_ref

    # print("\n--- Color-term fit ---")
    # print(f"Cleaned:   a = {a_clean:.4f}, b = {b_clean:.4f}")
    # print(f"Corrected: a = {a_corr:.4f}, b = {b_corr:.4f}")
    # print("\n--- ZP at median(bp-rp) ---")
    # print(f"ZP cleaned   = {ZP_clean_ref:.4f}")
    # print(f"ZP corrected = {ZP_corr_ref:.4f}")
    # print(f"ΔZP          = {ZP_corr_ref - ZP_clean_ref:.4f}")

    # ------------------------------------------------------------
    # 6. Plot: ZP vs color with fit
    # ------------------------------------------------------------
    xfit = np.linspace(np.min(bp_rp), np.max(bp_rp), 200)
    ax.scatter(bp_rp, zp_for_color, s=18, alpha=0.5, color='k', label=label_original)
    ax.errorbar(bp_rp, zp_for_color, yerr=zp_clean_err, fmt='none', color='k', alpha=0.3)
    ax.plot(xfit, linear(xfit, *popt_clean), color='k', lw=2)

    if plot_shifted:
        ax.scatter(bp_rp, zp_corr_for_color, s=18, alpha=0.5, color='r', label=label_shifted)
        ax.errorbar(bp_rp, zp_corr_for_color, yerr=zp_corr_clean_err, fmt='none', color='r', alpha=0.3)
        ax.plot(xfit, linear(xfit, *popt_corr), color='r', lw=2)
    # print(zp_median, zp_std)
    ax.set_ylim(zp_median - 0.2, zp_median + 0.2) 
    ax.set_xlim(np.min(bp_rp), np.max(bp_rp))
    # ax.set_xlabel(r'$(G_{\rm BP}-G_{\rm RP})$')
    # ax.set_ylabel(r'$ZP = m_{\rm ref} - m_{\rm inst}$')
    ax.legend(loc = 'lower left')
    ax.grid(alpha=0.3)
    ax.set_title(f'{filter, target_img.telname}', fontsize=12)
    return zp_corr_median, zp_corr_std, popt_corr, popt_magc, zp_orig_median, zp_orig_std, popt_clean, reference_catalog_used_coord_clean

#%% Define catalog stars
from ezphot.utils import DataBrowser
fig, ax = plt.subplots(figsize=(7, 5))
dbrowser = DataBrowser('scidata')
dbrowser.observatory = '7DT'
dbrowser.objname = tile_id
target_imgset = dbrowser.search(pattern = '*com*.fits', return_type = 'science')
target_imgset.select_images(filter = 'm500')
target_img = target_imgset.target_images[0]
reference_catalog = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP', catalog_version = 'v1')
reference_catalog_shifted = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = f'w0_sl0_sh0')
zp_median, zp_std, popt_corr, popt_mag, zp_orig_median, zp_orig_std, popt_clean, catalog_star_coords = plot_color_term(target_img, ax, reference_catalog.data, reference_catalog_shifted.data, None,
                label_original='Original',
                label_shifted=f'Corrected',
                mag_key = 'MAG_APER_2',
                magerr_key = 'MAGERR_APER_2',
                plot_shifted=False)
#%%
from astropy.time import Time, TimeDelta
depthlist = [img.depth for img in target_imgset.target_images]
deepest_idx = np.argmax(depthlist)
deepest_obsdate = Time(target_imgset.target_images[deepest_idx].obsdate)
deepest_obsdate_str = deepest_obsdate.datetime.strftime('%Y%m%d')
deepest_obsdate_str_end = Time(deepest_obsdate + 1 * u.day).datetime.strftime('%Y%m%d')
#%%
target_imgset.select_images(obs_start = deepest_obsdate_str, obs_end = deepest_obsdate_str_end)
catalog_star_coords_filter = dict()
catalog_star_coords = None
for target_img in target_imgset.target_images:
    reference_catalog = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP', catalog_version = 'v1')
    reference_catalog_shifted = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = f'w0_sl0_sh0')
    zp_median, zp_std, popt_corr, popt_mag, zp_orig_median, zp_orig_std, popt_clean, catalog_star_coords = plot_color_term(target_img, ax, reference_catalog.data, reference_catalog_shifted.data, catalog_star_coords,
                    label_original='Original',
                    label_shifted=f'Corrected',
                    mag_key = 'MAG_APER_2',
                    magerr_key = 'MAGERR_APER_2',
                    plot_shifted=False)
    catalog_star_coords_filter[target_img.filter] = catalog_star_coords
filters = list(catalog_star_coords_filter.keys())
#%%
# start from the g band
catalog_star_coords_common = catalog_star_coords_filter['g']

for filt in filters[1:]:
    coords_this = catalog_star_coords_filter[filt]

    idx_common, idx_this, _ = helper.cross_match(
        catalog_star_coords_common,
        coords_this,
        max_distance_second=10
    )

    catalog_star_coords_common = catalog_star_coords_common[idx_common]

print(f"Common reference stars across {len(filters)} filters: "
      f"{len(catalog_star_coords_common)}")    
#%% 
filterset_medium1 = ['m400', 'm425', 'm450', 'm475', 'm500', 'm525', 'm550', 'm575', 'm600', 'm625']
filterset_medium2 = ['m650', 'm675', 'm700', 'm725', 'm750', 'm775', 'm800', 'm825', 'm850', 'm875']
filterset_broad = ['g', 'r', 'i']
# filterset_broad = ['g', 'm450']
filterset = filterset_medium1 + filterset_medium2 #+ filterset_broad

dbrowser = DataBrowser('scidata')
dbrowser.observatory = '7DT'
dbrowser.objname = tile_id
target_imgset = dbrowser.search(pattern = '*com*.fits', return_type = 'science')
target_imgset.select_images(filter = filterset)
target_imglist = target_imgset.target_images
# Desired filter order
filter_order = {filt: i for i, filt in enumerate(filterset)}

# Sort target_imglist by filterset order
target_imglist_sorted = sorted(
    target_imglist,
    key=lambda img: filter_order.get(img.filter, 1e9)
)
#%% Visualize m400 as example
ncol = 7
nrow = 3
fig, axes = plt.subplots(
    nrow, ncol,
    figsize=(4.5 * ncol, 4.0 * nrow),
    sharex=False,
    sharey=False
)
axes = axes.flatten()
i = 0
for shift in shift_map:
    reference_catalog = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP', catalog_version = 'v1')
    reference_catalog_shifted = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = f'w0_sl0_sh{shift}')

    ax = axes[i]
    filter = 'm400'
    target_img = target_imglist_sorted[0]
    zp_corr_median, zp_corr_std, popt_corr, popt_mag, zp_orig_median, zp_orig_std, popt_orig, catalog_coord = plot_color_term(target_img, ax, reference_catalog.data, reference_catalog_shifted.data, catalog_star_coords_common,
                    label_original='Original',
                    label_shifted=f'Corrected [Shift: {shift}]',
                    mag_key = 'MAG_APER_2',
                    magerr_key = 'MAGERR_APER_2',
                    plot_shifted=True)
    ax.text(0.05, 0.9, f'ZP = {zp_orig_median:.3f} ± {zp_orig_std:.3f}', transform=ax.transAxes, fontsize = 15, ha='left', va='bottom')
    ax.text(0.05, 0.8, f'ZP_corr = {zp_corr_median:.3f} ± {zp_corr_std:.3f}', transform=ax.transAxes, fontsize = 15, ha='left', va='bottom')
    i += 1
fig.supxlabel(r'$(G_{\rm BP}-G_{\rm RP})$', fontsize = 15)
fig.supylabel(r'$ZP = m_{\rm ref} - m_{\rm inst}$', fontsize = 15)

#%%
ncol = 4
nrow = 2
fig, axes = plt.subplots(
    nrow, ncol,
    figsize=(4.5 * ncol, 4.0 * nrow),
    sharex=False,
    sharey=False
)
axes = axes.flatten()
i = 0
slope_map = np.arange(-30, 30.01, 10.0).astype(int)
for slope in slope_map:
    reference_catalog = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP', catalog_version = 'v1')
    reference_catalog_shifted = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = f'w0_sl{slope}_sh0')

    ax = axes[i]
    filter = 'm400'
    target_img = target_imglist_sorted[0]
    zp_corr_median, zp_corr_std, popt_corr, popt_mag, zp_orig_median, zp_orig_std, popt_orig, catalog_coord = plot_color_term(target_img, ax, reference_catalog.data, reference_catalog_shifted.data, catalog_star_coords_common,
                    label_original='Original',
                    label_shifted=f'Corrected [Slope: {slope}]',
                    mag_key = 'MAG_APER_2',
                    magerr_key = 'MAGERR_APER_2',
                    plot_shifted=True)
    ax.text(0.05, 0.9, f'ZP = {zp_orig_median:.3f} ± {zp_orig_std:.3f}', transform=ax.transAxes, fontsize = 15, ha='left', va='bottom')
    ax.text(0.05, 0.8, f'ZP_corr = {zp_corr_median:.3f} ± {zp_corr_std:.3f}', transform=ax.transAxes, fontsize = 15, ha='left', va='bottom')
    i += 1
fig.supxlabel(r'$(G_{\rm BP}-G_{\rm RP})$', fontsize = 15)
fig.supylabel(r'$ZP = m_{\rm ref} - m_{\rm inst}$', fontsize = 15)
#%%
for j in range(i, len(axes)):
    axes[j].set_visible(False)
#%% Plot ogirinal color term
filterset_medium1 = ['m400', 'm425', 'm450', 'm475', 'm500', 'm525', 'm550', 'm575', 'm600', 'm625']
filterset_medium2 = ['m650', 'm675', 'm700', 'm725', 'm750', 'm775', 'm800', 'm825', 'm850', 'm875']
filterset_broad = ['g', 'r', 'i']
filterset_broad = ['g', 'm450']
filterset = filterset_medium1# + filterset_medium2 + filterset_broad#+ filterset_broad

dbrowser = DataBrowser('scidata')
dbrowser.observatory = '7DT'
dbrowser.objname = tile_id
target_imgset = dbrowser.search(pattern = '*com*.fits', return_type = 'science')
target_imgset.select_images(filter = filterset)
target_imglist = target_imgset.target_images
# Desired filter order
filter_order = {filt: i for i, filt in enumerate(filterset)}

# Sort target_imglist by filterset order
target_imglist_sorted = sorted(
    target_imglist,
    key=lambda img: filter_order.get(img.filter, 1e9)
)

shift = 0
reference_catalog = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP', catalog_version = 'v1')
reference_catalog_shifted = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = f'w0_sl0_sh{shift}')

import math

filters = sorted(pyphot_filters.keys())
n_filter = len(filters)

ncol = 3
nrow = math.ceil(len(target_imglist_sorted) / ncol)
fig, axes = plt.subplots(
    nrow, ncol,
    figsize=(4.5 * ncol, 4.0 * nrow),
    sharex=False,
    sharey=False
)
axes = axes.flatten()
i = 0

for target_img in target_imglist_sorted:
    ax = axes[i]
    filter = target_img.filter
    zp_median, zp_std, popt_corr, popt_mag, zp_orig_median, zp_orig_std, popt_orig, catalog_coord = plot_color_term(target_img, ax, reference_catalog.data, reference_catalog_shifted.data, catalog_star_coords_common,
                    label_original='Original',
                    label_shifted=f'Corrected',
                    mag_key = 'MAG_APER_2',
                    magerr_key = 'MAGERR_APER_2',
                    remove_mag_term = False,
                    plot_shifted=True)
    i += 1
for j in range(i, len(axes)):
    axes[j].set_visible(False)
#%%
fig.supxlabel(r'$(G_{\rm BP}-G_{\rm RP})$', fontsize = 15)
fig.supylabel(r'$ZP = m_{\rm ref} - m_{\rm inst}$', fontsize = 15)

#%%
filterset_medium1 = ['m400', 'm425', 'm450', 'm475', 'm500', 'm525', 'm550', 'm575', 'm600', 'm625']
filterset_medium2 = ['m650', 'm675', 'm700', 'm725', 'm750', 'm775', 'm800', 'm825', 'm850', 'm875']
filterset_broad = ['g', 'r', 'i']
# filterset_broad = ['g', 'm450']
filterset = filterset_medium1 + filterset_medium2 + filterset_broad#+ filterset_broad



dbrowser = DataBrowser('scidata')
dbrowser.observatory = '7DT'
dbrowser.objname = tile_id
target_imgset = dbrowser.search(pattern = '*com*.fits', return_type = 'science')
target_imgset.select_images(filter = filterset)
target_imglist = target_imgset.target_images
# Desired filter order
filter_order = {filt: i for i, filt in enumerate(filterset)}

def run_single_process(target_imgset, width_, slope_, shift_):

    import math
    import gc

    target_imglist = target_imgset.target_images
    target_imglist_sorted = sorted(
        target_imglist,
        key=lambda img: filter_order.get(img.filter, 1e9)
    )

    rows = []

    # 🔥 grid loop 제거 — 한 조합만 처리
    reference_catalog_shifted = SkyCatalog(
        objname=tile_id,
        catalog_type='GAIAXP_CORR_LAMOST',
        catalog_version=f'w{width_}_sl{slope_}_sh{shift_}'
    )

    ncol = 4
    nrow = math.ceil(len(target_imglist_sorted) / ncol)

    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(4.5 * ncol, 4.0 * nrow),
        sharex=False,
        sharey=False
    )
    axes = axes.flatten()

    for i, target_img in enumerate(target_imglist_sorted):

        obsdate_str = Time(target_img.obsdate).datetime.strftime('%Y%m%d')
        ax = axes[i]
        filter = target_img.filter

        zp_median, zp_std, popt_corr, popt_mag, \
        zp_orig_median, zp_orig_std, popt_orig, catalog_coord = plot_color_term(
            target_img,
            ax,
            reference_catalog.data,
            reference_catalog_shifted.data,
            catalog_star_coords_common,
            label_original='Original',
            label_shifted='Corrected',
            mag_key='MAG_APER_2',
            magerr_key='MAGERR_APER_2',
            plot_shifted=True
        )

        rows.append({
            'width': width_,
            'slope': slope_,
            'shift': shift_,
            'filter': filter,
            'zp_median': zp_median,
            'zp_std': zp_std,
            'slope_fit': popt_corr[0],
            'intercept_fit': popt_corr[1],
            'num_reference_stars': len(catalog_coord),
            'telname': target_img.telname,
            'obsdate': obsdate_str,
            'tile_id': tile_id,
        })

    fit_path = f'./gaiaxp_test/{tile_id}/{obsdate_str}/{tile_id}_{obsdate_str}_w{width_}_sl{slope_}_sh{shift_}_color.png'
    Path(fit_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fit_path, dpi=200)   # 200으로 줄이는 것도 추천
    plt.close(fig)

    del fig, axes, reference_catalog_shifted
    gc.collect()

    result_tbl = Table(rows)
    result_tbl.write(
        f'./gaiaxp_test/{tile_id}/{obsdate_str}/result_tbl_{tile_id}_{obsdate_str}_w{width_}_sl{slope_}_sh{shift_}.fits',
        overwrite=True
    )
#%%
from itertools import product
import multiprocessing as mp
from astropy.table import Table, vstack
target_imgsetlist_obsdate = target_imgset.divide_images(by_filter = False, by_obsdate = True)
#%%
slope_map = [-30, -20, 20, 30]
mp.set_start_method("fork", force=True)

param_grid = list(product(width_map, slope_map, shift_map))
#%%
BATCH_SIZE = 96
NPROC = 48

all_output_files = []

# 🔥 여기서 모든 imgset에 대해 반복
for img_idx, target_imgset in enumerate(target_imgsetlist_obsdate):

    print(f"\nProcessing ImageSet {img_idx+1}/{len(target_imgsetlist_obsdate)}")

    for start in range(0, len(param_grid), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(param_grid))
        batch = param_grid[start:end]

        print(f"  Running batch {start+1}-{end}/{len(param_grid)}")

        args_list = [
            (target_imgset, w, s, sh)
            for (w, s, sh) in batch
        ]

        with mp.Pool(processes=NPROC) as pool:
            batch_outputs = pool.starmap(run_single_process, args_list)

        all_output_files.extend(batch_outputs)

        print("  Batch finished.\n")
#%%
# 🔥 마지막에 한 번만 merge
import glob
tile_id = calspec_catalog_observed[0]['tileid']
all__result_files = glob.glob(f'./gaiaxp_test/{tile_id}/*/*.fits')
all_obsdates = glob.glob(f'./gaiaxp_test/{tile_id}/20*')
#%%
from tqdm import tqdm
for obsdate in tqdm(all_obsdates):
    print(f'Processing {obsdate}')
    try:
        result_files = glob.glob(f'{obsdate}/*.fits') 
        tables = [Table.read(f) for f in result_files]
        final_tbl = vstack(tables)
        final_tbl.write(f'{obsdate}/final_result.fits', overwrite=True)
    except Exception as e:
        print(f'Error processing {obsdate}: {e}')
        continue
#%%
all_final_tbl = Table()
for obsdate in all_obsdates:
    final_tbl = Table.read(f'{obsdate}/final_result.fits')
    all_final_tbl = vstack([all_final_tbl, final_tbl])
all_final_tbl.write(f'./gaiaxp_test/{tile_id}/final_result.fits', overwrite=True)
#%%
target_tbl = all_final_tbl[all_final_tbl['obsdate'] != '20250331']
#%%
import numpy as np
import matplotlib.pyplot as plt


import numpy as np

def make_profile_curves_per_filter(result_tbl, filt, J_key='zp_std'):

    t = result_tbl[result_tbl['filter'] == filt]

    shift_vals = np.unique(t['shift'])
    width_vals = np.unique(t['width'])
    slope_vals = np.unique(t['slope'])

    def reduce_func(values):
        if J_key == 'zp_std':
            return np.nanmin(values)
        elif J_key == 'slope_fit':
            if np.all(np.isnan(values)):
                return np.nan
            idx = np.nanargmin(np.abs(values))
            return values[idx]
        else:
            raise ValueError("Unsupported J_key")

    f_shift = np.array([
        reduce_func(t[J_key][t['shift'] == s])
        for s in shift_vals
    ])

    f_width = np.array([
        reduce_func(t[J_key][t['width'] == w])
        for w in width_vals
    ])

    f_slope = np.array([
        reduce_func(t[J_key][t['slope'] == sl])
        for sl in slope_vals
    ])

    return (shift_vals, f_shift), (width_vals, f_width), (slope_vals, f_slope)

#%%
def plot_profiles_for_filter(result_tbl, filt, J_key='zp_std'):
    (xs, ys), (xw, yw), (xsl, ysl) = make_profile_curves_per_filter(result_tbl, filt, J_key=J_key)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(xs, ys, marker='o');  axes[0].set_xlabel('shift'); axes[0].set_ylabel(f'profile min {J_key}')
    axes[1].plot(xw, yw, marker='o');  axes[1].set_xlabel('width'); axes[1].set_ylabel(f'profile min {J_key}')
    axes[2].plot(xsl, ysl, marker='o');axes[2].set_xlabel('slope'); axes[2].set_ylabel(f'profile min {J_key}')
    for ax in axes: ax.grid(alpha=0.3)
    fig.suptitle(f'Filter = {filt}', fontsize=14)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_profiles_for_filter_and_obsdate(result_tbl, filt, obsdate, J_keylist=['zp_std', 'slope_fit']):

    tbl = result_tbl[(result_tbl['filter'] == filt) & (result_tbl['obsdate'] == obsdate)]
    obsdates = np.unique(tbl['obsdate'])

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(obsdates)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for J_key in J_keylist:
        for i, obs in enumerate(obsdates):
            t_obs = tbl[tbl['obsdate'] == obs]

            (xs, ys), (xw, yw), (xsl, ysl) = make_profile_curves_per_filter(
                t_obs, filt, J_key=J_key
            )

            axes[0].plot(xs, ys, marker='o', color=colors[i], label=str(obs))
            axes[1].plot(xw, yw, marker='o', color=colors[i])
            axes[2].plot(xsl, ysl, marker='o', color=colors[i])

    axes[0].set_xlabel('shift')
    axes[1].set_xlabel('width')
    axes[2].set_xlabel('slope')

    # for ax in axes:
    #     ax.set_ylabel(f'profile min {J_key}')
    #     ax.grid(alpha=0.3)

    axes[0].legend(title="obsdate", fontsize=8)

    fig.suptitle(f'Filter = {filt}', fontsize=14)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_profiles_for_filter_and_obsdate(
    result_tbl,
    filt,
    obsdate,
    J_keylist=['zp_std', 'slope_fit']
):

    tbl = result_tbl[
        (result_tbl['filter'] == filt) &
        (result_tbl['obsdate'] == obsdate)
    ]

    if len(tbl) == 0:
        print("No data found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    linestyles = ['-', '--', ':', '-.']

    for j, J_key in enumerate(J_keylist):

        (xs, ys), (xw, yw), (xsl, ysl) = make_profile_curves_per_filter(
            tbl, filt, J_key=J_key
        )

        style = linestyles[j % len(linestyles)]

        axes[0].plot(xs, ys, marker='o', linestyle=style, label=J_key)
        axes[1].plot(xw, yw, marker='o', linestyle=style)
        axes[2].plot(xsl, ysl, marker='o', linestyle=style)

    axes[0].set_xlabel('shift')
    axes[1].set_xlabel('width')
    axes[2].set_xlabel('slope')

    for ax in axes:
        ax.set_ylabel('profile min')
        ax.grid(alpha=0.3)

    axes[0].legend(title="J_key")

    fig.suptitle(f'Filter = {filt}, obsdate = {obsdate}', fontsize=14)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def smooth_poly_fit(x, y, deg=3):
    coeff = np.polyfit(x, y, deg)
    poly = np.poly1d(coeff)
    return poly

def find_minimum_from_poly(poly, x_range):
    dpoly = poly.deriv()
    roots = dpoly.r
    real_roots = roots[np.isreal(roots)].real

    # ?? ? root? ??
    candidates = real_roots[
        (real_roots >= np.min(x_range)) &
        (real_roots <= np.max(x_range))
    ]

    # 2? ?? > 0 (minimum)
    d2 = poly.deriv(2)
    mins = [r for r in candidates if d2(r) > 0]

    if len(mins) == 0:
        return None

    return mins[np.argmin([poly(r) for r in mins])]


def find_zero_cross_from_poly(poly, x_range):
    roots = poly.r
    real_roots = roots[np.isreal(roots)].real

    valid = real_roots[
        (real_roots >= np.min(x_range)) &
        (real_roots <= np.max(x_range))
    ]

    if len(valid) == 0:
        return None

    # 0? ?? ??? root ??
    return valid[np.argmin(np.abs(valid))]

def plot_and_find_best(
    result_tbl,
    filt,
    obsdate,
    fit_degree=3
):

    tbl = result_tbl[
        (result_tbl['filter'] == filt) &
        (result_tbl['obsdate'] == obsdate)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    result = {}

    for i, param in enumerate(['shift', 'width', 'slope']):

        # profile curve ??
        unique_vals = np.unique(tbl[param])
        zp_profile = []
        slope_profile = []

        for val in unique_vals:
            sub = tbl[tbl[param] == val]

            # zp_std ? min
            zp_profile.append(np.nanmin(sub['zp_std']))

            # slope_fit ? |0|? ?? ??? ?
            sl_vals = sub['slope_fit']
            if np.all(np.isnan(sl_vals)):
                slope_profile.append(np.nan)
            else:
                idx = np.nanargmin(np.abs(sl_vals))
                slope_profile.append(sl_vals[idx])


        x = np.array(unique_vals)
        y_zp = np.array(zp_profile)
        y_sl = np.array(slope_profile)

        # --------------------------
        # smoothing
        # --------------------------
        poly_zp = smooth_poly_fit(x, y_zp, deg=fit_degree)
        poly_sl = smooth_poly_fit(x, y_sl, deg=fit_degree)

        x_dense = np.linspace(np.min(x), np.max(x), 300)

        # best values
        best_zp = find_minimum_from_poly(poly_zp, x)
        best_sl = find_zero_cross_from_poly(poly_sl, x)

        result[param] = {
            'zp_best': best_zp,
            'slope_zero': best_sl
        }

        # --------------------------
        # plotting
        # --------------------------
        ax1 = axes[i]
        ax2 = ax1.twinx()

        ax1.plot(x, y_zp, 'o', color='C0')
        ax1.plot(x_dense, poly_zp(x_dense), '-', color='C0', label='zp_std fit')
        ax1.set_ylabel("zp_std", color='C0')

        ax2.plot(x, y_sl, 's', color='C1')
        ax2.plot(x_dense, poly_sl(x_dense), '--', color='C1', label='slope_fit fit')
        ax2.set_ylabel("slope_fit", color='C1')

        if best_zp is not None:
            ax1.axvline(best_zp, color='C0', linestyle=':')
        if best_sl is not None:
            ax1.axvline(best_sl, color='C1', linestyle=':')

        ax1.set_xlabel(param)
        ax1.grid(alpha=0.3)

    fig.suptitle(f"{filt} | {obsdate}")
    plt.tight_layout()
    plt.show()

    return result

#%%
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def plot_shift_comparison(
    all_final_tbl,
    filt,
    shift,
    width=0,
    slope=0
):
    """
    Compare shift=0 and shift=X for a given filter.
    Shows:
        - scatter points
        - arrows (variation per night)
        - medians for each case
        - median differences
    """

    # -----------------------------
    # Filter table
    # -----------------------------
    target_tbl = all_final_tbl[all_final_tbl['filter'] == filt]
    target_tbl = target_tbl.to_pandas()

    # datetime conversion
    target_tbl['datetime'] = pd.to_datetime(
        target_tbl['obsdate'].astype(str),
        format='%Y%m%d',
        errors='coerce'
    )

    shift0 = target_tbl[
        (target_tbl['shift'] == 0) &
        (target_tbl['width'] == 0) &
        (target_tbl['slope'] == 0)
    ]

    shiftX = target_tbl[
        (target_tbl['shift'] == shift) &
        (target_tbl['width'] == width) &
        (target_tbl['slope'] == slope)
    ]

    # -----------------------------
    # Find common nights
    # -----------------------------
    common_dates = np.intersect1d(
        shift0['datetime'],
        shiftX['datetime']
    )

    delta_zp = []
    delta_slope = []

    for dt in common_dates:
        zp0 = shift0['zp_std'][shift0['datetime'] == dt].iloc[0]
        zpX = shiftX['zp_std'][shiftX['datetime'] == dt].iloc[0]

        sl0 = shift0['slope_fit'][shift0['datetime'] == dt].iloc[0]
        slX = shiftX['slope_fit'][shiftX['datetime'] == dt].iloc[0]

        delta_zp.append(zpX - zp0)
        delta_slope.append(slX - sl0)

    delta_zp = np.array(delta_zp)
    delta_slope = np.array(delta_slope)

    # -----------------------------
    # Median calculations
    # -----------------------------
    median_zp_0 = np.nanmedian(shift0['zp_std'])
    median_zp_X = np.nanmedian(shiftX['zp_std'])

    median_sl_0 = np.nanmedian(shift0['slope_fit'])
    median_sl_X = np.nanmedian(shiftX['slope_fit'])

    median_dzp = np.nanmedian(delta_zp)
    median_dsl = np.nanmedian(delta_slope)

    # -----------------------------
    # Plotting
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True
    )

    # ===== ZP PANEL =====
    ax1.scatter(
        shift0['datetime'],
        shift0['zp_std'],
        color='k',
        label='shift = 0'
    )

    ax1.scatter(
        shiftX['datetime'],
        shiftX['zp_std'],
        color='r',
        label=f'shift = {shift}'
    )

    # arrows for zp
    for dt in common_dates:
        zp0 = shift0['zp_std'][shift0['datetime'] == dt].iloc[0]
        zpX = shiftX['zp_std'][shiftX['datetime'] == dt].iloc[0]

        ax1.annotate(
            '',
            xy=(dt, zpX),
            xytext=(dt, zp0),
            arrowprops=dict(
                arrowstyle='->',
                color='gray',
                alpha=0.7,
                lw=1.5
            )
        )

    ax1.set_ylabel('zp_std')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # ===== SLOPE PANEL =====
    ax2.scatter(
        shift0['datetime'],
        shift0['slope_fit'],
        marker='D',
        color='k'
    )

    ax2.scatter(
        shiftX['datetime'],
        shiftX['slope_fit'],
        marker='D',
        color='r'
    )

    # arrows for slope
    for dt in common_dates:
        sl0 = shift0['slope_fit'][shift0['datetime'] == dt].iloc[0]
        slX = shiftX['slope_fit'][shiftX['datetime'] == dt][0]

        ax2.annotate(
            '',
            xy=(dt, slX),
            xytext=(dt, sl0),
            arrowprops=dict(
                arrowstyle='->',
                color='blue',
                alpha=0.7,
                lw=1.5
            )
        )

    ax2.set_ylabel('slope_fit')
    ax2.set_xlabel('datetime')
    ax2.grid(alpha=0.3)

    # -----------------------------
    # Median summary text
    # -----------------------------
    fig.text(
        0.4, 0.87,
        f"[shift = 0]\n"
        f"  Median zp = {median_zp_0:+.4f}\n"
        f"  Median slope = {median_sl_0:+.4f}\n\n"
        f"[shift = {shift}]\n"
        f"  Median zp = {median_zp_X:+.4f}\n"
        f"  Median slope = {median_sl_X:+.4f}\n\n"
        f"[Difference]\n"
        f"  Median zp = {median_dzp:+.4f}\n"
        f"  Median slope = {median_dsl:+.4f}\n"
        f"  N nights = {len(common_dates)}",
        fontsize=11,
        verticalalignment='top'
    )

    plt.suptitle(f'{filt} | shift comparison')
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Return values
    # -----------------------------
    return {
        "median_shift0": {
            "zp_std": median_zp_0,
            "slope_fit": median_sl_0
        },
        "median_shiftX": {
            "zp_std": median_zp_X,
            "slope_fit": median_sl_X
        },
        "median_difference": {
            "delta_zp": median_dzp,
            "delta_slope": median_dsl
        },
        "n_nights": len(common_dates)
    }
#%%
plot_shift_comparison(
    all_final_tbl,
    filt='m400',
    shift=0,
    width=0,
    slope=10
)
#%%
filter_ = 'm400'
width = 0
slope = 0
shift = 0
all_final_tbl[(all_final_tbl['filter'] == filter_) & (all_final_tbl['width'] == width) & (all_final_tbl['slope'] == slope) & (all_final_tbl['shift'] == shift)]
#%%

#%%
import matplotlib.pyplot as plt
import numpy as np
filter_ = 'm500'
df = all_final_tbl[(all_final_tbl['filter'] == filter_) & (all_final_tbl['width'] == 0)].to_pandas()
J_min = df['zp_std'].min()
df.sort_values(by='zp_std', inplace=True)
df['dzp_std'] = df['zp_std'] - J_min

pivot = df.pivot_table(
    index='shift',
    columns='slope',
    values='dzp_std',
    aggfunc='min'  # slope 방향으로 최소값
)
plt.figure(figsize=(6,5))

plt.imshow(
    pivot.values,
    origin='lower',
    aspect='auto',
    extent=[
        pivot.columns.min(), pivot.columns.max(),
        pivot.index.min(),   pivot.index.max()
    ]
)

plt.colorbar(label='Δzp_std')
plt.xlabel('slope')
plt.ylabel('shift')
plt.show()

#%%
def plot_magnitude_term(
    target_img,
    ax,
    reference_catalog,
    reference_catalog_shifted,
    catalog_star_coords = None, 
    remove_color_term: bool = True,
    label_original = 'Original', label_shifted = 'Corrected', 
    plot_shifted = True,
    mag_key = 'MAG_APER_2',
    magerr_key = 'MAGERR_APER_2'):
    """
    Plot magnitude-dependent ZP residuals after color-term removal.
    """
    target_catalog = target_img.catalog
    filter = target_img.filter

    # ------------------------------------------------------------
    # Load the reference catalog
    # ------------------------------------------------------------

    # shift = 0
    # catalog_version_shifted = f'v{shift}'
    # reference_catalog = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = 'v0')
    # reference_catalog = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP', catalog_version = 'v1')
    # reference_catalog_shifted = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = catalog_version_shifted)

    # ------------------------------------------------------------
    # Select common reference sources
    # ------------------------------------------------------------
    reference_catalog_tbl = reference_catalog.data
    reference_catalog_shifted_tbl = reference_catalog_shifted.data
    # build coords for full catalogs (always needed)
    reference_catalog_coord = SkyCoord(
        reference_catalog_tbl['ra'],
        reference_catalog_tbl['dec'],
        unit='deg'
    )
    reference_catalog_shifted_coord = SkyCoord(
        reference_catalog_shifted_tbl['ra'],
        reference_catalog_shifted_tbl['dec'],
        unit='deg'
    )
    reference_inputted = True

    if catalog_star_coords is None:
        # --------------------------------------------------
        # Case 1: NO input reference stars → use common stars
        # --------------------------------------------------
        reference_inputted = False
        catalog_star_coords = reference_catalog_coord

        ref_idx, ref_shift_idx, _ = helper.cross_match(
            catalog_star_coords,
            reference_catalog_shifted_coord,
            max_distance_second=10
        )

        reference_catalog_tbl = reference_catalog_tbl[ref_idx]
        reference_catalog_shifted_tbl = reference_catalog_shifted_tbl[ref_shift_idx]

        print('Matched objects in the original catalog: ', len(ref_idx))
        print('Matched objects in the corrected catalog: ', len(ref_shift_idx))

    else:
        # --------------------------------------------------
        # Case 2: user-input reference star coordinates -> use common stars
        # --------------------------------------------------
        _, ref_idx, _ = helper.cross_match(
            catalog_star_coords,
            reference_catalog_coord,
            max_distance_second=10
        )

        _, ref_shift_idx, _ = helper.cross_match(
            catalog_star_coords,
            reference_catalog_shifted_coord,
            max_distance_second=10
        )

        reference_catalog_tbl = reference_catalog_tbl[ref_idx]
        reference_catalog_shifted_tbl = reference_catalog_shifted_tbl[ref_shift_idx]
        reference_catalog_coord = reference_catalog_coord[ref_idx]
        reference_catalog_shifted_coord = reference_catalog_shifted_coord[ref_shift_idx]
        
        ref_idx, ref_shift_idx, _ = helper.cross_match(
            reference_catalog_coord,
            reference_catalog_shifted_coord,
            max_distance_second=10
        )
        
        reference_catalog_tbl = reference_catalog_tbl[ref_idx]
        reference_catalog_shifted_tbl = reference_catalog_shifted_tbl[ref_shift_idx]

        print('Matched objects in the original catalog: ', len(ref_idx))
        print('Matched objects in the corrected catalog: ', len(ref_shift_idx))
        
    # ------------------------------------------------------------
    # Calculation of the ZP (rough calculationm)
    # ------------------------------------------------------------

    magref_key = f'{target_catalog.info.filter}_mag'
    target_catalog_tbl = target_catalog.data
    target_catalog_coord = SkyCoord(target_catalog_tbl['X_WORLD'], target_catalog_tbl['Y_WORLD'], unit = 'deg')
    reference_catalog_coord = SkyCoord(reference_catalog_tbl['ra'], reference_catalog_tbl['dec'], unit = 'deg')
    reference_catalog_shifted_coord = SkyCoord(reference_catalog_shifted_tbl['ra'], reference_catalog_shifted_tbl['dec'], unit = 'deg')
    matched_obj_indices, matched_ref_indices, unmatched_obj_indices = helper.cross_match(target_catalog_coord, reference_catalog_coord, max_distance_second = 10)
    matched_obj_indices_corrected, matched_ref_indices_corrected, unmatched_obj_indices_corrected = helper.cross_match(target_catalog_coord, reference_catalog_shifted_coord, max_distance_second = 10)
    print('Matched objects in the original catalog: ', len(matched_obj_indices))
    print('Matched objects in the corrected catalog: ', len(matched_obj_indices_corrected))
    matched_target_catalog_tbl = target_catalog_tbl[matched_obj_indices]
    matched_reference_catalog_tbl = reference_catalog_tbl[matched_ref_indices]
    matched_target_catalog_corrected_tbl = target_catalog_tbl[matched_obj_indices_corrected]
    matched_reference_catalog_shifted_tbl = reference_catalog_shifted_tbl[matched_ref_indices_corrected]

    zp = matched_reference_catalog_tbl[magref_key] - matched_target_catalog_tbl[mag_key]
    zp_corrected = matched_reference_catalog_shifted_tbl[magref_key] - matched_target_catalog_corrected_tbl[mag_key]
    zp_median = np.nanmedian(zp)
    zp_corrected_median = np.nanmedian(zp_corrected)
    zp_std = np.nanstd(zp)
    zp_corrected_std = np.nanstd(zp_corrected)
    print('ZP: ', zp_median, '±', zp_std)
    print('ZP corrected: ', zp_corrected_median, '±', zp_corrected_std)
    target_catalog_tbl['MAGSKY_AUTO'] = target_catalog_tbl['MAG_AUTO'] + zp_median
    target_catalog_tbl['MAGSKY_AUTO_CORR'] = target_catalog_tbl['MAG_AUTO'] + zp_corrected_median


    # ------------------------------------------------------------
    # 1. Select stars
    # ------------------------------------------------------------
    if filter in ['g', 'r']:
        mag_lower = 13
        mag_upper = 16.5
    elif filter in ['i']:
        mag_lower = 12
        mag_upper = 16
    elif filter in ['m425', 'm450', 'm475', 'm500', 'm525', 'm550', 'm575', 'm600', 'm625', 'm650']:
        mag_lower = 11.5
        mag_upper = 15.5
    else:
        mag_lower = 10.5
        mag_upper = 14.5
    if not reference_inputted:
        filtered_catalog_tbl, filter_info, seeing = select_stars(
            target_catalog_tbl,
            mag_lower=mag_lower,
            mag_upper=mag_upper,
            visualize = False
        )
    else:
        filtered_catalog_tbl = target_catalog_tbl
        seeing = target_img.seeing

    filtered_catalog_coord = SkyCoord(
        filtered_catalog_tbl['X_WORLD'],
        filtered_catalog_tbl['Y_WORLD'],
        unit='deg'
    )

    # ------------------------------------------------------------
    # 2. Cross-match (independent; assumes common reference already ensured)
    # ------------------------------------------------------------
    matched_obj_idx, matched_ref_idx, _ = helper.cross_match(
        filtered_catalog_coord,
        reference_catalog_coord,
        max_distance_second=seeing
    )

    matched_obj_idx_corr, matched_ref_idx_corr, _ = helper.cross_match(
        filtered_catalog_coord,
        reference_catalog_shifted_coord,
        max_distance_second=seeing
    )

    matched_filtered_catalog_tbl = filtered_catalog_tbl[matched_obj_idx]
    matched_reference_catalog_tbl = reference_catalog_tbl[matched_ref_idx]

    matched_filtered_catalog_corr_tbl = filtered_catalog_tbl[matched_obj_idx_corr]
    matched_reference_catalog_corr_tbl = reference_catalog_shifted_tbl[matched_ref_idx_corr]

    print(f"Matched (original):  {len(matched_obj_idx)}")
    print(f"Matched (corrected): {len(matched_obj_idx_corr)}")

    # ------------------------------------------------------------
    # 3. Zeropoint calculation
    # ------------------------------------------------------------
    magref_key = f'{target_catalog.info.filter}_mag'
    magerrref_key = f'{target_catalog.info.filter}_magerr'

    zp = (
        matched_reference_catalog_tbl[magref_key]
        - matched_filtered_catalog_tbl[mag_key]
    )

    zp_corr = (
        matched_reference_catalog_corr_tbl[magref_key]
        - matched_filtered_catalog_corr_tbl[mag_key]
    )

    # ------------------------------------------------------------
    # 4. Sigma clipping
    # ------------------------------------------------------------
    sc = SigmaClip(sigma=3, maxiters=5)

    zp_clean = sc(zp)
    if magerrref_key in matched_reference_catalog_tbl.keys():
        zp_clean_err = np.sqrt(matched_reference_catalog_tbl[magerrref_key]**2 + matched_filtered_catalog_tbl[magerr_key]**2)
    else:
        zp_clean_err = np.sqrt(matched_filtered_catalog_tbl[magerr_key]**2)
    zp_corr_clean = sc(zp_corr)
    if magerrref_key in matched_reference_catalog_corr_tbl.keys():
        zp_corr_clean_err = np.sqrt(matched_reference_catalog_corr_tbl[magerrref_key]**2 + matched_filtered_catalog_corr_tbl[magerr_key]**2)
    else:
        zp_corr_clean_err = np.sqrt(matched_filtered_catalog_corr_tbl[magerr_key]**2)
    
    reference_catalog_used_tbl = matched_reference_catalog_tbl
    bp_rp = reference_catalog_used_tbl['bp-rp']

    mask = (
        ~zp_clean.mask
        & ~zp_corr_clean.mask
        & np.isfinite(bp_rp)
    )

    zp_clean = zp_clean.data[mask]
    zp_corr_clean = zp_corr_clean.data[mask]
    zp_clean_err = zp_clean_err[mask]
    zp_corr_clean_err = zp_corr_clean_err[mask]
    zp_orig_median = np.nanmedian(zp_clean)
    zp_orig_std = np.nanstd(zp_clean)
    zp_corr_median = np.nanmedian(zp_corr_clean)
    zp_corr_std = np.nanstd(zp_corr_clean)
    bp_rp_clean = bp_rp[mask]

    reference_catalog_used_tbl_clean = reference_catalog_used_tbl[mask]
    reference_catalog_used_coord_clean = SkyCoord(reference_catalog_used_tbl_clean['ra'], reference_catalog_used_tbl_clean['dec'], unit = 'deg')

    # ------------------------------------------------------------
    # Color-term fit (needed for removal)
    # ------------------------------------------------------------
    def linear(x, a, b):
        return a * x + b

    # mag_inst = matched_filtered_catalog_tbl[mag_key].astype(float)[mask]
    mag_ref = matched_reference_catalog_tbl[magref_key].astype(float)[mask]
    
    # ------------------------------------------------------------
    # Color-term fit
    # ------------------------------------------------------------
    popt_clean, pcov_clean = curve_fit(linear,
                                    bp_rp_clean,
                                    zp_clean,
                                    sigma=zp_clean_err,
                                    absolute_sigma=True
                                    )
    popt_corr, pcov_corr = curve_fit(linear,
                                    bp_rp_clean,
                                    zp_corr_clean,
                                    sigma=zp_corr_clean_err,
                                    absolute_sigma=True
                                    )
    a_clean, b_clean = popt_clean
    a_corr, b_corr = popt_corr

    bp_rp_ref = np.nanmedian(bp_rp_clean)

    # ------------------------------------------------------------
    # Magnitude-term fit
    # ------------------------------------------------------------

    # original
    if remove_color_term:
        # subtract color term per star
        zp_for_mag = (
            zp_clean
            - a_clean * (bp_rp_clean - bp_rp_ref)
        )
        zp_corr_for_mag = (
            zp_corr_clean
            - a_corr * (bp_rp_clean - bp_rp_ref)
        )
    else:
        # use raw ZP residuals
        zp_for_mag = zp_clean
        zp_corr_for_mag = zp_corr_clean
        
    popt_mag, _ = curve_fit(
        linear,
        mag_ref,
        zp_for_mag,
        sigma=zp_clean_err,
        absolute_sigma=True
    )

    popt_magc, _ = curve_fit(
        linear,
        mag_ref,
        zp_corr_for_mag,
        sigma=zp_corr_clean_err,
        absolute_sigma=True
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    ax.scatter(
        mag_ref, zp_for_mag,
        s=18, alpha=0.5, color='k',
        label=label_original
    )
    ax.errorbar(
        mag_ref, zp_for_mag,
        yerr=zp_clean_err,
        fmt='none', color='k', alpha=0.3
    )

    xfit = np.linspace(mag_ref.min(), mag_ref.max(), 300)
    ax.plot(xfit, linear(xfit, *popt_mag), color='k', lw=2)

    if plot_shifted:
        ax.scatter(
            mag_ref, zp_corr_for_mag,
            s=18, alpha=0.5, color='r',
            label=label_shifted
        )
        ax.errorbar(
            mag_ref, zp_corr_for_mag,
            yerr=zp_corr_clean_err,
            fmt='none', color='r', alpha=0.3
        )
        ax.plot(xfit, linear(xfit, *popt_magc), color='r', lw=2)
        
    ax.legend()
    ax.set_ylim(zp_corr_median - 0.2, zp_corr_median + 0.2)
    ax.set_title(f'{filter}, {target_img.telname}', fontsize=12)
    ax.grid(alpha=0.3)

    return zp_corr_median, zp_corr_std, popt_corr, zp_orig_median, zp_orig_std, popt_clean, reference_catalog_used_coord_clean

#%%
i = 0
filterset_medium1 = ['m400', 'm425', 'm450', 'm475', 'm500', 'm525', 'm550', 'm575', 'm600', 'm625']
filterset_medium2 = ['m650', 'm675', 'm700', 'm725', 'm750', 'm775', 'm800', 'm825', 'm850', 'm875']
filterset_broad = ['g', 'r', 'i']
filterset_broad = ['g', 'm450']
filterset = filterset_broad

from ezphot.utils import DataBrowser
dbrowser = DataBrowser('scidata')
dbrowser.observatory = '7DT'
dbrowser.objname = tile_id
target_imgset = dbrowser.search(pattern = '*com*.fits', return_type = 'science')
target_imgset.select_images(filter = filterset)
target_imglist = target_imgset.target_images
# Desired filter order
filter_order = {filt: i for i, filt in enumerate(filterset)}

# Sort target_imglist by filterset order
target_imglist_sorted = sorted(
    target_imglist,
    key=lambda img: filter_order.get(img.filter, 1e9)
)


reference_catalog = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP', catalog_version = 'v1')
reference_catalog_shifted = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = 'v0')

import math

filters = sorted(pyphot_filters.keys())
n_filter = len(filters)

ncol = 2
nrow = math.ceil(len(target_imglist_sorted) / ncol)
fig, axes = plt.subplots(
    nrow, ncol,
    figsize=(4.5 * ncol, 4.0 * nrow),
    sharex=False,
    sharey=False
)
axes = axes.flatten()
i = 0
for target_img in target_imglist_sorted:
    ax = axes[i]
    plot_magnitude_term(
        target_img,
        ax,
        reference_catalog,
        reference_catalog_shifted,
        catalog_star_coords = catalog_star_coords_common,
        plot_shifted=True,
        remove_color_term=True
    )    
    print(target_img.filter)
    i += 1
#%%
fig.supxlabel(r'$(G_{\rm BP}-G_{\rm RP})$', fontsize = 15)
fig.supylabel(r'$ZP = m_{\rm ref} - m_{\rm inst}$', fontsize = 15)
#%%
# disble axes that are not used
for j in range(i, len(axes)):
    axes[j].set_visible(False)
#%%
fig
# %%
reference_catalog = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP', catalog_version = 'v1')
import math

filters = sorted(pyphot_filters.keys())
n_filter = len(filters)

ncol = 4
nrow = math.ceil(len(target_imglist_sorted) / ncol)
fig, axes = plt.subplots(
    nrow, ncol,
    figsize=(4.5 * ncol, 4.0 * nrow),
    sharex=False,
    sharey=False
)
axes = axes.flatten()
i = 0

for target_img in target_imglist_sorted:
    ax = axes[i]
    filter = target_img.filter
    reference_catalog_shifted = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = f'v{shift_map_filter[filter]}')
    plot_magnitude_term(target_img, ax, reference_catalog, reference_catalog_shifted,
                    label_original='Original',
                    label_shifted=f'Corrected [Shift: {shift_map_filter[filter]}]',
                    mag_key = 'MAG_APER_2',
                    magerr_key = 'MAGERR_APER_2',
                    plot_shifted=True)
    i += 1
fig.supxlabel(r'$(G_{\rm BP}-G_{\rm RP})$', fontsize = 15)
fig.supylabel(r'$ZP = m_{\rm ref} - m_{\rm inst}$', fontsize = 15)

# disble axes that are not used
for j in range(i, len(axes)):
    axes[j].set_visible(False)
#%%
fig
#%%
# import numpy as np
# from scipy.optimize import minimize

# def estimate_sigma_int_mle(resid, sigma):
#     resid = np.asarray(resid)
#     sigma = np.asarray(sigma)

#     good = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)
#     r = resid[good]
#     s = sigma[good]

#     # 최적화 변수는 log(sigma_int)로 두면 sigma_int>0 자동 보장
#     def nll(log_sigint):
#         sigint = np.exp(log_sigint)
#         var = s**2 + sigint**2
#         return 0.5 * np.sum(np.log(2*np.pi*var) + (r**2)/var)

#     # 초기값: 관측 scatter - 측정 scatter 정도
#     sigma_obs = np.std(r, ddof=1)
#     sigma_meas = np.sqrt(np.mean(s**2))
#     sig0 = max(1e-6, np.sqrt(max(0.0, sigma_obs**2 - sigma_meas**2)))

#     out = minimize(nll, x0=np.log(sig0))
#     sigint_hat = float(np.exp(out.x[0]))

#     # (선택) 근사적인 1-sigma 오차: 헤시안(2차 미분) 이용
#     # out.hess_inv는 BFGS에서 근사 inverse Hessian
#     try:
#         var_log = float(out.hess_inv[0, 0])
#         sigint_err = sigint_hat * np.sqrt(var_log)
#     except Exception:
#         sigint_err = np.nan

#     return sigint_hat, sigint_err, out.success

# # 사용 예:
# # residuals = res_clean_final  (color+mag 제거 후 잔차)
# # zp_err = zp_clean_err        (각 점의 zp error)
# sig_int, sig_int_err, ok = estimate_sigma_int_mle(res_clean_final, zp_clean_err)
# print(sig_int, sig_int_err, ok)

# # %%
# Z = zp_clean
# sig = zp_clean_err

# # weighted mean + its uncertainty
# w = 1.0 / sig**2
# Z_mean = np.sum(w * Z) / np.sum(w)
# Z_mean_err = np.sqrt(1.0 / np.sum(w))

# # observed scatter of final residuals
# res = res_clean_final
# res_err = zp_clean_err  # <- 같은 마스크로 만든 에러를 쓰는 게 안전
# sigma_obs = np.std(res, ddof=1)

# # measurement-only scatter
# sigma_meas = np.sqrt(np.mean(res_err**2))

# # intrinsic/unknown scatter
# sigma_int = np.sqrt(max(0.0, sigma_obs**2 - sigma_meas**2))
# print('Original')
# print(f"Z_mean      = {Z_mean:.5f}")
# print(f"Z_mean_err  = {Z_mean_err:.5f}")
# print(f"sigma_obs   = {sigma_obs:.5f}")
# print(f"sigma_meas  = {sigma_meas:.5f}")
# print(f"sigma_int   = {sigma_int:.5f}")

# Z = zp_corr_clean
# sig = zp_corr_clean_err

# w = 1.0 / sig**2
# Z_mean = np.sum(w * Z) / np.sum(w)
# Z_mean_err = np.sqrt(1.0 / np.sum(w))
# res = res_corr_final
# res_err = zp_corr_clean_err
# sigma_obs = np.std(res, ddof=1)
# sigma_meas = np.sqrt(np.mean(res_err**2))
# sigma_int = np.sqrt(max(0.0, sigma_obs**2 - sigma_meas**2))
# print('Corrected')
# print(f"Z_mean      = {Z_mean:.5f}")
# print(f"Z_mean_err  = {Z_mean_err:.5f}")
# print(f"sigma_obs   = {sigma_obs:.5f}")
# print(f"sigma_meas  = {sigma_meas:.5f}")
# print(f"sigma_int   = {sigma_int:.5f}")

# # %%

