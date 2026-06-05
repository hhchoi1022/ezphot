

#%%
from ezphot.skycatalog import SkyCatalog
from ezphot.dataobjects import Spectrum
import json
import numpy as np
from pathlib import Path
from ezphot.helper import Helper
from astropy.coordinates import SkyCoord
from astropy.table import Table
from pyphot import unit
from pyphot.phot import Filter
import matplotlib.pyplot as plt
from astropy.table import vstack
from astropy.time import Time
helper = Helper()

spec_tmp = Spectrum(wavelength = np.arange(3360, 10220, 20), flux = np.ones(343), fluxerr = np.ones(343), wavelength_unit = 'AA', flux_unit = 'flamb')
_, pyphot_filters, _, _, _ = spec_tmp.synphot(filterset = ['medium', 'g', 'r', 'i'], visualize = False, visualize_transmission = False)
#%%
plt.figure(figsize = (14, 10))
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
cmap = plt.cm.jet
norm = plt.Normalize(vmin = -50, vmax = 50)

shift_map = np.arange(-50, 50.01, 5).astype(int)
# slope_map = np.arange(-10, 10.01, 10.0).astype(int)
slope_map = [-30, -20, 20, 30]
width_map = np.arange(-20, 20.01, 5.0).astype(int)   # percent
print(len(shift_map), len(slope_map), len(width_map), 'Total number of shifted catalogs: ', len(shift_map) * len(slope_map) * len(width_map))
#%%
shifted_pyphot_filters_list = dict()

wl_grid = np.arange(3000, 11000 + 1, 1) * unit['AA']

shifted_pyphot_filters_list = dict()

plt.figure(figsize = (14, 10))
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
                        label=f'w={width:+.0f}% s={shift:+.0f}% sl={slope:+.0f}%',
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

import numpy as np
import json
from tqdm import tqdm
from astropy.table import Table
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

# --- The Worker Function ---
def process_catalog_chunk(chunk, pyphot_filters):
    """
    Processes a block of sources. This is much faster than 
    processing one-by-one because of reduced overhead.
    """
    results = []
    # Pre-define wavelength to avoid re-creation
    wl = np.arange(3360, 10220, 20)
    
    for source_corrected in chunk:
        try:
            row_dict = {
                'id': source_corrected.get('gaiaxp_photinfo_replenish_source_id'),
                'ra': source_corrected.get('gaiaxp_photinfo_replenish_ra'),
                'dec': source_corrected.get('gaiaxp_photinfo_replenish_dec'),
                'parallax': source_corrected.get('gaiaxp_photinfo_replenish_parallax'),
                'pmra': source_corrected.get('gaiaxp_photinfo_replenish_pmra'),
                'pmdec': source_corrected.get('gaiaxp_photinfo_replenish_pmdec'),
                'g_mean': source_corrected.get('gaiaxp_photinfo_replenish_g_mag'),
                'bp-rp': source_corrected.get('gaiaxp_photinfo_replenish_bp_mag') - source_corrected.get('gaiaxp_photinfo_replenish_rp_mag'),
            }

            # Faster parsing: split once, convert once
            f_str = source_corrected['correctedxpspecv1_flux'].split(';')
            fe_str = source_corrected['correctedxpspecv1_flux_error'].split(';')
            fc_str = source_corrected['correctedxpspecv1_flux_cor'].split(';')

            flux = np.array(f_str, dtype=float) * 1e2
            fluxerr = np.array(fe_str, dtype=float) * 1e2
            corrected_flux = np.array(fc_str, dtype=float) * 1e2

            # Initialize Spectrum objects
            spec_original = Spectrum(wavelength=wl, flux=flux, fluxerr=fluxerr, 
                                     wavelength_unit='AA', flux_unit='flamb', verbose=False)
            spec_corrected = Spectrum(wavelength=wl, flux=corrected_flux, fluxerr=fluxerr, 
                                      wavelength_unit='AA', flux_unit='flamb', verbose=False)

            # Synthetic Photometry
            orig_dict, _, _, _, _ = spec_original.synphot(filterset=['medium', 'u', 'g', 'r', 'i', 'z'], visualize=False, 
                                                          pyphot_filters=pyphot_filters)
            corr_dict, _, _, _, _ = spec_corrected.synphot(filterset=['medium', 'u', 'g', 'r', 'i', 'z'], visualize=False, 
                                                           pyphot_filters=pyphot_filters)

            for filt in orig_dict.keys():
                # row_dict[f'{filt}_mag'] = orig_dict[filt]['mag']
                row_dict[f'{filt}_mag'] = corr_dict[filt]['mag']
                row_dict[f'{filt}_magerr'] = corr_dict[filt]['mag_err']

            results.append(row_dict)
        except Exception as e:
            # Basic error handling to prevent one bad row from killing the whole process
            continue
            
    return results
#%%
# --- Main Execution ---
from astropy.io import ascii
calspec_catalog = ascii.read('./calspec.sacii_fixed_width', format = 'fixed_width')
calspec_catalog = calspec_catalog[calspec_catalog['is_observed'] == 'True']
#%%
idx = 2
target = calspec_catalog[idx]
tile_id = target['tileid']
#%%
base_path = Path('/home/hhchoi1022/ezphot/data/skycatalog/archive').resolve()
original_catalog_dir = base_path.parent / 'original' / 'GAIAXP_CORR_LAMOST'
original_catalog_path = list(original_catalog_dir.glob(f'*{tile_id}.json'))[0]

rows = {
    'file': [],
    'objname': [],
    'catalog_type': [],
    'ra': [],
    'dec': [],
    'fov_ra': [],
    'fov_dec': [],
    'area': [],
    'file_size_bytes': [],
    'modified_time': [],
    'catalog_version': []
}
#%%
for width_, slope_dict in shifted_pyphot_filters_list.items():
    for slope, shift_dict in slope_dict.items():
        for shift, pyphot_filters in shift_dict.items():
            print(f'Processing width: {width_}, slope: {slope}, shift: {shift}')
            corr_path = base_path / f'GAIAXP_CORR_LAMOST/gaiaxp_dr3_corr_synphot_{tile_id}_w{width_}_sl{slope}_sh{shift}.csv'
            if not corr_path.exists():
        
                from ezphot.dataobjects import Spectrum
                from ezphot.helper import Helper

                # 1. Setup Environment
                helper = Helper()
                
                # 2. Pre-load Filters (Do this once)
                # spec_tmp = Spectrum(wavelength=np.arange(3360, 10220, 20), flux=np.ones(343), 
                #                     fluxerr=np.ones(343), wavelength_unit='AA', flux_unit='flamb')
                # _, pyphot_filters, _, _, _ = spec_tmp.synphot(filterset='medium', visualize=False)

                # 3. Load Data
                with open(original_catalog_path, 'r') as f:
                    original_catalog = json.load(f)
                
                # 4. Prepare Chunks
                # Use as many workers as physical cores (usually half of logical cores for math-heavy tasks)
                n_workers = 48
                # Split the list of dicts into n_workers chunks
                catalog_chunks = np.array_split(original_catalog, n_workers)
                
                # print(f"Starting multiprocessing with {n_workers} workers...")

                # 5. Execute
                table_data = []
                worker_func = partial(process_catalog_chunk, pyphot_filters=pyphot_filters)
                
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # We wrap the map in a list(tqdm(...)) to see progress by CHUNK
                    chunk_results = list(executor.map(worker_func, catalog_chunks))
                    
                    # Flatten the list of lists into a single list of dicts
                    for result_list in chunk_results:
                        table_data.extend(result_list)

                # 6. Create Final Table
                synphot_table = Table(table_data)
                synphot_df = synphot_table.to_pandas()            
                
                synphot_df.to_csv(corr_path, index=False, float_format='%.5f')
            else:
                synphot_table = Table.read(corr_path, format='ascii.csv')
            
            ra_vals = np.asarray(synphot_table['ra'], dtype=float)
            dec_vals = np.asarray(synphot_table['dec'], dtype=float)

            finite = np.isfinite(ra_vals) & np.isfinite(dec_vals)
            ra_vals = ra_vals[finite]
            dec_vals = dec_vals[finite]

            # --- Dec span ---
            ra_min = ra_vals.min()
            ra_max = ra_vals.max()
            dec_min = dec_vals.min()
            dec_max = dec_vals.max()
            height = dec_max - dec_min

            # --- RA span (robust circular method) ---
            ra_sorted = np.sort(ra_vals)
            dra_candidates = np.diff(np.concatenate([ra_sorted, ra_sorted[:1] + 360]))
            dra = 360.0 - np.max(dra_candidates)

            # --- Project RA span ---
            ra_rad = np.deg2rad(ra_vals)
            ra_center = np.rad2deg(np.arctan2(
                np.mean(np.sin(ra_rad)),
                np.mean(np.cos(ra_rad))
            )) % 360
            dec_center = 0.5 * (dec_min + dec_max)
            width = dra * np.cos(np.deg2rad(dec_center))
            area = width * height
            
            rel_path = corr_path.relative_to(base_path)
            rows['file'].append(str(rel_path))
            rows['objname'].append(tile_id)
            rows['catalog_type'].append('GAIAXP_CORR_LAMOST')
            rows['ra'].append(np.round(ra_center, 4))
            rows['dec'].append(np.round(dec_center, 4))
            rows['fov_ra'].append(np.round(width, 4))
            rows['fov_dec'].append(np.round(height, 4))
            rows['area'].append(np.round(area, 4))
            rows['file_size_bytes'].append(int(corr_path.stat().st_size))
            rows['modified_time'].append(Time(corr_path.stat().st_mtime, format='unix').iso)
            rows['catalog_version'].append(f'w{width_}_sl{slope}_sh{shift}')

summary_path = f'/home/hhchoi1022/ezphot/data/skycatalog/archive/summary.ascii_fixed_width'
existing_table = Table.read(summary_path, format='ascii.fixed_width')
new_table = Table(rows)
full_table = vstack([existing_table, new_table])

#%%
full_table.write(summary_path, format='ascii.fixed_width', overwrite=True)
# %%

# ------------------------------------------------------------
# Load the target catalog
# ------------------------------------------------------------
filter = 'm400'
from ezphot.utils import DataBrowser
dbrowser = DataBrowser('scidata')
dbrowser.observatory = '7DT'
dbrowser.objname = 'T22956'
dbrowser.filter = filter
target_imgset = dbrowser.search(pattern = '*.fits', return_type = 'science')
target_imgset.select_images(obs_start = '20250501',
                            obs_end = '20250531')
target_imglist = target_imgset.target_images
target_imglist[0].show(close_fig = False)
#%%
target_img = target_imglist[0]
target_catalog = target_img.catalog
print(target_img.seeing)
#%%
# ------------------------------------------------------------
# Load the reference catalog
# ------------------------------------------------------------
shift_map = {
    50: 'v1',
    45: 'v2',
    40: 'v3',
    35: 'v4',
    30: 'v5',
    25: 'v6',
    20: 'v7',
    15: 'v8',
    10: 'v9',
    5: 'v10',
    0: 'v11',
    -5: 'v12',
    -10: 'v13',
    -15: 'v14',
    -20: 'v15',
    -25: 'v16',
    -30: 'v17',
    -35: 'v18',
    -40: 'v19',
    -45: 'v20',
    -50: 'v21'
}
shift = 30
reference_catalog = SkyCatalog(objname = 'T22956', catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = shift_map[0])
reference_catalog_corrected = SkyCatalog(objname = 'T22956', catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = shift_map[shift])
#%%
# ------------------------------------------------------------
# Select common reference sources
# ------------------------------------------------------------
reference_catalog_tbl = reference_catalog.data
reference_catalog_corrected_tbl = reference_catalog_corrected.data
reference_catalog_coord = SkyCoord(reference_catalog_tbl['ra'], reference_catalog_tbl['dec'], unit = 'deg')
reference_catalog_corrected_coord = SkyCoord(reference_catalog_corrected_tbl['ra'], reference_catalog_corrected_tbl['dec'], unit = 'deg')
reference_cat_indices, reference_cat_corrected_indices, unmatched_cat_indices = helper.cross_match(reference_catalog_coord, reference_catalog_corrected_coord, max_distance_second = target_img.seeing)
reference_catalog_tbl = reference_catalog_tbl[reference_cat_indices]
reference_catalog_corrected_tbl = reference_catalog_corrected_tbl[reference_cat_corrected_indices]
print('Matched objects in the original catalog: ', len(reference_cat_indices))
print('Matched objects in the corrected catalog: ', len(reference_cat_corrected_indices))
#%%
# ------------------------------------------------------------
# Visualize the difference between the original and corrected catalog
# ------------------------------------------------------------
mag_key = f'{filter}_mag'
mag_diff = reference_catalog_tbl[mag_key] - reference_catalog_corrected_tbl[mag_key]
mag_diff_median = np.nanmedian(mag_diff)
mag_diff_std = np.nanstd(mag_diff)

plt.title(f'[{mag_key}] Mag diff = {mag_diff_median:.3f} ± {mag_diff_std:.3f}')
plt.scatter(reference_catalog_tbl['g_mean'], reference_catalog_tbl[mag_key] - reference_catalog_corrected_tbl[mag_key], color = 'red', alpha = 0.5)
plt.ylim(mag_diff_median - 2*mag_diff_std, mag_diff_median + 2*mag_diff_std)
plt.show()
# %%
# ------------------------------------------------------------
# Calculation of the ZP (rough calculationm)
# ------------------------------------------------------------
mag_key = 'MAG_AUTO'
magerr_key = 'MAGERR_AUTO'
magref_key = f'{target_catalog.info.filter}_mag'

target_catalog_tbl = target_catalog.data
target_catalog_coord = SkyCoord(target_catalog_tbl['X_WORLD'], target_catalog_tbl['Y_WORLD'], unit = 'deg')
reference_catalog_coord = SkyCoord(reference_catalog_tbl['ra'], reference_catalog_tbl['dec'], unit = 'deg')
reference_catalog_corrected_coord = SkyCoord(reference_catalog_corrected_tbl['ra'], reference_catalog_corrected_tbl['dec'], unit = 'deg')
matched_obj_indices, matched_ref_indices, unmatched_obj_indices = helper.cross_match(target_catalog_coord, reference_catalog_coord, max_distance_second = target_img.seeing)
matched_obj_indices_corrected, matched_ref_indices_corrected, unmatched_obj_indices_corrected = helper.cross_match(target_catalog_coord, reference_catalog_corrected_coord, max_distance_second = target_img.seeing)
print('Matched objects in the original catalog: ', len(matched_obj_indices))
print('Matched objects in the corrected catalog: ', len(matched_obj_indices_corrected))
matched_target_catalog_tbl = target_catalog_tbl[matched_obj_indices]
matched_reference_catalog_tbl = reference_catalog_tbl[matched_ref_indices]
matched_target_catalog_corrected_tbl = target_catalog_tbl[matched_obj_indices_corrected]
matched_reference_catalog_corrected_tbl = reference_catalog_corrected_tbl[matched_ref_indices_corrected]

zp = matched_reference_catalog_tbl[magref_key] - matched_target_catalog_tbl[mag_key]
zp_corrected = matched_reference_catalog_corrected_tbl[magref_key] - matched_target_catalog_corrected_tbl[mag_key]
zp_median = np.nanmedian(zp)
zp_corrected_median = np.nanmedian(zp_corrected)
zp_std = np.nanstd(zp)
zp_corrected_std = np.nanstd(zp_corrected)
print('ZP: ', zp_median, '±', zp_std)
print('ZP corrected: ', zp_corrected_median, '±', zp_corrected_std)
plt.scatter(matched_target_catalog_tbl['FWHM_WORLD']*3600, zp, color = 'red', alpha = 0.5)
plt.scatter(matched_target_catalog_corrected_tbl['FWHM_WORLD']*3600, zp_corrected, color = 'blue', alpha = 0.5)
plt.ylim(zp_median - 4*zp_std, zp_median + 4*zp_std)
plt.xlim(1.5, 7)
plt.show()
target_catalog_tbl['MAGSKY_AUTO'] = target_catalog_tbl['MAG_AUTO'] + zp_median
target_catalog_tbl['MAGSKY_AUTO_CORR'] = target_catalog_tbl['MAG_AUTO'] + zp_corrected_median
# %%
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

# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
from scipy.optimize import curve_fit

# ------------------------------------------------------------
# 1. Select stars
# ------------------------------------------------------------
filtered_catalog_tbl, filter_info, seeing = select_stars(
    target_catalog_tbl,
    mag_lower=10,
    mag_upper=15
)

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
    reference_catalog_corrected_coord,
    max_distance_second=seeing
)

matched_filtered_catalog_tbl = filtered_catalog_tbl[matched_obj_idx]
matched_reference_catalog_tbl = reference_catalog_tbl[matched_ref_idx]

matched_filtered_catalog_corr_tbl = filtered_catalog_tbl[matched_obj_idx_corr]
matched_reference_catalog_corr_tbl = reference_catalog_corrected_tbl[matched_ref_idx_corr]

print(f"Matched (original):  {len(matched_obj_idx)}")
print(f"Matched (corrected): {len(matched_obj_idx_corr)}")

# ------------------------------------------------------------
# 3. Zeropoint calculation
# ------------------------------------------------------------
mag_key = 'MAG_AUTO'
magerr_key = 'MAGERR_AUTO'
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
sc = SigmaClip(sigma=5, maxiters=1)

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

print(f"ZP (cleaned):          {np.nanmedian(zp_clean):.4f} ± {np.nanstd(zp_clean):.4f}")
print(f"ZP (corrected clean):  {np.nanmedian(zp_corr_clean):.4f} ± {np.nanstd(zp_corr_clean):.4f}")
#%%
# ------------------------------------------------------------
# 5. Linear model for color term
# ------------------------------------------------------------
def linear(x, a, b):
    return a * x + b

popt_clean, pcov_clean = curve_fit(linear,
                                   bp_rp,
                                   zp_clean,
                                   sigma=zp_clean_err,
                                   absolute_sigma=True
                                   )
popt_corr, pcov_corr = curve_fit(
                                linear,
                                bp_rp,
                                zp_corr_clean,
                                sigma=zp_corr_clean_err,
                                absolute_sigma=True
                                )


a_clean, b_clean = popt_clean
a_corr, b_corr = popt_corr

bp_rp_ref = np.nanmedian(bp_rp)

ZP_clean_ref = b_clean + a_clean * bp_rp_ref
ZP_corr_ref = b_corr + a_corr * bp_rp_ref

print("\n--- Color-term fit ---")
print(f"Cleaned:   a = {a_clean:.4f}, b = {b_clean:.4f}")
print(f"Corrected: a = {a_corr:.4f}, b = {b_corr:.4f}")
print("\n--- ZP at median(bp-rp) ---")
print(f"ZP cleaned   = {ZP_clean_ref:.4f}")
print(f"ZP corrected = {ZP_corr_ref:.4f}")
print(f"ΔZP          = {ZP_corr_ref - ZP_clean_ref:.4f}")

# ------------------------------------------------------------
# 6. Plot: ZP vs color with fit
# ------------------------------------------------------------
xfit = np.linspace(np.min(bp_rp), np.max(bp_rp), 200)

plt.figure(figsize=(7, 5))

plt.scatter(bp_rp, zp_clean, s=20, alpha=0.5, color='red', label='Original')
plt.scatter(bp_rp, zp_corr_clean, s=20, alpha=0.5, color='blue', label='Corrected')
plt.errorbar(bp_rp, zp_clean, yerr=zp_clean_err, fmt='None', color='red', alpha=0.4)
plt.errorbar(bp_rp, zp_corr_clean, yerr=zp_corr_clean_err, fmt='None', color='blue', alpha=0.4)

plt.plot(xfit, linear(xfit, *popt_clean), color='red', lw=2)
plt.plot(xfit, linear(xfit, *popt_corr), color='blue', lw=2)

plt.axvline(bp_rp_ref, color='k', ls='--', lw=1, alpha=0.5)

plt.xlabel(r'$(G_{\rm BP}-G_{\rm RP})$')
plt.ylabel(r'$ZP = m_{\rm ref} - m_{\rm inst}$')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Magnitude (instrumental)
# ------------------------------------------------------------
mag_inst = matched_filtered_catalog_tbl[mag_key].astype(float)[mask]

# ------------------------------------------------------------
# 2. Linear model (color term already fitted)
# ------------------------------------------------------------
def linear(x, a, b):
    return a * x + b

# color-term removed residuals
res_clean = zp_clean - linear(bp_rp, *popt_clean)
res_corr  = zp_corr_clean - linear(bp_rp, *popt_corr)

# ------------------------------------------------------------
# 3. Robust linear fit vs magnitude
# ------------------------------------------------------------
popt_mag_clean, pcov_mag_clean = curve_fit(
    linear,
    mag_inst,
    res_clean,
    sigma=zp_clean_err,
    absolute_sigma=True
)

popt_mag_corr, pcov_mag_corr = curve_fit(
    linear,
    mag_inst,
    res_corr,
    sigma=zp_corr_clean_err,
    absolute_sigma=True
)

a_mag_clean, b_mag_clean = popt_mag_clean
a_mag_corr,  b_mag_corr  = popt_mag_corr

# ------------------------------------------------------------
# 4. Plot: residual ZP vs magnitude
# ------------------------------------------------------------
xfit = np.linspace(
    np.nanmin(mag_inst),
    np.nanmax(mag_inst),
    300
)

plt.figure(figsize=(7, 5))

plt.scatter(
    mag_inst, res_clean,
    s=20, alpha=0.4, color='red',
    label='Original'
)
plt.errorbar(
    mag_inst, res_clean,
    yerr=zp_clean_err,
    fmt='None', color='red', alpha=0.4
)

plt.scatter(
    mag_inst, res_corr,
    s=20, alpha=0.4, color='blue',
    label='Corrected'
)
plt.errorbar(
    mag_inst, res_corr,
    yerr=zp_corr_clean_err,
    fmt='None', color='blue', alpha=0.4
)

plt.plot(
    xfit, linear(xfit, *popt_mag_clean),
    color='darkred', lw=2
)

plt.plot(
    xfit, linear(xfit, *popt_mag_corr),
    color='darkblue', lw=2
)

plt.axhline(0, color='k', ls='--', lw=1, alpha=0.5)

plt.xlabel(r'$m_{\rm inst}$')
plt.ylabel(r'$\Delta ZP$ (color-term removed)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. Print summary
# ------------------------------------------------------------
print('--- Magnitude dependence (after color-term removal) ---')
print(f'Original  slope = {a_mag_clean:.3e}, intercept = {b_mag_clean:.4f}')
print(f'Corrected slope = {a_mag_corr:.3e}, intercept = {b_mag_corr:.4f}')

# %%
def linear(x, a, b):
    return a * x + b

# color-term removed residuals
res_clean_color = zp_clean - linear(bp_rp, *popt_clean)
res_corr_color  = zp_corr_clean - linear(bp_rp, *popt_corr)

def robust_mag_fit(x, y, nsig=3.0, niter=3):
    good = np.isfinite(x) & np.isfinite(y)
    x0, y0 = x[good], y[good]

    for _ in range(niter):
        p = np.polyfit(x0, y0, 1)
        resid = y0 - np.polyval(p, x0)
        sig = 1.4826 * np.nanmedian(np.abs(resid))
        keep = np.abs(resid) < nsig * sig
        x0, y0 = x0[keep], y0[keep]

    return p
p_mag_clean = robust_mag_fit(mag_inst, res_clean_color)
p_mag_corr  = robust_mag_fit(mag_inst, res_corr_color)

res_clean_final = res_clean_color - np.polyval(p_mag_clean, mag_inst)
res_corr_final  = res_corr_color  - np.polyval(p_mag_corr,  mag_inst)

from astropy.stats import mad_std

sig_clean_final = mad_std(res_clean_final)
sig_corr_final  = mad_std(res_corr_final)

print('--- Final scatter (color + mag corrected) ---')
print(f'Original  σ_final = {sig_clean_final:.4f} mag')
print(f'Corrected σ_final = {sig_corr_final:.4f} mag')


plt.figure(figsize=(14, 5))

# ------------------------------------------------------------
# (A) ΔZP_final vs magnitude
# ------------------------------------------------------------
plt.subplot(1, 2, 1)

plt.scatter(
    mag_inst, res_clean_final,
    s=20, alpha=0.4, color='red',
    label='Original'
)
plt.scatter(
    mag_inst, res_corr_final,
    s=20, alpha=0.4, color='blue',
    label='Corrected'
)
plt.errorbar(
    mag_inst, res_clean_final,
    yerr=zp_clean_err,
    fmt='None', color='red', alpha=0.4
)
plt.errorbar(
    mag_inst, res_corr_final,
    yerr=zp_corr_clean_err,
    fmt='None', color='blue', alpha=0.4
)
plt.axhline(0, color='k', ls='--', lw=1)

plt.xlabel(r'$m_{\rm inst}$')
plt.ylabel(r'$\Delta ZP_{\rm final}$')
plt.legend()
plt.grid(alpha=0.3)

# ------------------------------------------------------------
# (B) ΔZP_final vs color
# ------------------------------------------------------------
plt.subplot(1, 2, 2)

plt.scatter(
    bp_rp, res_clean_final,
    s=20, alpha=0.4, color='red',
    label='Original'
)
plt.scatter(
    bp_rp, res_corr_final,
    s=20, alpha=0.4, color='blue',
    label='Corrected'
)
plt.errorbar(
    bp_rp, res_clean_final,
    yerr=zp_clean_err,
    fmt='None', color='red', alpha=0.4
)
plt.errorbar(
    bp_rp, res_corr_final,
    yerr=zp_corr_clean_err,
    fmt='None', color='blue', alpha=0.4
)
plt.axhline(0, color='k', ls='--', lw=1)

plt.xlabel(r'$(G_{\rm BP}-G_{\rm RP})$')
plt.ylabel(r'$\Delta ZP_{\rm final}$')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
# %%



import numpy as np
from scipy.optimize import minimize

def estimate_sigma_int_mle(resid, sigma):
    resid = np.asarray(resid)
    sigma = np.asarray(sigma)

    good = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)
    r = resid[good]
    s = sigma[good]

    # 최적화 변수는 log(sigma_int)로 두면 sigma_int>0 자동 보장
    def nll(log_sigint):
        sigint = np.exp(log_sigint)
        var = s**2 + sigint**2
        return 0.5 * np.sum(np.log(2*np.pi*var) + (r**2)/var)

    # 초기값: 관측 scatter - 측정 scatter 정도
    sigma_obs = np.std(r, ddof=1)
    sigma_meas = np.sqrt(np.mean(s**2))
    sig0 = max(1e-6, np.sqrt(max(0.0, sigma_obs**2 - sigma_meas**2)))

    out = minimize(nll, x0=np.log(sig0))
    sigint_hat = float(np.exp(out.x[0]))

    # (선택) 근사적인 1-sigma 오차: 헤시안(2차 미분) 이용
    # out.hess_inv는 BFGS에서 근사 inverse Hessian
    try:
        var_log = float(out.hess_inv[0, 0])
        sigint_err = sigint_hat * np.sqrt(var_log)
    except Exception:
        sigint_err = np.nan

    return sigint_hat, sigint_err, out.success

# 사용 예:
# residuals = res_clean_final  (color+mag 제거 후 잔차)
# zp_err = zp_clean_err        (각 점의 zp error)
sig_int, sig_int_err, ok = estimate_sigma_int_mle(res_clean_final, zp_clean_err)
print(sig_int, sig_int_err, ok)

# %%
Z = zp_clean
sig = zp_clean_err

# weighted mean + its uncertainty
w = 1.0 / sig**2
Z_mean = np.sum(w * Z) / np.sum(w)
Z_mean_err = np.sqrt(1.0 / np.sum(w))

# observed scatter of final residuals
res = res_clean_final
res_err = zp_clean_err  # <- 같은 마스크로 만든 에러를 쓰는 게 안전
sigma_obs = np.std(res, ddof=1)

# measurement-only scatter
sigma_meas = np.sqrt(np.mean(res_err**2))

# intrinsic/unknown scatter
sigma_int = np.sqrt(max(0.0, sigma_obs**2 - sigma_meas**2))
print('Original')
print(f"Z_mean      = {Z_mean:.5f}")
print(f"Z_mean_err  = {Z_mean_err:.5f}")
print(f"sigma_obs   = {sigma_obs:.5f}")
print(f"sigma_meas  = {sigma_meas:.5f}")
print(f"sigma_int   = {sigma_int:.5f}")

Z = zp_corr_clean
sig = zp_corr_clean_err

w = 1.0 / sig**2
Z_mean = np.sum(w * Z) / np.sum(w)
Z_mean_err = np.sqrt(1.0 / np.sum(w))
res = res_corr_final
res_err = zp_corr_clean_err
sigma_obs = np.std(res, ddof=1)
sigma_meas = np.sqrt(np.mean(res_err**2))
sigma_int = np.sqrt(max(0.0, sigma_obs**2 - sigma_meas**2))
print('Corrected')
print(f"Z_mean      = {Z_mean:.5f}")
print(f"Z_mean_err  = {Z_mean_err:.5f}")
print(f"sigma_obs   = {sigma_obs:.5f}")
print(f"sigma_meas  = {sigma_meas:.5f}")
print(f"sigma_int   = {sigma_int:.5f}")

# %%

