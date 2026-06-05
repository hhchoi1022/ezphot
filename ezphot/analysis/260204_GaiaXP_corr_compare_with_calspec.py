

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
%matplotlib inline
helper = Helper()

spec_tmp = Spectrum(wavelength = np.arange(3360, 10220, 20), flux = np.ones(343), fluxerr = np.ones(343), wavelength_unit = 'AA', flux_unit = 'flamb')
_, pyphot_filters, _, _, _ = spec_tmp.synphot(filterset = ['medium', 'u', 'g', 'r', 'i', 'z'], visualize = False, visualize_transmission = False)
#%%
shift_map_filter = {
 'm400': 30,
 'm425': 10,
 'm450': -25,
 'm475': -30,
 'm500': -5,
 'm525': 0,
 'm550': -45,
 'm575': -40,
 'm600': -50,
 'm625': 20,
 'm650': -30,
 'm675': 10,
 'm700': -35,
 'm750': -50,
 'm775': -35,
 'm800': -50,
 'm825': -50,
 'm850': -50,
 'm875': 30,
 'g': 35,
 'r': -40,
 'i': -50}


{'m400': 5,
 'm425': 15,
 'm500': -10,
 'm525': -15,
 'm550': -35,
 'm575': -25,
 'm600': -50,
 'm625': -35,
 'm650': -15,
 'm675': -50,
 'm700': -50,
 'm725': -10,
 'm750': -35,
 'm775': -45,
 'm800': 40,
 'm825': 15,
 'm875': -20}

#%%
plt.figure(figsize = (14, 10))
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
cmap = plt.cm.jet
norm = plt.Normalize(vmin = -50, vmax = 50)

shift_map = [-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
shifted_pyphot_filters_list = dict()
wl_grid = np.arange(3000, 11000 + 1, 1) * unit['AA']
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
        shifted_pyphot_filters[filter_name] = Filter(
            wl_grid,
            trans_resampled,
            name=f"{filter_name}_shift{shift:+d}"
        )

        # 시각화 예시
        if filter_name == 'm425':
            plt.plot(
                wl_grid.value,
                trans_resampled,
                label=f'{filter_name} + {shift} Å',
                color=cmap(norm(shift))
            )

    shifted_pyphot_filters_list[shift] = shifted_pyphot_filters

plt.xlim(4000, 4500)
plt.xlabel('Wavelength (AA)')
plt.ylabel('Transmission')
plt.legend(ncols =2, fontsize = 15, loc = 'lower center')
plt.show()
# %%
from astropy.time import Time
from bridge.connector import GWPortalConnector
from bridge.connector import SQLConnector
from bridge.alertmonitor import AlertProcessor
from bridge.objects import Alert
from ezphot.utils import DataBrowser
calspec_catalog = Table.read('./calspec.sacii_fixed_width', format = 'ascii.fixed_width')
is_exists = []
is_processed = []
alert_instance_list = []
for idx in range(len(calspec_catalog)):
    # Load calspec catalog
    target = calspec_catalog[idx]
    tile_id = target['tileid']

    gwportal_connector = GWPortalConnector()
    db_connector = SQLConnector()
    gwportal_connector.query_type = 'raw'
    query_result = gwportal_connector.query(tile_name = tile_id)
    dbrowser = DataBrowser('scidata')
    dbrowser.observatory = '7DT'
    dbrowser.objname = tile_id
    target_pathlist = dbrowser.search(pattern = '*.fits', return_type = 'path')
    if len(target_pathlist) == 0:
        is_processed.append(False)
    else:
        is_processed.append(True)

    alert_instance = Alert(objname = tile_id)
    alert_instance.trigger_time = Time('2001-01-01')
    alert_instance_list.append(alert_instance)
    print(tile_id)
    if len(query_result) == 0:
        is_exists.append(False)
    else:
        is_exists.append(True)
calspec_catalog['is_observed'] = is_exists
calspec_catalog['is_processed'] = is_processed
#%%
from astropy.io import ascii
# calspec_catalog.write('./calspec.sacii_fixed_width', format = 'ascii.fixed_width', overwrite = True)
calspec_catalog = ascii.read('./calspec.sacii_fixed_width', format = 'fixed_width')
calspec_catalog_observed = calspec_catalog[calspec_catalog['is_observed'].astype(str) == 'True']
#%%
from bridge.alertmonitor import AlertProcessor
from bridge.objects import Alert
from astropy.time import Time
import time
#for idx in range(len(calspec_catalog_observed)):
idx = 0
target = calspec_catalog_observed[idx]
# alert_instance = Alert(objname = target['tileid'])
# alert_instance.trigger_time = Time('2001-01-01')
# processor = AlertProcessor()
# processor.load_files_gwportal(alert_instance = alert_instance)
# processor.target_images = [target_img for target_img in processor.target_images if target_img.is_saved == False]
# processor.pipeline_before_stacking(alert_instance)
# processor.stacking(alert_instance)
# processor.pipeline_after_stacking(alert_instance)

#%%
#%% Load calspec spectrum
import glob
calspec_name = calspec_catalog_observed['star_names'][0]
calspec_name = calspec_name.lower().replace('-', '_')
calspec_files = glob.glob(f'calspec/*{calspec_name}_*')
calspec_file1 = calspec_files[0]
calspec_file2 = calspec_files[1]
tbl1 = Table.read(calspec_file1, format = 'fits')
tbl2 = Table.read(calspec_file2, format = 'fits')
# plt.plot(tbl1['WAVELENGTH'], tbl1['FLUX'], label = 'Calspec 1')
# plt.plot(tbl2['WAVELENGTH'], tbl2['FLUX'], label = 'Calspec 2')
# plt.xlim(0,10000)
# plt.show()
from ezphot.dataobjects import Spectrum
wl_range = (3000, 11000)
tbl1_mask = (tbl1['WAVELENGTH'] > wl_range[0]) & (tbl1['WAVELENGTH'] < wl_range[1])
tbl2_mask = (tbl2['WAVELENGTH'] > wl_range[0]) & (tbl2['WAVELENGTH'] < wl_range[1])
wl1 = np.array(tbl1['WAVELENGTH'])[tbl1_mask]
wl2 = np.array(tbl2['WAVELENGTH'])[tbl2_mask]
flux1 = np.array(tbl1['FLUX'])[tbl1_mask]
flux2 = np.array(tbl2['FLUX'])[tbl2_mask]
spec1 = Spectrum(wavelength = wl1, flux = flux1, wavelength_unit = 'AA', flux_unit = 'flamb')
spec2 = Spectrum(wavelength = wl2, flux = flux2, wavelength_unit = 'AA', flux_unit = 'flamb')
#%%
synphot_result, pyphot_filters, fig_calspec1, ax_mag_calspec1, ax_transmission_calspec1 = spec1.synphot(filterset = ['medium', 'u', 'g', 'r', 'i', 'z'], visualize = True)
synphot_result2, pyphot_filters2, fig_calspec2, ax_mag_calspec2, ax_transmission_calspec2 = spec2.synphot(filterset = ['medium', 'u', 'g', 'r', 'i', 'z'], visualize = True, pyphot_filters = pyphot_filters)
#%%
# ------------------------------------------------------------
# Load the target catalog
# ------------------------------------------------------------
#%%
idx = 1
# filter = 'm400'
filter = None
calspec_target = calspec_catalog_observed[idx]
calspec_name = calspec_target['star_names'].lower()
calspec_ra = calspec_target['ra']
calspec_dec = calspec_target['dec']
calspec_coord = SkyCoord(calspec_ra, calspec_dec, unit = ('hourangle', 'deg'))
calspec_ra_deg = calspec_coord.ra.deg + 2/3600
calspec_dec_deg = calspec_coord.dec.deg
#%%
tile_id = calspec_target['tileid']
filter = None
from ezphot.utils import DataBrowser
dbrowser = DataBrowser('scidata')
dbrowser.observatory = '7DT'
dbrowser.objname = tile_id
if filter is not None:
    dbrowser.filter = filter
target_imgset = dbrowser.search(pattern = '*com.fits', return_type = 'science')
target_catalogset = dbrowser.search(pattern = '*.com.fits.cat', return_type = 'catalog')
target_imglist = target_imgset.target_images
target_imglist[0].show_position(x = calspec_ra_deg, y = calspec_dec_deg, coord_type = 'coord', zoom_radius_pixel = 100)[0]
#%% Plot photometric spctrum from original catalogs
mag_key = 'MAGSKY_APER_2'
fluxerr_key = 'MAGERR_APER_2'
zp_key = 'ZP_APER_2'
zperr_key = 'ZPERR_APER_2'
depth_key = 'UL5SKY_APER_2'
from ezphot.dataobjects import PhotometricSpectrum
photspec = PhotometricSpectrum(catalogset = target_catalogset)
fig, _, ax, photspec_tbl = photspec.plot(ra = calspec_ra_deg, dec = calspec_dec_deg, flux_key = mag_key, fluxerr_key = fluxerr_key, zp_key = zp_key, zperr_key = zperr_key, depth_key = depth_key)
fig = list(fig.values())[0]
ax = list(ax.values())[0]
fig
#%%
mag_diff_all = dict()
for filter_name, result_filter in synphot_result2.items():
    mag = result_filter['mag']
    wl = result_filter['wl_pivot'].value
    if filter_name in set(photspec_tbl['filter']):
        photspec_tbl_mask = photspec_tbl['filter'] == filter_name
        photspec_tbl_mag = photspec_tbl[mag_key][photspec_tbl_mask]
        ax.scatter(wl, mag, edgecolor = 'red', facecolor = 'none', alpha = 1, marker = 'D', zorder = 100, s = 60)
        mag_diff = mag - photspec_tbl_mag
        mag_diff_all[filter_name] = mag_diff
        # Center the text
        ax.text(wl, mag-0.2, f'{mag_diff[0]:.3f}', fontsize = 12, color = 'red', ha = 'center', va = 'center')
ax.scatter(0, 0, edgecolor = 'blue', facecolor = 'none', alpha = 1, marker = 'D', zorder = 100, s = 60, label = 'Calspec(2)')
ax.legend(loc = 'upper left')
plt.show()
fig
#%%
filter = 'm700'
dbrowser = DataBrowser('scidata')
dbrowser.observatory = '7DT'
dbrowser.objname = tile_id
if filter is not None:
    dbrowser.filter = filter
target_imgset = dbrowser.search(pattern = '*com.fits', return_type = 'science')
target_catalogset = dbrowser.search(pattern = '*com.fits.cat', return_type = 'catalog')
target_imglist = target_imgset.target_images
target_imglist[0].show(close_fig = False)
target_img = target_imglist[0]
target_catalog = target_img.catalog
print(target_img.seeing)
#%%
target_catalog.select_sources(x = calspec_ra_deg, y = calspec_dec_deg, matching_radius = 5)
#%%
# ------------------------------------------------------------
# Load the reference catalog
# ------------------------------------------------------------
shift = 0
reference_catalog = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP', catalog_version = f'v1')
reference_catalog_corrected = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = f'v{shift}')
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
%matplotlib inline
filter = 'm450'
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
mag_key = 'MAG_APER_2'
magerr_key = 'MAGERR_APER_2'
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
    mag_lower=11,
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

