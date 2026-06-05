
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
import glob
%matplotlib inline
helper = Helper()

wl  = np.arange(3360, 10220, 20)
spec_tmp = Spectrum(wavelength = np.arange(3360, 10220, 20), flux = np.ones(343), fluxerr = np.ones(343), wavelength_unit = 'AA', flux_unit = 'flamb')
_, pyphot_filters, _, _, _ = spec_tmp.synphot(filterset = ['medium', 'g', 'r', 'i'], visualize = False, visualize_transmission = False)
#%% Load calspec catalog
from astropy.io import ascii
# calspec_catalog.write('./calspec.sacii_fixed_width', format = 'ascii.fixed_width', overwrite = True)
calspec_catalog = ascii.read('./calspec.sacii_fixed_width', format = 'fixed_width')
calspec_catalog_observed = calspec_catalog[calspec_catalog['is_observed'].astype(str) == 'True']
#%%
idx = 0
target = calspec_catalog_observed[idx]
tile_id = target['tileid']
calspec_name = target['names']
calspec_name = calspec_name.replace('-', '_')
calspec_ra = target['ra']
calspec_dec = target['dec']
calspec_coord = SkyCoord(ra = calspec_ra, dec = calspec_dec, unit = ('hourangle', 'deg'))
calspec_ra_deg = calspec_coord.ra.deg
calspec_dec_deg = calspec_coord.dec.deg
#%% Load Corrected GAIAXP catalog
corrected_catalog_path = glob.glob(f'/home/hhchoi1022/ezphot/data/skycatalog/original/GAIAXP_CORR_LAMOST/*{tile_id}.json')[0]
source_corrected_all = json.load(open(corrected_catalog_path, 'r'))
source_corrected_all_tbl = Table(source_corrected_all)
source_corrected_coord = SkyCoord(ra = source_corrected_all_tbl['gaiaxp_photinfo_replenish_ra'], dec = source_corrected_all_tbl['gaiaxp_photinfo_replenish_dec'], unit = ('deg', 'deg'))
calspec_idx = np.argmin(source_corrected_coord.separation(calspec_coord))
gaiaxp_id = source_corrected_all_tbl[calspec_idx]['gaiaxp_photinfo_replenish_source_id']
source_corrected = source_corrected_all[calspec_idx]
#%%
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
row_dict_corr = {
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
orig_dict, _, _, _, _ = spec_original.synphot(filterset=['medium', 'g', 'r', 'i'], visualize=False, 
                                                pyphot_filters=pyphot_filters)
corr_dict, _, _, _, _ = spec_corrected.synphot(filterset=['medium', 'g', 'r', 'i'], visualize=False, 
                                                pyphot_filters=pyphot_filters)

for filt in orig_dict.keys():
    row_dict[f'{filt}_mag'] = orig_dict[filt]['mag']
    row_dict[f'{filt}_magerr'] = orig_dict[filt]['mag_err']
    row_dict_corr[f'{filt}_mag'] = corr_dict[filt]['mag']
    row_dict_corr[f'{filt}_magerr'] = corr_dict[filt]['mag_err']

#%% GAIAXP (7DT) vs GAIAXP (HUANG)
catalog = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP', catalog_version = f'v1')
catalog_data = catalog.data[catalog.data['id'] == gaiaxp_id]

fig, ax = plt.subplots(figsize = (10, 5))

mag_diff_all = []
ax.plot(wl, spec_original.ab.flux, color = 'k', label = 'Original')
for filter in pyphot_filters.keys():
    mag_key = f'{filter}_mag'
    wl_filter = pyphot_filters[filter].lpivot.value
    if mag_key in catalog_data.colnames and mag_key in row_dict.keys():
        ax.scatter(wl_filter, catalog_data[mag_key][0], color = 'r', marker = 'o')
        ax.scatter(wl_filter, row_dict[mag_key], color = 'b', marker = 'x')
        diff = catalog_data[mag_key][0] - row_dict[mag_key]
        mag_diff_all.append(diff)
        ax.text(wl_filter, 16, f'{diff:.4f}', color = 'b', rotation = 90)
        print(mag_key)
    else:
        print(mag_key, 'not found')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.scatter(0, 0, color = 'r', marker = 'o', label = 'GAIAXP (7DT)')
ax.scatter(0, 0, color = 'b', marker = 'x', label = 'GAIAXP (HUANG)')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.invert_yaxis()
ax.legend()
ax.set_xlabel('Wavelength (AA)')
ax.set_ylabel('Magnitude (AB)')
ax.set_title('GAIAXP (7DT) vs GAIAXP (HUANG, not corrected)')
ax.legend()
plt.show()    
#%% GAIAXP vs GAIAXP_CORR
fig_original, ax_original = spec_original.show(show_flux_unit = 'AB', ax = None, color = 'k', label = 'Original')
fig_corrected, ax_corrected = spec_corrected.show(show_flux_unit = 'AB', ax = ax_original, color = 'r', label = 'Corrected')
ax_original.set_ylim(15.5, 14)
ax_original.legend(loc = 'lower center')
ax_corrected.legend(loc = 'lower center')
ax_diff = ax_original.twinx()
mag_diff = spec_original.ab.flux.value - spec_corrected.ab.flux.value
ax_diff.plot(wl, mag_diff, color = 'g', label = 'Difference')
flux_error = spec_original.ab.uncertainty.array
ax_diff.fill_between(wl, mag_diff - flux_error, mag_diff + flux_error, color = 'g', alpha = 0.2)
ax_diff.set_ylabel('Difference (AB)')
ax_diff.legend(loc = 'upper center')
ax_diff.set_ylim(-0.1, 0.1)
plt.show()
#%% GAIAXP vs GAIAXP_CORR (synphot)
fig_original, ax_original = spec_original.show(show_flux_unit = 'AB', ax = None, color = 'k', label = 'Original')
fig_corrected, ax_corrected = spec_corrected.show(show_flux_unit = 'AB', ax = ax_original, color = 'r', label = 'Corrected')
ax_original.set_ylim(15.5, 14)
ax_original.legend(loc = 'lower center')
ax_corrected.legend(loc = 'lower center')
ax_diff = ax_original.twinx()
mag_diff = spec_original.ab.flux.value - spec_corrected.ab.flux.value
ax_diff.plot(wl, mag_diff, color = 'g', label = 'Difference')
ax_diff.set_ylabel('Difference (AB)')
ax_diff.legend(loc = 'upper center')
ax_diff.set_ylim(-0.1, 0.1)

mag_diff_all = []
for filter in pyphot_filters.keys():
    wl_filter = orig_dict[filter]['wl_pivot'].value * 10
    mag_filter_original = orig_dict[filter]['mag']
    mag_filter_corrected = corr_dict[filter]['mag']
    ax_original.scatter(wl_filter, mag_filter_original, color = 'k', marker = 'o')
    ax_corrected.scatter(wl_filter, mag_filter_corrected, color = 'r', marker = 'x')
    diff = mag_filter_original - mag_filter_corrected
    mag_diff_all.append(diff)
    ax_diff.scatter(wl_filter, diff, color = 'b', marker = 'x')
    ax_diff.text(wl_filter, -0.07, f'{diff:.4f}', color = 'b', rotation = 90)
    print(filter, diff)
plt.show()
#%%
print(np.mean(mag_diff_all), np.median(mag_diff_all), np.std(mag_diff_all))
#%%
#%% GAIAXP vs GAIAXP_CORR vs Calspec
import glob
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
wl_range = (3000, 9500)
tbl1_mask = (tbl1['WAVELENGTH'] > wl_range[0]) & (tbl1['WAVELENGTH'] < wl_range[1])
tbl2_mask = (tbl2['WAVELENGTH'] > wl_range[0]) & (tbl2['WAVELENGTH'] < wl_range[1])
wl1 = np.array(tbl1['WAVELENGTH'])[tbl1_mask]
wl2 = np.array(tbl2['WAVELENGTH'])[tbl2_mask]
flux1 = np.array(tbl1['FLUX'])[tbl1_mask]
flux2 = np.array(tbl2['FLUX'])[tbl2_mask]
spec1 = Spectrum(wavelength = wl1, flux = flux1, wavelength_unit = 'AA', flux_unit = 'flamb')
spec2 = Spectrum(wavelength = wl2, flux = flux2, wavelength_unit = 'AA', flux_unit = 'flamb')
#%%
synphot_result1, pyphot_filters1, fig_calspec1, ax_mag_calspec1, ax_transmission_calspec1 = spec1.synphot(filterset = ['medium', 'g', 'r', 'i'], visualize = True, visualize_spectrum = False, visualize_transmission = False, pyphot_filters = pyphot_filters)
synphot_result2, pyphot_filters2, fig_calspec2, ax_mag_calspec2, ax_transmission_calspec2 = spec2.synphot(filterset = ['medium', 'g', 'r', 'i'], visualize = True, visualize_spectrum = False, visualize_transmission = False, pyphot_filters = pyphot_filters)
#%%
fig_original, ax_original = spec_original.show(show_flux_unit = 'AB', ax = None, color = 'k', label = 'Original')
fig_corrected, ax_corrected = spec_corrected.show(show_flux_unit = 'AB', ax = ax_original, color = 'r', label = 'Corrected')
#%%

fig, ax_corrected = plt.subplots(figsize = (10, 8))
#ax_corrected.plot(wl, spec_original.ab.flux, color = 'k', label = 'Original', alpha = 0.3)
ax_corrected.plot(wl1, spec1.ab.flux, color = 'k', label = 'Calspec Spectrum', alpha = 0.3)


for filter in pyphot_filters.keys():
    wl_filter = orig_dict[filter]['wl_pivot'].value * 10
    mag_filter_original = orig_dict[filter]['mag']
    mag_filter_corrected = corr_dict[filter]['mag']
    synphot_filter_original = synphot_result1[filter]['mag']
    
    ax_corrected.scatter(wl_filter, mag_filter_original, edgecolor = 'k', facecolor = 'none', marker = 'o')
    ax_corrected.scatter(wl_filter, mag_filter_corrected, edgecolor = 'r', facecolor = 'none', marker = 'o')
    ax_corrected.scatter(wl_filter, synphot_filter_original, edgecolor = 'orange', facecolor = 'none', marker = 'D')

ax_corrected.scatter(0,0, color = 'orange', marker = 'D', label = 'Calspec (synphot)')
ax_corrected.scatter(0,0, edgecolor = 'k', facecolor = 'none', marker = 'o', label = 'GaiaXP (Original)')
ax_corrected.scatter(0,0, edgecolor = 'r', facecolor = 'none', marker = 'o', label = 'GaiaXP (Corrected)')
ax_corrected.set_xlim(3500, 9500)
ax_corrected.legend(loc = 'upper right', fontsize = 15)
ax_corrected.set_ylim(18, 16)
ax_corrected.set_xlabel('Wavelength (AA)', fontsize = 15)
ax_corrected.set_ylabel('Magnitude (AB)', fontsize = 15)
plt.show()
#%%
print(np.mean(mag_diff_all), np.median(mag_diff_all), np.std(mag_diff_all))
# %%

fig, ax_diff = plt.subplots(figsize = (10, 5))
#ax_corrected.plot(wl, spec_original.ab.flux, color = 'k', label = 'Original', alpha = 0.3)
ax_diff.set_xlabel('Wavelength (AA)')
ax_diff.set_ylabel('Difference (AB)')
ax_diff.legend(loc = 'upper center')
ax_diff.set_ylim(-0.1, 0.1)

mag_diff_original_all = []
mag_diff_corrected_all = []
for filter in pyphot_filters.keys():
    wl_filter = orig_dict[filter]['wl_pivot'].value * 10
    mag_filter_original = orig_dict[filter]['mag']
    mag_filter_corrected = corr_dict[filter]['mag']
    synphot_filter_original = synphot_result1[filter]['mag']
    diff_original = mag_filter_original - synphot_filter_original
    diff_corrected = mag_filter_corrected - synphot_filter_original
    mag_diff_original_all.append(diff_original)
    mag_diff_corrected_all.append(diff_corrected)

    ax_diff.scatter(wl_filter, diff_original, edgecolor = 'k', facecolor = 'none', marker = 'o')
    ax_diff.scatter(wl_filter, diff_corrected, edgecolor = 'r', facecolor = 'none', marker = 'o')
    print(filter, diff_original, diff_corrected)
mean, median, std = np.mean(mag_diff_original_all), np.median(mag_diff_original_all), np.std(mag_diff_original_all)
mean_corr, median_corr, std_corr = np.mean(mag_diff_corrected_all), np.median(mag_diff_corrected_all), np.std(mag_diff_corrected_all)

ax_diff.axhline(0, color = 'g', linestyle = '--')
ax_diff.axhline(mean, color = 'k', linestyle = '-')
ax_diff.axhline(mean_corr, color = 'r', linestyle = '-')
ax_diff.axhspan(mean - std, mean + std,
                color='k', alpha=0.2, zorder=0)

ax_diff.axhspan(mean_corr - std_corr, mean_corr + std_corr,
                color='r', alpha=0.2, zorder=0)


ax_diff.scatter(0, 0, color = 'k', marker = 'x', label = rf'$\Delta$m (Original - Calspec): {mean:.4f} ± {std:.4f}')
ax_diff.scatter(0, 0, color = 'r', marker = 'x', label = rf'$\Delta$m (Corrected - Calspec): {mean_corr:.4f} ± {std_corr:.4f}')
ax_diff.legend(loc = 'upper center', fontsize = 15)
ax_diff.set_xlim(3500, 9500)
ax_diff.set_ylim(-0.05, 0.1)
plt.show()
#%%
print(np.mean(mag_diff_all), np.median(mag_diff_all), np.std(mag_diff_all))
# %% Compare GAIAXP (Original) vs GAIAXP (Corrected)
gaiaxp_catalog_uncorrected = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP_CORR_LAMOST', catalog_version = f'v1')
coord_uncorrected = SkyCoord(ra = gaiaxp_catalog_uncorrected.data['ra'], dec = gaiaxp_catalog_uncorrected.data['dec'], unit = ('deg', 'deg'))
gaiaxp_catalog_corrected = SkyCatalog(objname = tile_id, catalog_type = 'GAIAXP', catalog_version = f'v1')
coord_corrected = SkyCoord(ra = gaiaxp_catalog_corrected.data['ra'], dec = gaiaxp_catalog_corrected.data['dec'], unit = ('deg', 'deg'))
matched_idx_uncorrected, matched_idx_corrected, _ = helper.cross_match(coord_uncorrected, coord_corrected, max_distance_second = 5)
gaiaxp_catalog_uncorrected_tbl = gaiaxp_catalog_uncorrected.data[matched_idx_uncorrected]
gaiaxp_catalog_corrected_tbl = gaiaxp_catalog_corrected.data[matched_idx_corrected]
mag_diff_all_filter = dict()
for filter in pyphot_filters.keys():
    mag_key = f'{filter}_mag'
    if mag_key in gaiaxp_catalog_uncorrected_tbl.colnames and mag_key in gaiaxp_catalog_corrected_tbl.colnames:
        mag_uncorrected = gaiaxp_catalog_uncorrected_tbl[mag_key]
        mag_corrected = gaiaxp_catalog_corrected_tbl[mag_key]
        mag_diff = mag_uncorrected - mag_corrected
        mag_diff_all_filter[filter] = mag_diff
    else:
        print(mag_key, 'not found')
#%%
# Sort filters by pivot wavelength
filters_sorted = sorted(
    pyphot_filters.keys(),
    key=lambda f: pyphot_filters[f].lpivot.value
)

data = []
labels = []

for filt in filters_sorted:
    if filt in mag_diff_all_filter.keys():
        mag_diff = mag_diff_all_filter[filt]
        mag_diff = mag_diff[np.isfinite(mag_diff)]  # safety
        data.append(mag_diff)
        labels.append(filt)
    else:
        print(filt, 'not found')


fig, ax = plt.subplots(figsize=(12, 5))

parts = ax.violinplot(
    data,
    positions=np.arange(len(data)) + 1,
    widths=0.8,
    showmeans=False,
    showmedians=True,
    showextrema=False
)

# Style violins
for pc in parts['bodies']:
    pc.set_facecolor('lightsteelblue')
    pc.set_edgecolor('black')
    pc.set_alpha(0.8)

ax.set_xticks(np.arange(len(labels)) + 1)
ax.set_xticklabels(labels, rotation=45)

ax.set_ylabel(r'$\Delta m = m_{\rm uncorrected} - m_{\rm corrected}$')
ax.set_xlabel('Filter')
ax.axhline(0, color='k', ls='--', lw=1, alpha=0.6)
ax.set_ylim(-0.2, 0.2)

ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %%
bp_rp = gaiaxp_catalog_corrected_tbl['bp-rp']
filter = 'm550'
mag_key = f'{filter}_mag'
mag_uncorrected = gaiaxp_catalog_uncorrected_tbl[mag_key]
mag_corrected = gaiaxp_catalog_corrected_tbl[mag_key]
mag_diff = mag_uncorrected - mag_corrected
plt.scatter(bp_rp, mag_diff)
plt.xlabel(r'$(G_{\rm BP}-G_{\rm RP})$')
plt.ylabel(f'{filter} Magnitude Difference')
plt.ylim(-0.005, 0.005)
plt.show()
# %%
np.std(mag_diff)

# %%
