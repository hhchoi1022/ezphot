
#%%

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from ezphot.catalog import SourceCatalogDataset, SkyCatalog
from ezphot.dataobjects import PhotometricSpectrum
from ezphot.imageojbects import ScienceImage
from ezphot.helper import Helper
#%% Load Dataset
dataset = SourceCatalogDataset()
succeeded_catalogs, failed_catalogs, skipped_catalogs = dataset.search_catalogs(target_name='NGC6121', search_key = '100.com.fits.cat')
photspec_instance = PhotometricSpectrum(dataset)
# %% Contruct merged catalog over filter
best_catalog = photspec_instance.source_catalogs.catalogs[8]
photspec_instance.update_data(ra_key = 'X_WORLD', 
                              dec_key = 'Y_WORLD', 
                              max_distance_second = 2, 
                              data_keys = ['MAGSKY_PSF_CORR', 'MAGERR_PSF', 'x_fit', 'y_fit'],)

catalog = photspec_instance.data
metadata = photspec_instance.metadata

#%% Select filters for CMD

idx1 = 2
idx2 = 8
target_img = photspec_instance.source_catalogs.catalogs[idx2].target_img
print('Filter 1: ',metadata[idx1]['filter'], 'Filter 2: ', metadata[idx2]['filter'])
filter_1 = metadata[idx1]['filter']
filter_2 = metadata[idx2]['filter']
mag_key_1 = 'MAGSKY_PSF_CORR' + f'_{idx1}'
mag_key_2 = 'MAGSKY_PSF_CORR' + f'_{idx2}'
magerr_key_1 = 'MAGERR_PSF' + f'_{idx1}'
magerr_key_2 = 'MAGERR_PSF' + f'_{idx2}'
image_height, image_width = target_img.data.shape
x_center = image_width / 2
y_center = image_height / 2
# Source positions
x = catalog['x_fit_6']
y = catalog['y_fit_6']
dist_from_center = np.sqrt((x - x_center)**2 + (y - y_center)**2)
catalog['DIST_CENTER'] = dist_from_center
catalog['color'] = catalog[mag_key_1] - catalog[mag_key_2]
#%%

def plot_cmd_with_selected(ax, center_catalog, filter_1, filter_2,
                           mag_key_1, mag_key_2, magerr_key_1, magerr_key_2,
                           selected_source=None, color='red', label='Selected Source', save: bool = False):

    # Compute magnitudes, color, and errors
    mag_1 = center_catalog[mag_key_1]
    mag_2 = center_catalog[mag_key_2]
    magerr_1 = center_catalog[magerr_key_1]
    magerr_2 = center_catalog[magerr_key_2]
    color_arr = mag_1 - mag_2
    e_color = np.sqrt(magerr_1**2 + magerr_2**2)
    plt.title(f'CMD for NGC6121 with {np.sum(~np.isnan(color_arr))} stars', fontsize=16)


    # Sigma clipping for axis limits
    from astropy.stats import sigma_clipped_stats
    mean_x, median_x, std_x = sigma_clipped_stats(color_arr, sigma=3)

    ax.set_xlim(median_x - 1, median_x + 1)
    ax.set_ylim(18, 9.5)  # Inverted y-axis for mag

    # CMD background
    ax.scatter(color_arr, mag_2, c='black', s=5, alpha=0.5, label='Sources', zorder=2)
    ax.errorbar(
        color_arr, mag_2,
        xerr=e_color,
        yerr=magerr_2,
        fmt='none',
        ecolor='gray',
        elinewidth=0.5,
        capsize=1.3,
        alpha=0.7,
        zorder=1
    )

    # Highlight selected source
    if selected_source is not None:
        selected_color = selected_source[mag_key_1] - selected_source[mag_key_2]
        selected_mag = selected_source[mag_key_2]
        ax.scatter(selected_color, selected_mag,
                   s=60, facecolors='none', edgecolors=color,
                   label=label, zorder=3, linewidth=3)

    # Labels
    ax.set_xlabel(f"${filter_1} - {filter_2}$ (mag)", fontsize=14)
    ax.set_ylabel(f"${filter_2}$ (mag)", fontsize=14)
    ax.grid(True, which='major', alpha=0.3)
    ax.minorticks_on()
    ax.tick_params(direction='in', which='both', top=True, right=True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    if save:
        plt.savefig(f'CMD_{filter_1}_{filter_2}.png', dpi=300, bbox_inches='tight')

#%%
center_catalog = catalog[catalog['DIST_CENTER'] < 1000]
fig, ax = plt.subplots(figsize=(10, 8))

plot_cmd_with_selected(
    ax=ax,
    center_catalog=center_catalog,
    filter_1=filter_1,
    filter_2=filter_2,
    mag_key_1=mag_key_1,
    mag_key_2=mag_key_2,
    magerr_key_1=magerr_key_1,
    magerr_key_2=magerr_key_2,
    selected_source=None,
    color='red'
)
#%%
color_lower = 0.4
color_upper = 0.67
mag_lower = 12
mag_upper = 11
color = center_catalog['color']
mag_2 = center_catalog[mag_key_2]
# Filter the catalog based on color and magnitude
filtered_catalog = center_catalog[
    (color > color_lower) & (color < color_upper) &
    (mag_2 > mag_lower) & (mag_2 < mag_upper)
]
#filtered_catalog = filtered_catalog[filtered_catalog['n_detections'] > 15]
print(f'Filtered catalog contains {len(filtered_catalog)} sources.')
filtered_catalog.sort('DIST_CENTER')
#%% 8 
idx = 2
selected_source = filtered_catalog[idx]
fig, ax = plt.subplots(figsize=(10, 8))
plt.title(f'CMD for NGC6121 with {np.sum(~np.isnan(color))} stars', fontsize=16)
plot_cmd_with_selected(
    ax=ax,
    center_catalog=center_catalog,
    filter_1=filter_1,
    filter_2=filter_2,
    mag_key_1=mag_key_1,
    mag_key_2=mag_key_2,
    magerr_key_1=magerr_key_1,
    magerr_key_2=magerr_key_2,
    selected_source=selected_source,
    color='red'
)
print('Selected source distance from center: ', selected_source['DIST_CENTER'])
#%%
coord = selected_source['coord']
print('Target: ', selected_source['match_id'])
result_fig, result_ax, result_tbl = photspec_instance.plot(ra = coord.ra.deg, 
                        dec = coord.dec.deg, 
                        matching_radius_arcsec = 1,
                        flux_key = 'MAGSKY_PSF',
                        fluxerr_key = 'MAGERR_PSF',
                        color_key = 'OBSERVATORY',
                        overplot_gaiaxp = True,
                        overplot_sdss = False,
                        overplot_ps1 = True)
print(selected_source['match_id'])
# %%
_ = best_catalog.show_source(
    target_ra = coord.ra.deg,
    target_dec = coord.dec.deg,
    downsample = 4,
    zoom_radius_pixel = 100
)
# %% COLORED IMAGE



import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import (ImageNormalize, AsinhStretch, MinMaxInterval)
catalogs = photspec_instance.source_catalogs.catalogs
all_seeing = [catalog.target_img.seeing for catalog in catalogs if catalog.target_img is not None]
all_sky = [catalog.target_img.header['SKYBRGHT'] for catalog in catalogs if catalog.target_img is not None]
all_cloudfrac = [catalog.target_img.header['CLUDFRAC'] for catalog in catalogs if catalog.target_img is not None]
#%%

from astropy.coordinates import SkyCoord, EarthLocation, get_body
from astropy.time import Time
from astropy import units as u
from astroplan.moon import moon_illumination

# Define target position
target_img = catalogs[0].target_img
ra = catalogs[0].target_img.ra    # example: in degrees
dec = catalogs[0].target_img.dec      # example: in degrees
target_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

# Define observation time (UTC)
obs_time = Time(catalogs[0].target_img.obsdate, scale='utc')

latitude = target_img.header['SITELAT']
longitude = target_img.header['SITELONG']
elevation = target_img.header['SITEELEV']  # in meters
location = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg, height=elevation*u.m)
# Moon coordinates
moon_coord = get_body('moon', obs_time, location)

# Separation
sep = target_coord.separation(moon_coord)
print(f"Moon separation: {sep.to(u.deg):.2f}")

# Moon phase
illum = moon_illumination(obs_time)
print(f"Moon illumination: {illum:.2%}")
# %%
