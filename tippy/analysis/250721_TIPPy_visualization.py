#%%

import psutil, os
from pympler import asizeof
from tqdm import tqdm
import gc
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib

from tippy.photometry import *
from tippy.imageojbects import *
from tippy.helper import Helper
from tippy.utils import SDTData
from tippy.helper import TIPDataBrowser
# Initialize the classes
Preprocessor = TIPPreprocess()
PlateSolver = TIPPlateSolve()
MaskGenerator = TIPMasking()
BkgGenerator = TIPBackground()
ErrormapGenerator = TIPErrormap()
AperturePhotometry = TIPAperturePhotometry()
PSFPhotometry = TIPPSFPhotometry()
PhotometricCalibration = TIPPhotometricCalibration()
Stacking = TIPStacking()
helper = Helper()
#%%
# LOAD DATA
sdtdata = SDTData()
#sdtdata.sync_scidata(targetname = 'T17274')
data_browser = TIPDataBrowser('scidata')
data_browser.observatory = '7DT'
data_browser.objname = 'T22956'
target_imglist_g = data_browser.search('calib*_g_*100.fits', 'science')
target_imglist_r = data_browser.search('calib*_g_*100.fits', 'science')
stacked_imglist = data_browser.search('calib*com.fits', 'science')
#%% Source Masking
save = True
verbose = True
visualize = True
save_fig = True
do_generatemask = True
#%%
mask_kwargs = dict(
    sigma = 5,
    mask_radius_factor = 1.5,
    saturation_level = 35000,
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig
)

do_circularmask = True
circlemask_kwargs = dict(
    x_position = 233.8476795,
    y_position = 12.0483362,
    radius_arcsec = 150,
    unit = 'deg',
    save = False,
    verbose = True,
    visualize = visualize,
    save_fig = True
)
target_img = target_imglist_g[0]
status = dict()
status['mask'] = None
target_srcmask = None
if do_generatemask:
    try:
        # Mask the object frames
        target_srcmask = MaskGenerator.mask_sources(
            target_img = target_img,
            target_mask = None,
            **mask_kwargs
        )
        status['mask'] = True
    except:
        status['mask'] = e

status['circular_mask'] = None
if do_circularmask:
    try:
        # Generate the circular mask
        target_srcmask = MaskGenerator.mask_circle(
            target_img = target_img,
            target_mask = target_srcmask,
            mask_type = 'source',
            **circlemask_kwargs
        )
        status['circular_mask'] = True

    except Exception as e:
        status['circular_mask'] = e
#%%
# BACKGROUND ESTIMATION
do_generatebkg = True
bkg_kwargs = dict(
    box_size = 256,
    is_2D_bkg = True,
    filter_size = 3,
    n_iterations = 0,
    mask_sigma = 5,
    mask_radius_factor = 1.5,
    mask_saturation_level = 35000,
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig
)
status['background'] = None
target_bkg = None
# With mask
if do_generatebkg:
    try:
        # Generate the background
        target_bkg, bkg_instance = BkgGenerator.calculate_sep(
            target_img = target_img,
            target_mask = target_srcmask,
            ** bkg_kwargs
        )
        status['background'] = True
    except Exception as e:
        status['background'] = e

#%%        
# # Without mask
if do_generatebkg:
    try:
        # Generate the background
        target_bkg, bkg_instance = BkgGenerator.calculate_sep(
            target_img = target_img,
            target_mask = None,
            ** bkg_kwargs
        )
        status['background'] = True
    except Exception as e:
        status['background'] = e
#%%
BkgGenerator.subtract_background(
    target_img = target_img,
    target_bkg = target_bkg,
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig
)
        
# %% Errormap
do_generateerrormap = True
errormap_from_propagation = True
errormap_kwargs = dict(
    readout_noise = target_img.telinfo['readnoise'],
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig
)
errormap_from_sourcemask = False
errormap_mask_kwargs = dict(
    box_size = 256,
    filter_size = 5,
    errormap_type = 'bkgrms',
    mode = 'sep',
    n_iterations = 0,
    mask_sigma = 3,
    mask_radius_factor = 3,
    mask_saturation_level = 35000,
    bkg_estimator = 'median',
    save =save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig
)
#%%
mbias_path = Preprocessor.get_masterframe_from_image(target_img, 'BIAS')[0]['file']
mdark_path = Preprocessor.get_masterframe_from_image(target_img, 'DARK')[0]['file']
mflat_path = Preprocessor.get_masterframe_from_image(target_img, 'FLAT')[0]['file']
mbias = MasterImage(path = mbias_path, telinfo = target_img.telinfo, status = target_img.status, load = True)
mdark = MasterImage(path = mdark_path, telinfo = target_img.telinfo, status = target_img.status, load = True)
mflat = MasterImage(path = mflat_path, telinfo = target_img.telinfo, status = target_img.status, load = True)
if do_generateerrormap:
    try:
        if errormap_from_propagation:
            if target_bkg is None:
                target_bkg, bkg_instance = BkgGenerator.calculate_sep(
                    target_img = target_img,
                    target_mask = target_srcmask,
                    ** bkg_kwargs)
            target_bkgrms = ErrormapGenerator.calculate_from_bkg(
                target_bkg = target_bkg,
                mbias_img = mbias,
                mdark_img = mdark,
                mflat_img = mflat,
                **errormap_kwargs
            )
            status['errormap'] = True
        if errormap_from_sourcemask:
            target_bkgrms, target_bkg_tmp, bkg_instance = ErrormapGenerator.calculate_from_sourcemask(
                target_img = target_img,
                target_mask = None,
                **errormap_mask_kwargs
            )
            status['errormap'] = True
    except Exception as e:
        status['errormap'] = e

# %% Aperture photometry

do_aperturephotometry = True
aperturephotometry_kwargs = dict(
    sex_params = None,
    detection_sigma = 5,
    aperture_diameter_arcsec = [5,7,10],
    saturation_level = 35000,
    kron_factor = 2.5,
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig,
)

do_photometric_calibration = True
photometriccalibration_kwargs = dict(
    catalog_type = 'GAIAXP',
    max_distance_second = 4,
    calculate_color_terms = True,
    calculate_mag_terms = True,
    mag_lower = None,
    mag_upper = None,
    snr_lower = 15,
    snr_upper = 500,
    classstar_lower = 0.5,
    elongation_upper = 3,
    elongation_sigma = 5,
    fwhm_lower = 1,
    fwhm_upper = 15,
    fwhm_sigma = 5,
    flag_upper = 1,
    maskflag_upper = 1,
    inner_fraction = 0.95, # Fraction of the images
    isolation_radius = 15.0,
    magnitude_key = 'MAG_APER_1',
    flux_key = 'FLUX_APER_1',
    fluxerr_key = 'FLUXERR_APER_1',
    fwhm_key = 'FWHM_IMAGE',
    x_key = 'X_IMAGE',
    y_key = 'Y_IMAGE',
    classstar_key = 'CLASS_STAR',
    elongation_key = 'ELONGATION',
    flag_key = 'FLAGS',
    maskflag_key = 'IMAFLAGS_ISO',
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig,
    save_refcat = True,
)
status = dict()
#%%
target_img = target_imglist_g[0]
target_bkg = Background(target_img.savepath.bkgpath, load = True)
target_bkgrms = Background(target_img.savepath.bkgrmspath, load = True)
#%%
sex_catalog = None
if do_aperturephotometry:
    try:
        # Perform aperture photometry
        sex_catalog = AperturePhotometry.sex_photometry(
            target_img = target_img,
            target_bkg = target_bkg,
            target_bkgrms = target_bkgrms,
            target_mask = None, ################################ HERE, Central stars are masked for ZP caclculation
            **aperturephotometry_kwargs
        )
        status['aperture_photometry'] = True
    except Exception as e:
        status['aperture_photometry'] = e
# %%
sex_catalog.to_region(shape = 'ellipse')
# %%
import numpy as np
import matplotlib.pyplot as plt
catalog_data = sex_catalog.data
# Compute SNR
snr = catalog_data['FLUX_AUTO'] / catalog_data['FLUXERR_AUTO']
snr = snr[np.isfinite(snr) & (snr > 0)]

# Set up bins with log spacing
bins = np.logspace(np.log10(snr.min()), np.log10(snr.max()), 100)

# Plot
plt.figure(figsize=(8, 6))
plt.hist(snr, bins=bins, histtype='stepfilled', color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

# Add a vertical line at SNR=5 for detection threshold reference
plt.axvline(5, color='red', linestyle='--', linewidth=1, label='5 sigma threshold')

# Aesthetics
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Signal-to-Noise Ratio (SNR)', fontsize=13)
plt.ylabel('Number of Sources', fontsize=13)
plt.title('SNR Distribution of Detected Sources', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# %% Photometric Calibration

do_photometric_calibration = True
photometriccalibration_kwargs = dict(
    catalog_type = 'GAIAXP',
    max_distance_second = 1.0,
    calculate_color_terms = True,
    calculate_mag_terms = True,
    mag_lower = None,
    mag_upper = None,
    snr_lower = 20,
    snr_upper = 500,
    classstar_lower = 0.8,
    elongation_upper = 1.7,
    elongation_sigma = 5,
    fwhm_lower = 2,
    fwhm_upper = 15,
    fwhm_sigma = 5,
    flag_upper = 1,
    maskflag_upper = 1,
    inner_fraction = 0.7, # Fraction of the images
    isolation_radius = 10.0,
    magnitude_key = 'MAG_APER_1',
    flux_key = 'FLUX_APER_1',
    fluxerr_key = 'FLUXERR_APER_1',
    fwhm_key = 'FWHM_IMAGE',
    x_key = 'X_IMAGE',
    y_key = 'Y_IMAGE',
    classstar_key = 'CLASS_STAR',
    elongation_key = 'ELONGATION',
    flag_key = 'FLAGS',
    maskflag_key = 'IMAFLAGS_ISO',
    save = True,
    verbose = True,
    visualize = True,
    save_fig = True,
    save_refcat = True,
)
calib_catalog = None
filtered_catalog = None
if do_photometric_calibration and do_aperturephotometry:
    try:
        target_img, calib_catalog, filtered_catalog = PhotometricCalibration.photometric_calibration(
            target_img = target_img,
            target_catalog = sex_catalog,
            **photometriccalibration_kwargs
        )
        status['photometric_calibration'] = True
    except Exception as e:
        status['photometric_calibration'] = e


# %%
all_sources = filtered_catalog.data
#%%
i = 1
ra = all_sources['X_WORLD'][i]
dec = all_sources['Y_WORLD'][i]
mag_aper = all_sources['MAG_APER'][i]
mag_aper_1 = all_sources['MAG_APER_1'][i]
mag_aper_2 = all_sources['MAG_APER_2'][i]
mag_auto = all_sources['MAG_AUTO'][i]
magsky_aper = all_sources['MAGSKY_APER'][i]
magsky_aper_1 = all_sources['MAGSKY_APER_1'][i]
magsky_aper_2 = all_sources['MAGSKY_APER_2'][i]
magsky_auto = all_sources['MAGSKY_AUTO'][i]
magerr_aper = all_sources['MAGERR_APER'][i]
magerr_aper_1 = all_sources['MAGERR_APER_1'][i]
magerr_aper_2 = all_sources['MAGERR_APER_2'][i]
zp_aper = all_sources['ZP_APER'][i]
zp_aper_1 = all_sources['ZP_APER_1'][i]
zp_aper_2 = all_sources['ZP_APER_2'][i]
zp_auto = all_sources['ZP_AUTO'][i]
calib_catalog.show_source(ra, dec)
import matplotlib.pyplot as plt
import numpy as np

# X-axis and labels
x_labels = ['APER[5"]', 'APER_1[7"]', 'APER_2[10"]', 'AUTO']
x = np.arange(len(x_labels))

# Values to plot
mag_values = np.array([mag_aper, mag_aper_1, mag_aper_2, mag_auto])
magsky_values = np.array([magsky_aper, magsky_aper_1, magsky_aper_2, magsky_auto])

# Compute mean and y-limits
mag_mean = np.mean(mag_values)
magsky_mean = np.mean(magsky_values)
mag_ylim = (mag_mean - 0.3, mag_mean + 0.3)
magsky_ylim = (magsky_mean - 0.3, magsky_mean + 0.3)

# Plotting
fig, ax1 = plt.subplots(figsize=(8, 5))

# Left y-axis: calibrated magnitudes
ax1.set_xlabel("Aperture Type")
ax1.set_ylabel("Calibrated Magnitude (MAG)", color='blue')
ax1.scatter(x, mag_values, color='blue', marker='o', label='MAG')
ax1.errorbar(x, mag_values, yerr=magerr_aper, fmt='none', ecolor='blue', elinewidth=1, capsize=3)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(mag_ylim)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)

# Right y-axis: sky magnitudes
ax2 = ax1.twinx()
ax2.set_ylabel("Sky Magnitude (MAGSKY)", color='r')
ax2.scatter(x, magsky_values, color='r', marker='x', label='MAGSKY')
ax2.errorbar(x, magsky_values, yerr=magerr_aper, fmt='none', ecolor='r', elinewidth=1, capsize=3)
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim(magsky_ylim)

plt.title("Comparison of Calibrated and Sky Magnitudes")
fig.tight_layout()
plt.show()


# %%
