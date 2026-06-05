

#%%
import matplotlib.pyplot as plt

#%%
import glob
image_path = glob.glob('/home/hhchoi1022/ezphot/data/scidata/HSC/*.fits')[0]
#%%
from astropy.io import fits
header = fits.getheader(image_path)
# %%
from ezphot.imageobjects import ScienceImage
if __name__ == '__main__':
    telinfo = dict()
    telinfo['telescope'] = 'HSC'
    telinfo['ccd'] = 'TEMP'
    telinfo['binning'] = 1
    telinfo['readoutmode'] = 'NORMAL'
    hdu = fits.open(image_path)
    hdr = hdu[0].header
    hdr['IMGTYPE'] = 'LIGHT'
    hdr['FILTER'] = 'g'
    fits.writeto(image_path, hdu[0].data, hdr, overwrite=True)
    target_img = ScienceImage(image_path, telinfo=telinfo)
    target_img.header['EGAIN'] = 1.0
# %%
ra = 34.53761241994699
dec = -4.973741432375881
#%%
target_srcmask = target_img.calculate_sourcemask(save= True, verbose = True, visualize = True,  mask_radius_factor = 1.5)
# %%
target_bkg = target_img.calculate_bkg(target_srcmask = target_srcmask, save = True, verbose = True, visualize = True, is_2D_bkg = True)
#%%
target_bkgrms = target_img.calculate_bkgrms(target_srcmask = target_srcmask, save = True, verbose = True, visualize = True)
# %%
target_img.show_position(x = ra, y = dec, coord_type = 'coord', radius_arcsec = 2, downsample = 1)
# %%
target_img.header['SEEING'] = 1.5
target_catalog = target_img.photometry_sex(
    target_bkg = target_bkg,
    target_bkgrms = target_bkgrms,
    detection_sigma = 1.5,
    aperture_diameter_arcsec = [2,4,6],
    aperture_diameter_seeing = [3.5, 4.5]
)
#%%
from ezphot.methods import PhotometricCalibration
photcal = PhotometricCalibration()
reference_catalog, _, _ = photcal.select_stars(target_catalog, 
                     mag_lower = None, 
                     mag_upper = None, 
                     snr_lower = 5, 
                     isolation_radius_arcsec = 3,
                     fwhm_lower_arcsec = 0.5,
                     maskflag_upper = 1,
                     inner_fraction = 0.7,
                     verbose = True, visualize = True)
#%%

from pathlib import Path
from astropy.table import Table
ref_catalog_path = '/home/hhchoi1022/ezphot/data/skycatalog/archive/HSC/926853.csv'
ref_catalog_tbl = Table.read(ref_catalog_path, format='ascii.csv')

# %%
from ezphot.helper import Helper
helper = Helper()
# %%
from astropy.coordinates import SkyCoord
idx_observed_ref, idx_catalog_ref, _ =  helper.cross_match(
    SkyCoord(reference_catalog.data['X_WORLD'], reference_catalog.data['Y_WORLD'], unit='deg'),
    SkyCoord(ref_catalog_tbl['ra'], ref_catalog_tbl['dec'], unit='deg'),
    max_distance_second = 0.5
)
idx_observed_all, idx_catalog_all, _ =  helper.cross_match(
    SkyCoord(target_catalog.data['X_WORLD'], target_catalog.data['Y_WORLD'], unit='deg'),
    SkyCoord(ref_catalog_tbl['ra'], ref_catalog_tbl['dec'], unit='deg'),
    max_distance_second = 0.5
)
observed_catalog_ref = reference_catalog.data[idx_observed_ref]
observed_catalog_all = target_catalog.data[idx_observed_all]
ref_catalog_matched = ref_catalog_tbl[idx_catalog_ref]
ref_catalog_matched_all = ref_catalog_tbl[idx_catalog_all]

aper_key = 'MAG_APER_2'
aper_key_ref = 'gmag_aper40'
zp_all = ref_catalog_matched_all[aper_key_ref] - observed_catalog_all[aper_key]
zp_ref = ref_catalog_matched[aper_key_ref] - observed_catalog_ref[aper_key]

from astropy.stats import SigmaClip
sc = SigmaClip(sigma = 3, maxiters = 5)
zp_clip_mask = sc(zp_ref).mask
zp_ref_clipped = zp_ref[~zp_clip_mask]
zp_sources_ref = ref_catalog_matched[aper_key_ref][~zp_clip_mask]
zp_sources_observed = observed_catalog_ref[aper_key][~zp_clip_mask]

import numpy as np
zp_median = np.median(zp_ref_clipped)
zp_std = np.std(zp_ref_clipped)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot magnitude vs ZP
ax.scatter(ref_catalog_matched_all[aper_key_ref] , zp_all, alpha = 0.1, c = 'k')
ax.scatter( zp_sources_ref, zp_ref_clipped, c = 'r')
ax.set_xlabel('MAG_APER_REF')
ax.set_ylabel('ZP_REF')
ax.set_title('MAG_APER_REF vs ZP_REF')
ax.text(0.2, 0.8, f'ZP_MEDIAN = {zp_median:.3f}\nZP_STD = {zp_std:.3f}', transform=ax.transAxes, ha='center', va='center')
ax.set_ylim(zp_median - 1, zp_median + 1)
ax.set_xlim(15, 29)
plt.show()

#%%
target_catalog_forced = target_img.photometry_forced_circular(
    x_arr = [ra],
    y_arr = [dec],
    aperture_diameter_arcsec = [1, 2, 3, 4, 5],
    aperture_diameter_seeing = [],
    annulus_width_arcsec = None,
    unit = 'coord',
    target_bkg = target_bkg,
    target_bkgrms = target_bkgrms,
    target_mask = None,
    visualize = True
)
# %%
aper_key = 'MAG_APER'
for i in range(1, 5):
    aper_key = 'MAG_APER_' + str(i)
    print('APER_%d: %f' % (i, target_catalog_forced.data[aper_key] + 27))
#%%
target_catalog_forced.data['BACKGROUND']
# %%













# %%
ra = 350.41552979654585	
dec = -0.435905859072109
#%%
image_path = glob.glob('/home/hhchoi1022/ezphot/data/scidata/HSC/*.fits')[1]
#%%
hdulist = fits.open(image_path)
#%%
hdulist[0].header


# %%
from ezphot.imageobjects import ScienceImage
if __name__ == '__main__':
    telinfo = dict()
    telinfo['telescope'] = 'HSC'
    telinfo['ccd'] = 'TEMP'
    telinfo['binning'] = 1
    telinfo['readoutmode'] = 'NORMAL'
    from astropy.io import fits
    hdulist = fits.open(image_path)
    header = hdulist[1].header
    data = hdulist[1].data
    header['IMGTYPE'] = 'LIGHT'
    header['FILTER'] = 'g'
    header['EGAIN'] = 1.0
    header['SEEING'] = 1.5
    fits.writeto(image_path, data, header, overwrite=True)
    target_img = ScienceImage(image_path, telinfo=telinfo)
# %%

target_srcmask = target_img.calculate_sourcemask(save= True, verbose = True, visualize = True,  mask_radius_factor = 1)
# %%
target_bkg = target_img.calculate_bkg(target_srcmask = target_srcmask, save = True, verbose = True, visualize = True, is_2D_bkg = True, box_size = 16, filter_size = 5)
#%%
target_bkgrms = target_img.calculate_bkgrms(target_srcmask = target_srcmask, save = True, verbose = True, visualize = True)
# %%
target_img.show_position(x = ra, y = dec, coord_type = 'coord', radius_arcsec = 2, downsample = 1)

#%%
target_catalog = target_img.photometry_sex(
    target_bkg = target_bkg,
    target_bkgrms = target_bkgrms,
    detection_sigma = 0.8,
    aperture_diameter_arcsec = [2,3,4],
    aperture_diameter_seeing = [3.5, 4.5]
)
# %%
from ezphot.methods import PhotometricCalibration
photcal = PhotometricCalibration()
reference_catalog, _, seeing = photcal.select_stars(target_catalog, 
                     mag_lower = None, 
                     mag_upper = None, 
                     snr_lower = 10, 
                     isolation_radius_arcsec = 3,
                     fwhm_lower_arcsec = 0.5,
                     maskflag_upper = 1,
                     inner_fraction = 0.7,
                     verbose = True, visualize = True)
target_img.header['SEEING'] = seeing
# %%

from pathlib import Path
from astropy.table import Table
ref_catalog_path = '/home/hhchoi1022/ezphot/data/skycatalog/archive/HSC/926872.csv'
ref_catalog_tbl = Table.read(ref_catalog_path, format='ascii.csv')

# %%
from ezphot.helper import Helper
helper = Helper()
# %%
from astropy.coordinates import SkyCoord
idx_observed_ref, idx_catalog_ref, _ =  helper.cross_match(
    SkyCoord(reference_catalog.data['X_WORLD'], reference_catalog.data['Y_WORLD'], unit='deg'),
    SkyCoord(ref_catalog_tbl['ra'], ref_catalog_tbl['dec'], unit='deg'),
    max_distance_second = 0.5
)
idx_observed_all, idx_catalog_all, _ =  helper.cross_match(
    SkyCoord(target_catalog.data['X_WORLD'], target_catalog.data['Y_WORLD'], unit='deg'),
    SkyCoord(ref_catalog_tbl['ra'], ref_catalog_tbl['dec'], unit='deg'),
    max_distance_second = 0.5
)
observed_catalog_ref = reference_catalog.data[idx_observed_ref]
observed_catalog_all = target_catalog.data[idx_observed_all]
ref_catalog_matched = ref_catalog_tbl[idx_catalog_ref]
ref_catalog_matched_all = ref_catalog_tbl[idx_catalog_all]

aper_key = 'MAG_APER_2'
aper_key_ref = 'gmag_aper40'
zp_all = ref_catalog_matched_all[aper_key_ref] - observed_catalog_all[aper_key]
zp_ref = ref_catalog_matched[aper_key_ref] - observed_catalog_ref[aper_key]

from astropy.stats import SigmaClip
sc = SigmaClip(sigma = 3, maxiters = 5)
zp_clip_mask = sc(zp_ref).mask
zp_ref_clipped = zp_ref[~zp_clip_mask]
zp_sources_ref = ref_catalog_matched[aper_key_ref][~zp_clip_mask]
zp_sources_observed = observed_catalog_ref[aper_key][~zp_clip_mask]

import numpy as np
zp_median = np.median(zp_ref_clipped)
zp_std = np.std(zp_ref_clipped)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot magnitude vs ZP
ax.scatter(ref_catalog_matched_all[aper_key_ref] , zp_all, alpha = 0.1, c = 'k')
ax.scatter( zp_sources_ref, zp_ref_clipped, c = 'r')
ax.set_xlabel('MAG_APER_REF')
ax.set_ylabel('ZP_REF')
ax.set_title('MAG_APER_REF vs ZP_REF')
ax.text(0.2, 0.8, f'ZP_MEDIAN = {zp_median:.3f}\nZP_STD = {zp_std:.3f}', transform=ax.transAxes, ha='center', va='center')
ax.set_ylim(zp_median - 1, zp_median + 1)
ax.set_xlim(15, 29)
plt.show()
# %%

target_catalog_forced = target_img.photometry_forced_circular(
    x_arr = [ra],
    y_arr = [dec],
    aperture_diameter_arcsec = [0.5, 1, 2, 3],
    aperture_diameter_seeing = [],
    annulus_width_arcsec = 2,
    unit = 'coord',
    target_bkg = target_bkg,
    target_bkgrms = target_bkgrms,
    target_mask = None,
    visualize = True
)
#%%
target_img.show_position(x = ra, y = dec, coord_type = 'coord', radius_arcsec = 1.5, downsample = 1)
#%%

target_catalog.select_sources(
    x = ra,
    y = dec,
    unit = 'coord'
)
target_catalog_forced.select_sources(
    x = ra,
    y = dec,
    unit = 'coord'
)
# %%
aper_key = 'MAG_APER'
for i in range(1, 4):
    aper_key = 'MAG_APER_' + str(i)
    print('APER_%d: %f' % (i, target_catalog_forced.data[aper_key] + 27))
#%%
print(target_catalog.target_data[aper_key] + zp_median)
print(target_catalog_forced.target_data['MAG_APER'] + zp_median)
# %%

# %%
from astropy.coordinates import SkyCoord
_, idx_hsc, _ =  helper.cross_match(
    SkyCoord([ra], [dec], unit='deg'),
    SkyCoord(ref_catalog_tbl['ra'], ref_catalog_tbl['dec'], unit='deg'),
    max_distance_second = 0.5
)
# %%
ref_catalog_tbl[idx_hsc]
# %%
