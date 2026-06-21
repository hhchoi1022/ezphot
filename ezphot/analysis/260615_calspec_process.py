

#%%
from astropy.io import ascii
calspec_info = ascii.read('calspec_info.dat', format = 'fixed_width')
#%%
calspec_info_observed = calspec_info[calspec_info['is_observed'] == 'True']
target_row = calspec_info_observed[4]
tile_id = target_row['tileid']
ra_str = target_row['ra']
dec_str = target_row['dec']
from astropy.coordinates import SkyCoord
coord = SkyCoord(ra = ra_str, dec = dec_str, unit = ('hourangle', 'deg'))
ra = coord.ra.deg
dec = coord.dec.deg
objname = target_row['star_names']
print(tile_id)
#%%
from bridge.alertmonitor import AlertProcessor
from bridge.objects import Alert
from ezphot.imageobjects import ScienceImage
#%%
from astropy.time import Time
alert_instance = Alert(objname = objname, ra = ra, dec = dec, trigger_time = Time('2001-01-01'), alert_time = Time('2001-01-01'))
# %%
alertprocessor = AlertProcessor()
alertprocessor.load_images_db(alert_instance = alert_instance)
# %%
alertprocessor.pipeline_before_stacking(alert_instance = alert_instance)
# %%
alertprocessor.stacking()
# %%
alertprocessor.pipeline_after_stacking(alert_instance = alert_instance)
# %%
# %%
from ezphot.utils import DataBrowser
dbrowser = DataBrowser('scidata')
dbrowser.objname = tile_id
# dbrowser.objname = objname
target_imgset = dbrowser.search(pattern = '*.com.fits', return_type = 'science')
# target_imgset.select_images(filter = 'i')
#%%
target_imglist = target_imgset.target_images
#%%
# for target_img in target_imglist:
#     target_img.platesolve()
#%%
#%%
for target_img in target_imglist:
    target_ivpmask = target_img.calculate_invalidmask(save= True, verbose = True, visualize = False)
    target_srcmask = target_img.calculate_sourcemask(target_srcmask = None, save= True, verbose = True, visualize = False)
    target_bkg = target_img.calculate_bkg(target_srcmask = target_srcmask, target_ivpmask = target_ivpmask, save = True, verbose = True, visualize = False, is_2D_bkg = True)
    target_bkgrms = target_img.calculate_bkgrms(target_srcmask = target_srcmask, target_ivpmask = target_ivpmask, save = True, verbose = True, visualize = False)
    
    target_catalog = target_img.photometry_sex(
        target_bkg=target_bkg,
        target_bkgrms=target_bkgrms,
        detection_sigma=1.5,
        aperture_diameter_arcsec=[3, 5, 7, 10],
        aperture_diameter_seeing=[3.5, 4.5],
        visualize=False,
        save_fig=False,
        save=True,
        verbose=False
    )

    result = target_img.photometric_calibration(
        target_catalog=target_catalog,
        mag_lower=12.5,
        mag_upper=15.5,
        classstar_lower=0.5,
        elongation_upper=1.7,
        elongation_sigma=5,
        fwhm_lower_arcsec=1,
        fwhm_upper_arcsec=15,
        fwhm_sigma=5,
        flag_upper=1,
        maskflag_upper=1,
        ra_deg=ra,
        dec_deg=dec,
        radius_arcsec=1500,
        inner_fraction=1,
        isolation_radius_arcsec=10.0,
        magnitude_key='MAG_AUTO',
        magnitudeerr_key='MAGERR_AUTO',
        fwhm_key='FWHM_WORLD',
        ra_key='X_WORLD',
        dec_key='Y_WORLD',
        classstar_key='CLASS_STAR',
        elongation_key='ELONGATION',
        flag_key='FLAGS',
        maskflag_key='IMAFLAGS_ISO',
        verbose=False,
        visualize=True,
        save=True,
        save_fig=True,
        save_refcat=True
    )
    
        
#%%
#%%
#%%
target_catalogset = dbrowser.search(pattern = '*com.fits.cat', return_type = 'catalog')
filters_list = [catalog.info.filter for catalog in target_catalogset.catalogs]
filters_list_sorted = sorted(filters_list)
target_cataloglist_sorted = sorted(target_catalogset.catalogs, key=lambda x: filters_list_sorted.index(x.info.filter))
target_catalogset.select_catalogs(filter = 'g')
target_catalog_g = target_catalogset.target_catalogs[0]
#%%
target_catalog_g.select_sources(x = ra, y = dec, matching_radius = 10)
target_catalog_g.target_data['X_WORLD', 'Y_WORLD']
#%%
coord_original = SkyCoord(ra = ra, dec = dec, unit = 'deg')
print(coord_original.ra.hms)
print(coord_original.dec.dms)
target_catalog_g.target_img.show_position(ra, dec, zoom_radius_pixel = 100)
#%%
coord_updated = SkyCoord(ra = target_catalog_g.target_data['X_WORLD'][0], dec = target_catalog_g.target_data['Y_WORLD'][0], unit = 'deg')
updated_ra = coord_updated.ra.deg
updated_dec = coord_updated.dec.deg
print(coord_updated.ra.hms)
print(coord_updated.dec.dms)
target_catalog_g.target_img.show_position(updated_ra, updated_dec, zoom_radius_pixel = 100)
#%%

from ezphot.dataobjects import CatalogSet, PhotometricSpectrum
catset = CatalogSet(catalogs = target_cataloglist_sorted)
photspec = PhotometricSpectrum(catalogset = catset)
photspec.extract_source_info(
    ra = updated_ra,
    dec = updated_dec,
    matching_radius_arcsec = 5.0,
    flux_key = ['MAGSKY_AUTO', 'MAGSKY_APER', 'MAGSKY_APER_1', 'MAGSKY_APER_2', 'MAGSKY_APER_3', 'MAGSKY_APER_4', 'MAGSKY_APER_5', 'MAGSKY_APER_6', 'MAGSKY_APER_7', 'MAGSKY_APER_8'],
    fluxerr_key = ['MAGERR_AUTO', 'MAGERR_APER', 'MAGERR_APER_1', 'MAGERR_APER_2', 'MAGERR_APER_3', 'MAGERR_APER_4', 'MAGERR_APER_5', 'MAGERR_APER_6', 'MAGERR_APER_7', 'MAGERR_APER_8'],
    zperr_key = ['ZPERR_AUTO', 'ZPERR_APER', 'ZPERR_APER_1', 'ZPERR_APER_2', 'ZPERR_APER_3', 'ZPERR_APER_4', 'ZPERR_APER_5', 'ZPERR_APER_6', 'ZPERR_APER_7', 'ZPERR_APER_8'],
    depth_key = ['UL5SKY_AUTO', 'UL5SKY_APER', 'UL5SKY_APER_1', 'UL5SKY_APER_2', 'UL5SKY_APER_3', 'UL5SKY_APER_4', 'UL5SKY_APER_5', 'UL5SKY_APER_6', 'UL5SKY_APER_7', 'UL5SKY_APER_8']
)
# %%
%matplotlib inline
fig, _, ax, tbl = photspec.plot(ra = updated_ra, dec = updated_dec, matching_radius_arcsec = 3.0, flux_key = 'MAGSKY_AUTO')
fig = list(fig.values())[0]
ax = list(ax.values())[0]

# %%
from astropy.table import Table as AstropyTable
from astropy.utils.data import download_file

calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/c26202_mod_009.fits'
calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/c26202_mod_009.fits'
calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/wd1202_232_stiswfc_002.fits'
calspec_fits = download_file(calspec_url, cache=True)
calspec_tbl = AstropyTable.read(calspec_fits, format='fits')
#%%
from ezphot.dataobjects import Spectrum     
import numpy as np
wl_mask = (calspec_tbl['WAVELENGTH'] > 3000) & (calspec_tbl['WAVELENGTH'] < 11000)
calspec_spec = Spectrum(
    wavelength=np.array(calspec_tbl['WAVELENGTH'][wl_mask]),
    flux=np.array(calspec_tbl['FLUX'][wl_mask]),
    wavelength_unit='AA',
    flux_unit='flamb'
)
synphot_result = calspec_spec.synphot(filterset = ['g', 'r', 'm400', 'm425', 'm450', 'm475', 'm500', 'm525', 'm550', 'm575', 'm600', 'm625', 'm650', 'm675', 'm700', 'm725', 'm750', 'm775', 'm800', 'm825', 'm850', 'm875'])[0]

calspec_ab = calspec_spec.ab
wl_nm = calspec_ab.wavelength.value / 10
mag_ab = calspec_ab.flux.value

bin_size = 1
n_bins = len(wl_nm) // bin_size
wl_binned = np.array([np.mean(wl_nm[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])
mag_binned = np.array([np.mean(mag_ab[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])
ax_plot = ax
ax_plot.plot(wl_binned, mag_binned, color='k', alpha=0.5, label='CALSPEC', zorder=0)
for filter_, val in synphot_result.items():
    ax_plot.scatter(val['wl_pivot'].value, val['mag'], facecolor = 'r', edgecolor='k', alpha=0.5, zorder=1, s = 50)
ax_plot.legend()
# %%
fig
# %%
catset.select_catalogs(filter = 'g')
target_catalog_g = catset.target_catalogs[0]
catset.select_catalogs(filter = 'r')
target_catalog_r = catset.target_catalogs[0]
color_gr_dict = dict()
mag_keys = [c for c in target_catalog_g.data.colnames if c.startswith('MAGSKY_')]
for mag_key in mag_keys:
    color_gr_dict[mag_key] = target_catalog_g.target_data[mag_key][0] - target_catalog_r.target_data[mag_key][0]
#%%
# Apply color terms
mag_key = 'MAGSKY_APER_4'
all_magnitudes = []
magnitudes = []
filters = []
for target_catalog_forced in all_target_catalog_forced:
    target_img = target_catalog_forced.target_img
    target_hdr = target_img.header
    mag_keys = [c for c in target_catalog_forced.data.colnames if c.startswith('MAGSKY_')]
    # for mag_key in mag_keys:
    color_gr = color_gr_dict[mag_key]
    c_term_intercept_key = mag_key.replace('MAGSKY_', 'C_COLOR_') + '_g-r'
    c_term_slope_key = mag_key.replace('MAGSKY_', 'K_COLOR_') + '_g-r'
    c_term_intercept_value = target_hdr[c_term_intercept_key]
    c_term_slope_value = target_hdr[c_term_slope_key]
    c_term = c_term_intercept_value + c_term_slope_value * color_gr
    print(target_img.filter, c_term)
    magnitude = target_catalog_forced.target_data[mag_key][0]
    corrected_magnitude = target_catalog_forced.target_data[mag_key][0] + c_term
    filters.append(target_img.filter)
    magnitudes.append(magnitude)
    all_magnitudes.append(corrected_magnitude)
# %%
for filter, magnitude, all_magnitude in zip(filters, magnitudes, all_magnitudes):
    print(filter, magnitude, all_magnitude)
# %%
for filter, magnitude, all_magnitude in zip(filters, magnitudes, all_magnitudes):
    wl = synphot_result[filter]['wl_pivot'].value
    mag = all_magnitude
    ax_plot.scatter(wl, mag, color = 'r', marker = 'D', s = 50)    
    print(filter, wl, mag)
# %%
fig
# %%
