

#%%
from astropy.coordinates import SkyCoord
from astropy.io import ascii
calspec_info = ascii.read('calspec_info.dat', format = 'fixed_width')
#%%
# is_observed_all = []
# for target_row in calspec_info:
#     tile_id = target_row['tileid']
#     ra_str = target_row['ra']
#     dec_str = target_row['dec']
#     coord = SkyCoord(ra = ra_str, dec = dec_str, unit = ('hourangle', 'deg'))
#     ra = coord.ra.deg
#     dec = coord.dec.deg
#     objname = target_row['star_names']

#     from bridge.connector import GWPortalConnector
#     gwportal_connector = GWPortalConnector()
#     gwportal_connector.query_type = 'raw'
#     query_result = gwportal_connector.query(ra = ra, dec = dec)
#     is_observed = len(query_result) > 0
#     is_observed_all.append(is_observed)
# calspec_info['is_observed'] = is_observed_all
#%%
# calspec_info.write('calspec_info_updated.dat', format = 'ascii.fixed_width', overwrite = True)
calspec_info = ascii.read('calspec_info_updated.dat', format = 'fixed_width')
calspec_info_observed = calspec_info[calspec_info['is_observed'] == 'True']
#%%
check_later_objects = []
#%%
all_num_images = dict()
all_num_paths = dict()
for target_row in calspec_info_observed:
    tile_id = target_row['tileid']
    ra_str = target_row['ra']
    dec_str = target_row['dec']
    coord = SkyCoord(ra = ra_str, dec = dec_str, unit = ('hourangle', 'deg'))
    ra = coord.ra.deg
    dec = coord.dec.deg
    objname = target_row['star_names']

    from bridge.alertmonitor import AlertProcessor
    from bridge.objects import Alert
    from astropy.time import Time
    alert_instance = Alert(objname = objname, ra = ra, dec = dec, trigger_time = Time('2001-01-01'), alert_time = Time('2001-01-01'))
    alertprocessor = AlertProcessor()
    alertprocessor.load_images_db(alert_instance = alert_instance)
    from ezphot.utils import DataBrowser
    dbrowser = DataBrowser('scidata')
    dbrowser.objname = tile_id
    target_pathlist = dbrowser.search(pattern = '*.fits', return_type = 'path')
    dbrowser.objname = objname
    target_pathlist.extend(dbrowser.search(pattern = '*.fits', return_type = 'path'))

    all_num_images[objname] = len(alertprocessor.target_images)
    all_num_paths[objname] = len(target_pathlist)
    try:
        import numpy as np
        from ezphot.imageobjects import ImageSet
        all_exposure_times = np.array([target_img.exptime for target_img in alertprocessor.target_images])
        exposure_set = set(all_exposure_times)
        target_imgset = ImageSet(alertprocessor.target_images)
        for exposure_time in exposure_set:
            target_imgset.select_images(exptime = exposure_time)
            alertprocessor_to_process = AlertProcessor()
            alertprocessor_to_process.target_images = target_imgset.target_images
            alertprocessor_to_process.config.max_workers = 48
            alertprocessor_to_process.config.single_process['do_calculate_bkgrms_from_propagation'] = False
            alertprocessor_to_process.config.single_process['do_forced_photometry'] = True
            exptime_ratio = exposure_time / 100
            magnitude_offset = 2.5*np.log10(np.sqrt(exptime_ratio))
            alertprocessor_to_process.config.photcal['mag_range_default'] = list(np.array(alertprocessor_to_process.config.photcal['mag_range_default']) + magnitude_offset)
            for filter_, val in alertprocessor_to_process.config.photcal['mag_range_dict'].items():
                alertprocessor_to_process.config.photcal['mag_range_dict'][filter_] = list(np.array(val) + magnitude_offset)
            alertprocessor_to_process.pipeline_before_stacking(alert_instance = alert_instance)
    except Exception as e:
        check_later_objects.append(objname)
        print(f"Error processing {objname}: {e}")
# %%
import numpy as np
from bridge.alertmonitor import AlertProcessor
from bridge.objects import Alert
from astropy.time import Time
from ezphot.imageobjects import ImageSet
target_row = calspec_info_observed[1]
tile_id = target_row['tileid']
ra_str = target_row['ra']
dec_str = target_row['dec']
coord = SkyCoord(ra = ra_str, dec = dec_str, unit = ('hourangle', 'deg'))
ra = coord.ra.deg
dec = coord.dec.deg
objname = target_row['star_names']
alert_instance = Alert(objname = objname, ra = ra, dec = dec, trigger_time = Time('2001-01-01'), alert_time = Time('2001-01-01'))
alertprocessor = AlertProcessor()
alertprocessor.load_images_ezphot(alert_instance = alert_instance, file_pattern = '7DT*.fits')
target_imgset = ImageSet(alertprocessor.target_images)
target_imgsetlist = target_imgset.divide_images(by_exptime = True)
#%%
%matplotlib inline
for target_imgset in target_imgsetlist:
    if len(target_imgset.target_images) > 5:
        target_imgset.select_quality_images(
            visualize = True,
        )
    target_imgset.prepare_stack(n_proc = 32, zp_key = 'ZP_APER_3', verbose = False)
    result = target_imgset.stack(n_proc = 8, remove_intermediate = True, verbose = False)
#%%
alertprocessor.stacking()

# %%
alertprocessor.target_images = [result[0]]
alertprocessor.pipeline_after_stacking(alert_instance = alert_instance)
# %%
from ezphot.utils import DataBrowser
dbrowser = DataBrowser('scidata')
dbrowser.objname = tile_id
target_catalogset = dbrowser.search(pattern = '*.fits.cat', return_type = 'catalog')
#%%
for target_catalog in target_catalogset.catalogs:
    fig, ax = target_catalog.target_img.show_position(x = ra, y = dec)
    ax.set_title(target_catalog.target_img.path.name)
#%%
from ezphot.dataobjects import PhotometricSpectrum
photspec = PhotometricSpectrum(catalogset = target_catalogset)
photspec.plt_params.ylim = [15.5, 18]
#%%
fig, _, ax, tbl = photspec.plot(ra = ra, dec = dec, flux_key = 'MAGSKY_AUTO', fluxerr_key = 'MAGERR_AUTO')
fig = list(fig.values())[0]
ax = list(ax.values())[0]
#%%
from astropy.table import Table as AstropyTable
from astropy.utils.data import download_file

calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/c26202_mod_009.fits'
# calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/wd1202_232_stiswfc_002.fits'
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

bin_size = 500
n_bins = len(wl_nm) // bin_size
wl_binned = np.array([np.mean(wl_nm[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])
mag_binned = np.array([np.mean(mag_ab[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])
ax_plot = ax
ax_plot.plot(wl_binned, mag_binned, color='k', alpha=0.5, label='CALSPEC', zorder=0)
for filter_, val in synphot_result.items():
    ax_plot.scatter(val['wl_pivot'].value, val['mag'], facecolor = 'r', edgecolor='k', alpha=0.5, zorder=1, s = 100)
ax_plot.legend()
# %%
fig

# %%
