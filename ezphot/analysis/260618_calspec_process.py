

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
# all_num_images = dict()
# all_num_paths = dict()
# for target_row in calspec_info_observed:
#     tile_id = target_row['tileid']
#     ra_str = target_row['ra']
#     dec_str = target_row['dec']
#     coord = SkyCoord(ra = ra_str, dec = dec_str, unit = ('hourangle', 'deg'))
#     ra = coord.ra.deg
#     dec = coord.dec.deg
#     objname = target_row['star_names']

#     from bridge.alertmonitor import AlertProcessor
#     from bridge.objects import Alert
#     from astropy.time import Time
#     alert_instance = Alert(objname = objname, ra = ra, dec = dec, trigger_time = Time('2001-01-01'), alert_time = Time('2001-01-01'))
#     alertprocessor = AlertProcessor()
#     alertprocessor.load_images_db(alert_instance = alert_instance)
#     from ezphot.utils import DataBrowser
#     dbrowser = DataBrowser('scidata')
#     dbrowser.objname = tile_id
#     target_pathlist = dbrowser.search(pattern = '*.fits', return_type = 'path')
#     dbrowser.objname = objname
#     target_pathlist.extend(dbrowser.search(pattern = '*.fits', return_type = 'path'))

#     all_num_images[objname] = len(alertprocessor.target_images)
#     all_num_paths[objname] = len(target_pathlist)
#     try:
#         import numpy as np
#         from ezphot.imageobjects import ImageSet
#         all_exposure_times = np.array([target_img.exptime for target_img in alertprocessor.target_images])
#         exposure_set = set(all_exposure_times)
#         target_imgset = ImageSet(alertprocessor.target_images)
#         for exposure_time in exposure_set:
#             target_imgset.select_images(exptime = exposure_time)
#             alertprocessor_to_process = AlertProcessor()
#             alertprocessor_to_process.target_images = target_imgset.target_images
#             alertprocessor_to_process.config.max_workers = 48
#             alertprocessor_to_process.config.single_process['do_calculate_bkgrms_from_propagation'] = False
#             alertprocessor_to_process.config.single_process['do_forced_photometry'] = True
#             exptime_ratio = exposure_time / 100
#             magnitude_offset = 2.5*np.log10(np.sqrt(exptime_ratio))
#             alertprocessor_to_process.config.photcal['mag_range_default'] = list(np.array(alertprocessor_to_process.config.photcal['mag_range_default']) + magnitude_offset)
#             for filter_, val in alertprocessor_to_process.config.photcal['mag_range_dict'].items():
#                 alertprocessor_to_process.config.photcal['mag_range_dict'][filter_] = list(np.array(val) + magnitude_offset)
#             alertprocessor_to_process.pipeline_before_stacking(alert_instance = alert_instance)
#     except Exception as e:
#         check_later_objects.append(objname)
#         print(f"Error processing {objname}: {e}")
# %%
for target_row in calspec_info_observed[1:10]:
    tile_id = target_row['tileid']
    ra_str = target_row['ra']
    dec_str = target_row['dec']
    coord = SkyCoord(ra = ra_str, dec = dec_str, unit = ('hourangle', 'deg'))
    ra = coord.ra.deg
    dec = coord.dec.deg
    objname = target_row['star_names']
    alert_instance = Alert(objname = objname, ra = ra, dec = dec, trigger_time = Time('2001-01-01'), alert_time = Time('2001-01-01'))
    alertprocessor = AlertProcessor()
    alertprocessor.load_images_ezphot(alert_instance = alert_instance, file_pattern = 'coadd*com.fits')
    for target_img in alertprocessor.target_images:
        target_img.subtract_background(target_img.bkgmap, visualize = False, save = True)

#%%
import numpy as np
from bridge.alertmonitor import AlertProcessor
from bridge.objects import Alert
from astropy.time import Time
from ezphot.imageobjects import ImageSet
target_row = calspec_info_observed[0]
for target_row in calspec_info_observed[:10]:
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

    # Plot color slope by filter. Color term is "K_COLOR_g-r"
    target_imglist = alertprocessor.target_images

    all_mjds = dict()
    all_color_gr = dict()
    all_telnames = dict()
    for target_img in target_imglist:
        try:
            target_hdr = target_img.header
            filter = target_img.filter
            telname = target_img.telname
            mjd = target_img.mjd
            if filter not in all_color_gr:
                all_color_gr[filter] = []
            if filter not in all_mjds:
                all_mjds[filter] = []
            if filter not in all_telnames:
                all_telnames[filter] = []
            color_gr = target_hdr['K_COLOR_APER_3_g-r']
            all_color_gr[filter].append(color_gr)
            all_mjds[filter].append(mjd)
            all_telnames[filter].append(telname)
            if np.abs(color_gr) > 0.3:
                target_img.remove()
        except:
            continue

    # COLOR FOR FILTER AND SHAPE FOR TELNAME
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from ezphot.dataobjects import LightCurve

    cmap = LightCurve.FILTER_COLOR

    marker_dict = {
        '7DT01': 'o', '7DT02': 's', '7DT03': '^', '7DT04': 'p',
        '7DT05': 'D', '7DT06': 'v', '7DT07': 'h', '7DT08': 'H',
        '7DT09': '*', '7DT10': '<', '7DT11': 'x', '7DT12': '>',
        '7DT13': 'd', '7DT14': '|', '7DT15': '_', '7DT16': ',',
    }

    fig, ax = plt.subplots(dpi=300)

    used_filters = set()
    used_tels = set()

    for filt, color_gr in all_color_gr.items():

        mjds = np.array(all_mjds[filt])
        color_gr = np.array(color_gr)
        telnames = np.array(all_telnames[filt])

        used_filters.add(filt)

        for tel in np.unique(telnames):
            idx = telnames == tel
            used_tels.add(tel)

            ax.scatter(
                mjds[idx],
                color_gr[idx],
                color=cmap[filt],
                marker=marker_dict.get(tel, 'o'),
                alpha=0.8,
            )

    ax.set_xlabel('MJD')
    ax.set_ylabel('Color slope')
    ax.set_title('Color slope by filter')
    ax.set_ylim(-0.3, 0.3)

    # Legend 1: color = filter
    filter_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='none',
            markerfacecolor=cmap[filt],
            markeredgecolor=cmap[filt],
            label=filt,
            markersize=7,
        )
        for filt in sorted(used_filters)
    ]

    # Legend 2: marker = telescope
    tel_handles = [
        Line2D(
            [0], [0],
            marker=marker_dict.get(tel, 'o'),
            linestyle='none',
            color='k',
            label=tel,
            markersize=7,
        )
        for tel in sorted(used_tels)
    ]

    legend1 = ax.legend(
        handles=filter_handles,
        title='Filter color',
        loc='upper left',
        ncols=4,
        fontsize=8,
    )

    ax.add_artist(legend1)

    ax.legend(
        handles=tel_handles,
        title='Telescope marker',
        loc='upper right',
        ncols=2,
        fontsize=8,
    )

    plt.show()

    # alertprocessor.pipeline_before_stacking(alert_instance = alert_instance)

    alertprocessor.stacking(by_filter = True, by_exptime = True)
    # alertprocessor = AlertProcessor()
    # alertprocessor.load_images_ezphot(alert_instance = alert_instance, file_pattern = '*com.fits')

    alertprocessor.pipeline_after_stacking(alert_instance = alert_instance)
#%%
from ezphot.utils import DataBrowser
dbrowser = DataBrowser('scidata')
dbrowser.objname = objname
target_catalogset = dbrowser.search(pattern = '*.com.fits.cat', return_type = 'catalog')
#%%
proper_motion_ra = -813.038
proper_motion_dec = -870.609
yrs = 26
updated_ra = ra + proper_motion_ra * yrs / 3600000
updated_dec = dec + proper_motion_dec * yrs / 3600000
updated_ra = ra
updated_dec = dec
#%%
target_catalogset.catalogs[-8].target_img.show_position(target_catalogset.catalogs[-8].target_data['X_WORLD'][0], target_catalogset.catalogs[-8].target_data['Y_WORLD'][0], zoom_radius_pixel = 100)
#%%
from ezphot.dataobjects import PhotometricSpectrum
photspec = PhotometricSpectrum(catalogset = target_catalogset)
#%%
%matplotlib inline
fig, _, ax, tbl = photspec.plot(ra = updated_ra, dec = updated_dec, flux_key = 'MAGSKY_AUTO', fluxerr_key = 'MAGERR_APER_6', matching_radius_arcsec = 3.0)
fig = list(fig.values())[0]
ax = list(ax.values())[0]
#%%
from astropy.table import Table as AstropyTable
from astropy.utils.data import download_file

calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/gj7541a_mod_001.fits'
calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/wd1202_232_stiswfc_002.fits'
calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/gd71_mod_012.fits'
calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/hs2027_stis_006.fits'
calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/hz4_mod_001.fits'
calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/sf1615_001a_mod_006.fits'
calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/vb8_stiswfcnic_004.fits'
calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/wd0227_050_mod_001.fits'
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

bin_size = 5
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
