
#%%
import pandas as pd
from astropy.table import Table
from astropy.table import hstack
calspecmag_paths = '/qso/data3/doabdi/calspec_tileVg_match.csv'
calspecmag_tbl = pd.read_csv(calspecmag_paths)
calspecmag_tbl = Table().from_pandas(calspecmag_tbl)
calspec_paths = '/qso/data3/doabdi/calspec_targets_2026_with_spectrum.csv'
calspec_tbl = pd.read_csv(calspec_paths)
calspec_tbl = Table().from_pandas(calspec_tbl)
#%%
calspec_tbl = hstack([calspec_tbl, calspecmag_tbl])
calspec_tbl_dec_lt20 = calspec_tbl[(calspec_tbl['dec_2026_deg'] < 20) & (calspec_tbl['catalog_mag'] > 11) & (calspec_tbl['catalog_mag'] < 17)]
calspec_tbl_dec_lt20 = calspec_tbl_dec_lt20.to_pandas()
#%%
from astropy.table import vstack
from bridge.connector import GWPortalConnector
from bridge.objects import Alert
from bridge.alertmonitor import AlertProcessor
#%%
calspec_tbl_dec_lt20 = pd.read_csv('./calspec_info.csv')
row = calspec_tbl_dec_lt20.iloc[22]

#%%

from ezphot.imageobjects import ImageSet
from ezphot.imageobjects import ScienceImage
import numpy as np
#%%
import gc
# Start from 34th row
for i, row in calspec_tbl_dec_lt20.iloc[35:].iterrows():

    objname = row['star_name_1']
    target_ra = row['ra_2026_deg']
    target_dec = row['dec_2026_deg']
    print(f'Processing {objname} {target_ra} {target_dec}')
    alert_instance = Alert(objname = objname, ra=target_ra, dec=target_dec, trigger_time = '2000-01-01')
    
    gwconnector = GWPortalConnector('raw')
    # raw_tbl_tile = gwconnector.query(tile_name = alert_instance.tile_id[0])
    raw_tbl_objname = gwconnector.query(object_name = objname)
    if len(raw_tbl_objname) == 0:
        continue
    path_all = raw_tbl_objname['filepath']
    path_in_lyman = [path.replace('/lyman/', '/data/') for path in path_all]
    target_imglist = [ScienceImage(path) for path in path_in_lyman]
    target_imgset = ImageSet(target_imglist)
    target_imgsetlist = target_imgset.divide_images(by_filter = False, by_exptime = True)

    for imgset in target_imgsetlist: 
        alertprocessor = AlertProcessor()
        alertprocessor.target_images = imgset.target_images

        # alertprocessor.load_images_from_path(path_in_lyman) 
        # Remove space if exists in the path
        target_imglist = alertprocessor.target_images
        all_paths = []
        for img in target_imglist:
            img.filename = str(img.filename).replace(' ', '_')
            img.savedir = str(img.savedir).replace(' ', '_')
            objname = img.header['OBJECT']
            img.header['OBJECT'] = objname.replace(' ', '_')
            img.write()
            img.clear()
            all_paths.append(img.path)
        del target_imglist

        gc.collect()

        alertprocessor = AlertProcessor()
        alertprocessor.load_images_from_path(all_paths)

        # Change magnitude range for 10s exposure time 
        exptime = imgset.target_images[0].exptime
        factor = 100 / exptime
        mag_diff = 2.5 * np.log10(factor)
        alertprocessor.config.photcal['mag_range_default'][0] -= mag_diff # 1/10 of the original magnitude range
        alertprocessor.config.photcal['mag_range_default'][1] -= mag_diff # 1/10 of the original magnitude range
        for filt, val in alertprocessor.config.photcal['mag_range_dict'].items():
            mag_range = val
            mag_range[0] -= mag_diff # 1/10 of the original magnitude range
            mag_range[1] -= mag_diff # 1/10 of the original magnitude range
            alertprocessor.config.photcal['mag_range_dict'][filt] = mag_range

        alertprocessor.config.max_worders = 24
        alertprocessor.config.photcal['radius_arcsec'] = None
        alertprocessor.pipeline_before_stacking(alert_instance)

        alertprocessor.config.max_workers = 2
        alertprocessor.config.batch_size = 2
        alertprocessor.config.stack_prepare['n_proc'] = 12
        alertprocessor.config.stack['n_proc'] = 12
        alertprocessor.config.stack_select['enabled'] = True
        alertprocessor.config.stack_select['depth_limit'] = 14
        alertprocessor.stacking()


#%%

#%% Visualization
#%%
for i, row in calspec_tbl_dec_lt20.iloc[25:].iterrows():
    objname = row['star_name_1']
    objname = objname.replace(' ', '_')
    target_ra = row['ra_2026_deg']
    target_dec = row['dec_2026_deg']
    alert_instance = Alert(objname = objname, ra=target_ra, dec=target_dec, trigger_time = '2000-01-01')

    from ezphot.utils import DataBrowser
    dbrowser = DataBrowser('scidata')
    # dbrowser.objname = alert_instance.tile_id[0]
    dbrowser.objname = alert_instance.objname
    target_imgset = dbrowser.search(pattern = '*com.fits', return_type = 'science')
    target_imglist = target_imgset.target_images
    if len(target_imglist) == 0:
        continue
    target_imgset.exclude_images(filter = 'u')
    alertprocessor = AlertProcessor()
    alertprocessor.config.stacked_process['do_DIA'] = False
    alertprocessor.target_images = target_imglist
    alertprocessor.pipeline_after_stacking(alert_instance)
    for target_img in target_imglist:
        target_img.clear()
#%%

objname = row['star_name_1']
target_ra = row['ra_2026_deg']
target_dec = row['dec_2026_deg']
alert_instance = Alert(objname = objname, ra=target_ra, dec=target_dec, trigger_time = '2000-01-01')


dbrowser = DataBrowser('scidata')
dbrowser.objname = alert_instance.tile_id[0]
target_catalogset = dbrowser.search(pattern = '*com.fits.cat', return_type = 'catalog')
from ezphot.dataobjects import PhotometricSpectrum
photspec = PhotometricSpectrum(catalogset = target_catalogset)

#%%
%matplotlib inline
fig, _, ax, tbl = photspec.plot(ra = target_ra, dec = target_dec, flux_key = 'MAGSKY_APER_1', fluxerr_key = 'MAGERR_APER_2', matching_radius_arcsec = 3.0)
fig = list(fig.values())[0]
ax = list(ax.values())[0]
#%%
from ezphot.dataobjects import Spectrum     

#%%
from pathlib import Path
spectrum_filepath = row['spectrum_filepath']
spectrum_filepath = '/qso/' + spectrum_filepath
#%%
import numpy as np
from astropy.io import ascii
from astropy.table import Table
calspec_tbl = Table.read(spectrum_filepath)
#%%
import numpy as np
wl_mask = (calspec_tbl['WAVELENGTH'] > 3000) & (calspec_tbl['WAVELENGTH'] < 11000)
calspec_spec = Spectrum(
    wavelength=np.array(calspec_tbl['WAVELENGTH'][wl_mask]),
    flux=np.array(calspec_tbl['FLUX'][wl_mask]),
    wavelength_unit='AA',
    flux_unit='flamb'
)
synphot_result = calspec_spec.synphot(filterset = ['g', 'r', 'medium'])[0]# 'm400', 'm425', 'm450', 'm475', 'm500', 'm525', 'm550', 'm575', 'm600', 'm625', 'm650', 'm675', 'm700', 'm725', 'm750', 'm775', 'm800', 'm825', 'm850', 'm875'])[0]

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
import numpy as np
import matplotlib.pyplot as plt

old_filters = [
    'g', 'r',
    'm400', 'm425', 'm450', 'm475', 'm500', 'm525',
    'm550', 'm575', 'm600', 'm625', 'm650', 'm675',
    'm700', 'm725', 'm750', 'm775', 'm800', 'm825',
    'm850', 'm875'
]

mag_key = 'MAGSKY_APER'
magerr_key = 'MAGERR_APER'
zperr_key = 'ZPERR_APER'

fig, ax = plt.subplots(figsize=(16, 7))

old_label_used = False
new_label_used = False

for filt, val in synphot_result.items():

    if filt not in tbl['filter']:
        continue

    tbl_filt = tbl[tbl['filter'] == filt]

    # observed magnitude
    mag = tbl_filt[mag_key][0]

    # photometric error
    mag_err = tbl_filt[magerr_key][0]

    # zero-point error
    zp_err = tbl_filt[zperr_key][0]   # <- 실제 저장된 key 이름에 맞게 수정

    # total error
    total_err = np.sqrt(
        mag_err**2 + zp_err**2
    )

    # residual
    mag_diff = mag - val['mag']

    # central wavelength
    cent_wl = val['wl_pivot'].value

    # color / legend
    if filt in old_filters:
        color = 'tab:blue'

        if not old_label_used:
            plot_label = 'Existing medium-band'
            old_label_used = True
        else:
            plot_label = None

    else:
        color = 'tab:orange'

        if not new_label_used:
            plot_label = 'New medium-band'
            new_label_used = True
        else:
            plot_label = None

    # point + total error
    ax.errorbar(
        cent_wl,
        mag_diff,
        yerr=total_err,
        fmt='o',
        color=color,
        markersize=10,
        capsize=4,
        elinewidth=2,
        label=plot_label,
        zorder=3
    )

    # filter label
    ax.text(
        cent_wl,
        mag_diff + 0.012,
        filt,
        color='k',
        fontsize=10,
        ha='center'
    )

# zero residual line
ax.axhline(
    0,
    color='gray',
    linestyle='--',
    linewidth=1.5
)

ax.set_xlabel(
    'Transmission-weighted central wavelength [nm]',
    fontsize=14
)

ax.set_ylabel(
    r'$m_{\mathrm{TD,cal}} - m_{\mathrm{CALSPEC,syn}}$ [mag]',
    fontsize=14
)

ax.set_title(
    'SF1615+001A | MAG_AUTO | CALSPEC residual',
    fontsize=16
)

ax.set_xlim(380, 870)
ax.set_ylim(-0.175, 0.44)

ax.set_xticks([400, 500, 600, 700, 800])

ax.grid(True, alpha=0.3)

ax.legend(
    loc='upper left',
    fontsize=12
)

plt.tight_layout()
plt.show()
# %%
