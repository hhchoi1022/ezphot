
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
row = calspec_tbl_dec_lt20.iloc[2]

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
target_catalogset = dbrowser.search(pattern = '*com.fits.cat', return_type = 'catalog')
from ezphot.dataobjects import PhotometricSpectrum
photspec = PhotometricSpectrum(catalogset = target_catalogset)
#%%
target_ra = row['ra_j2000_deg'] + 26 * row['pmra_masyr'] / 3600 / 1000
target_dec = row['dec_j2000_deg'] + 26 * row['pmdec_masyr'] / 3600 / 1000
#%%

#%%
target_ra = row['ra_j2000_deg']
target_dec = row['dec_j2000_deg']
#%%
target_imgset.target_images[0].show_position(x = target_ra, y = target_dec)
#%%
%matplotlib inline
fig, _, ax, tbl = photspec.plot(ra = target_ra, dec = target_dec, flux_key = 'MAGSKY_AUTO', fluxerr_key = 'MAGERR_APER_6', matching_radius_arcsec = 3.0)
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



objname