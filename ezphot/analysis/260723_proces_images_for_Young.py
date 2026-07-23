
#%%
import pandas as pd
calspec_paths = '/qso/data3/doabdi/calspec_targets_2026_with_spectrum.csv'
calspec_tbl = pd.read_csv(calspec_paths)
calspec_tbl_dec_lt20 = calspec_tbl[calspec_tbl['dec_2026_deg'] < 20]
#%%
from astropy.table import vstack
from bridge.connector import GWPortalConnector
from bridge.objects import Alert
from bridge.alertmonitor import AlertProcessor
#%%
for i, row in calspec_tbl_dec_lt20.iterrows():
    objname = row['star_name']
    target_ra = row['ra_2026_deg']
    target_dec = row['dec_2026_deg']
    alert_instance = Alert(objname = objname, ra=target_ra, dec=target_dec, trigger_time = '2000-01-01')
    
    gwconnector = GWPortalConnector('raw')
    raw_tbl = gwconnector.query(tile_name = alert_instance.tile_id)
    path_all = raw_tbl['filepath']
    path_in_lyman = [path.replace('/lyman/', '/data/') for path in path_all]
    
    alertprocessor = AlertProcessor()

    alertprocessor.load_images_from_path(path_in_lyman)  
    alertprocessor.config.max_worders = 24
    alertprocessor.config.photcal['radius_arcsec'] = None
    alertprocessor.pipeline_before_stacking(alert_instance)
#%%

#%% Visualization
#%%
from ezphot.utils import DataBrowser
dbrowser = DataBrowser('scidata')
dbrowser.objname = alert_instance.tile_id[0]
target_catalogset = dbrowser.search(pattern = '*.fits.cat', return_type = 'catalog')
#%%
from ezphot.dataobjects import PhotometricSpectrum
photspec = PhotometricSpectrum(catalogset = target_catalogset)
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