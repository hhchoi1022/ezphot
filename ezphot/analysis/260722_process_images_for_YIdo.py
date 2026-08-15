#%%
target_ra = 323.94508749999994
target_dec = -63.90354722222222
objname = 'AT2026jqi'
from bridge.objects import Alert
alert_instance = Alert(objname = objname, ra=target_ra, dec=target_dec, trigger_time = '2000-01-01')
#%%
from bridge.connector import GWPortalConnector
gwconnector = GWPortalConnector('raw')
raw_tbl = gwconnector.query(tile_name = alert_instance.tile_id)
path_all = raw_tbl['filepath']
# %%
from bridge.alertmonitor import AlertProcessor
alertprocessor = AlertProcessor()
alertprocessor.load_images_from_path(path_all)  
# %%

from ezphot.utils import DataBrowser
dbrowser = DataBrowser('scidata')
# %%
dbrowser.objname = 'T02080'
# %%
target_imgset = dbrowser.search(pattern = '*.fits', return_type = 'science')
# %%
target_imglist = target_imgset.target_images
# %%
alertprocessor = AlertProcessor()
alertprocessor.target_images = target_imglist
#%%
alertprocessor.stacking()

#%%
alertprocessor.config.stacked_process['do_DIA'] = False
#%%
dbrowser = DataBrowser('scidata')
dbrowser.objname = 'T02080'
target_imgset = dbrowser.search(pattern = '*com.fits', return_type = 'science')
target_imglist = target_imgset.target_images
#%%
alertprocessor = AlertProcessor()
alertprocessor.config.stacked_process['do_DIA'] = False
alertprocessor.target_images = target_imglist
alertprocessor.pipeline_after_stacking(alert_instance)
#%%
from ezphot.dataobjects import PhotometricSpectrum
# %%
dbrowser = DataBrowser('scidata')
dbrowser.objname = 'T02080'
target_catalogset = dbrowser.search(pattern = '*20260419*com.fits.circ.cat', return_type = 'catalog')
photspec = PhotometricSpectrum(target_catalogset)
# %%
%matplotlib inline
result = photspec.plot(ra = alert_instance.ra, dec = alert_instance.dec, overplot_stamp = True, flux_key = 'MAGSKY_APER_1', fluxerr_key = 'MAGERR_APER_1', zperr_key = 'ZPERR_APER_1', depth_key = 'UL5SKY_APER_1')
# %%

result[2]['2026-04-19 07:03'].set_title('')
# %%
target_imgset.target_images[30].header['NOTE']
# %%
