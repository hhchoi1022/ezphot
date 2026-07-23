


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
path_in_lyman = [path.replace('/lyman/', '/data/') for path in path_all]
# %%
from bridge.alertmonitor import AlertProcessor
alertprocessor = AlertProcessor()
alertprocessor.load_images_from_path(path_in_lyman)  
# %%
alertprocessor.config
#%%
alertprocessor.pipeline_before_stacking(alert_instance)

#%%ssw