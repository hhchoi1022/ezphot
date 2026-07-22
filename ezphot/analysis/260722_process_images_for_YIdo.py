


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
alertprocessor.pipeline_before_stacking()

#%%

target_img = alertprocessor.target_images[0]
# %%
target_img.show('pixel')

#%%
corrected_img = target_img.correct_bdf(save = True)

# %%
corrected_img.show('pixel')
# %%
plate_solved_img = corrected_img.platesolve()
#%%
plate_solved_img.show('coord')
# %%
target_srcmask = plate_solved_img.calculate_sourcemask()
# %%
target_bkgmap = plate_solved_img.calculate_bkg(target_srcmask)
# %%
target_bkgrms = plate_solved_img.calculate_bkgrms_from_propagation(target_bkgmap)
# %%
target_catalog = plate_solved_img.photometry_sex(target_bkgmap, target_bkgrms, save = False)
#%%
target_catalog.show_source(target_ra, target_dec)
# %%
plate_solved_img.photometric_calibration(target_catalog)