

#%%
from bridge.connector import GWPortalConnector
# %%
from astropy.table import Table
tbl_target_path = '/home/hhchoi1022/RQ_tile_filter_availability_by_ID.csv'
tbl_target = Table.read(tbl_target_path)
# %%
from bridge.objects import Alert
tbl_row = tbl_target[4]
gwconnector = GWPortalConnector('raw')
failed_row = []
failed_reason = []

for tbl_row in tbl_target[5:]:
    try:
        target_ra = tbl_row['RA']
        target_dec = tbl_row['DEC']
        alert_instance = Alert(ra=target_ra, dec=target_dec, objname = tbl_row['ID'], trigger_time = '2000-01-01')
        from bridge.alertmonitor import AlertProcessor
        alert_processor = AlertProcessor()
        alert_processor.load_images_db(alert_instance)
        if len(alert_processor.target_images) == 0:
            continue

        alert_processor.config.single_process['do_forced_photometry'] = False
        alert_processor.pipeline_before_stacking(alert_instance)
        single_images = alert_processor.target_images

        alert_processor.config.stack_prepare
        alert_processor.config.stack
        alert_processor.stacking(alert_instance)

        alert_processor.config.stacked_process['do_DIA'] = False
        alert_processor.pipeline_after_stacking(alert_instance)

        from bridge.alertmonitor import AlertChecker
        alertchecker = AlertChecker()

        catalog_set = alertchecker.get_ezphot_photometry(alert_instance)

        alertchecker.draw_photometricspectrum(alert_instance, catalog_set)

        for single_img in single_images:
            single_img.remove(remove_main = True, remove_connected_files = True, verbose = False)
    except Exception as e:
        failed_row.append(tbl_row['ID'])
        failed_reason.append(e)
        continue
#%%