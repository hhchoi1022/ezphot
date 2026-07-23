

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








#%%
from bridge.connector import GWPortalConnector
# %%
from astropy.table import Table
tbl_target_path = '/home/hhchoi1022/RQ_tile_filter_availability_by_ID.csv'
tbl_target = Table.read(tbl_target_path)
# %%
from bridge.objects import Alert
from bridge.alertmonitor import AlertChecker
from ezphot.utils import DataBrowser
tbl_row = tbl_target[4]
gwconnector_raw = GWPortalConnector('raw')
gwconnector_processed = GWPortalConnector('processed')
gwconnector_combined = GWPortalConnector('combined')
dbrowser = DataBrowser('scidata')
alertchecker = AlertChecker()
helper = Helper()
#%%

failed_row = [0]
failed_reason = [0]
num_rawlist = [0]
num_processedlist = [0]
num_rawcoaddlist = [0]
num_coaddedlist = [0]
num_coadded_hhlist = [0]
all_tiles = ['']
for tbl_row in tbl_target[1:]:
    # try:
    target_ra = tbl_row['RA']
    target_dec = tbl_row['DEC']
    alert_instance = Alert(ra=target_ra, dec=target_dec, objname = tbl_row['ID'], trigger_time = '2000-01-01')
    # from bridge.alertmonitor import AlertProcessor
    # alert_processor = AlertProcessor()
    # alert_processor.load_images_db(alert_instance)
    tile_id = alert_instance.tile_id[0]
    # if tile_id in all_tiles:
    #     continue
    raw_tbl = gwconnector_raw.query(tile_name = alert_instance.tile_id[0])
    if len(raw_tbl) > 0:
        num_rawcoadd = 0
        grouped_raw_tbl = raw_tbl.group_by('night')
        for group in grouped_raw_tbl.groups:
            num_rawcoadd += len(set(group['filter']))
    else:
        num_rawcoadd = 0
    processed_tbl = gwconnector_processed.query(tile_name = alert_instance.tile_id[0])
    if len(processed_tbl) > 0:
        processed_tbl = processed_tbl[processed_tbl['processing_version'] == 'v2']
    coadded_tbl = gwconnector_combined.query(tile_name = alert_instance.tile_id[0])
    if len(coadded_tbl) > 0:
        coadded_tbl = coadded_tbl[coadded_tbl['processing_version'] == 'v2']

    num_coadded = len(coadded_tbl)
    dbrowser.objname = alert_instance.tile_id[0]
    coadded_path_hh = dbrowser.search(pattern = 'coadd_scaled*com.fits')
    num_coadded_hh = len(coadded_path_hh)
    
    all_tiles.append(alert_instance.tile_id[0])
    num_raw = len(raw_tbl)
    num_processed = len(processed_tbl)
    num_rawlist.append(num_raw)
    num_processedlist.append(num_processed)
    num_coaddedlist.append(num_coadded)
    num_coadded_hhlist.append(num_coadded_hh)
    num_rawcoaddlist.append(num_rawcoadd)
        # if len(alert_processor.target_images) == 0:
        #     continue

        # alert_processor.config.single_process['do_forced_photometry'] = False
        # alert_processor.pipeline_before_stacking(alert_instance)
        # single_images = alert_processor.target_images

        # alert_processor.config.stack_prepare
        # alert_processor.config.stack
        # alert_processor.stacking(alert_instance)

        # alert_processor.config.stacked_process['do_DIA'] = False
        # alert_processor.pipeline_after_stacking(alert_instance)

        # from bridge.alertmonitor import AlertChecker
        

        # catalog_set = alertchecker.get_ezphot_photometry(alert_instance, pattern = 'coadd_scaled*com.fits.circ.cat')

        # alertchecker.draw_photometricspectrum(alert_instance, catalog_set, catalog_type = 'ezphot_forced')

        # for single_img in single_images:
        #     single_img.remove(remove_main = True, remove_connected_files = True, verbose = False)
    # except Exception as e:
    #     failed_row.append(tbl_row['ID'])
    #     failed_reason.append(e)
    #     continue
#%%
%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(dpi = 300)
# plt.plot(num_rawlist, label = 'num_raw')
# plt.plot(num_processedlist, label = 'num_processed')
import numpy as np
x = range(len(num_coaddedlist))
plt.scatter(x, num_coaddedlist, label = f'py7DT, {np.sum(num_coaddedlist)} coadded images', s = 15, facecolor = 'red', edgecolor = 'red')
plt.scatter(x, num_coadded_hhlist, label = f'HH, {np.sum(num_coadded_hhlist)} coadded images', s = 15, facecolor = 'blue', edgecolor = 'blue')
plt.scatter(x, num_rawcoaddlist, label = f'Observed, {np.sum(num_rawcoaddlist)} coadded images', s = 35, facecolor = 'none', edgecolor = 'black')

plt.legend()
plt.show()
#%%
tbl_target['num_coadded_pipeline'] = num_coaddedlist
tbl_target['num_coadded_hh'] = num_coadded_hhlist
tbl_target['num_filters'] = num_rawcoaddlist
tbl_target['tile_id'] = all_tiles

# %%
tbl_target
# %%
tbl_row = a[2]
target_ra = tbl_row['RA']
target_dec = tbl_row['DEC']
alert_instance = Alert(ra=target_ra, dec=target_dec, objname = tbl_row['ID'], trigger_time = '2000-01-01')
# from bridge.alertmonitor import AlertProcessor
# alert_processor = AlertProcessor()
# alert_processor.load_images_db(alert_instance)
raw_tbl = gwconnector_raw.query(tile_name = alert_instance.tile_id[0])
processed_tbl = gwconnector_processed.query(tile_name = alert_instance.tile_id[0])
if len(processed_tbl) > 0:
    processed_tbl = processed_tbl[processed_tbl['processing_version'] == 'v2']
coadded_tbl = gwconnector_combined.query(tile_name = alert_instance.tile_id[0])
if len(coadded_tbl) > 0:
    coadded_tbl = coadded_tbl[coadded_tbl['processing_version'] == 'v2']

num_coadded = len(coadded_tbl)
dbrowser.objname = alert_instance.tile_id[0]
coadded_path_hh = dbrowser.search(pattern = 'coadd_scaled*com.fits')
num_coadded_hh = len(coadded_path_hh)
if len(raw_tbl) > 0:
    num_rawcoadd = len(set(raw_tbl['filter']))
else:
    num_rawcoadd = 0

num_raw = len(raw_tbl)
num_processed = len(processed_tbl)
# num_rawlist.append(num_raw)
# num_processedlist.append(num_processed)
# num_coaddedlist.append(num_coadded)
# num_coadded_hhlist.append(num_coadded_hh)
# num_rawcoaddlist.append(num_rawcoadd)
    # if len(alert_processor.target_images) == 0
# %% SEND IMAGES WITH RSYNC

server_address = 'TARS@pnu.ac.kr'
port = 2201
destination_path = '/mnt/dataset/KS4/7DS/'

for tbl_row in tbl_target[1:]:
    target_ra = tbl_row['RA']
    target_dec = tbl_row['DEC']
    alert_instance = Alert(ra=target_ra, dec=target_dec, objname = tbl_row['ID'], trigger_time = '2000-01-01')
    tile_id = alert_instance.tile_id[0]
    dbrowser.objname = tile_id
    coadded_pathlist_hh = dbrowser.search(pattern = 'coadd_scaled*com.fits')
    destination_path = f'{alert_instance.objname}/{coadded_paththlist}'


# %%
from pathlib import Path
import subprocess

account = "hyeonho"
server_address = "TARS.pnu.ac.kr"
port = 2201

remote_host = f"{account}@{server_address}"
remote_base_path = "/mnt/dataset/KS4/7DS"

for tbl_row in tbl_target[37:]:
    target_ra = tbl_row["RA"]
    target_dec = tbl_row["DEC"]

    alert_instance = Alert(
        ra=target_ra,
        dec=target_dec,
        objname=tbl_row["ID"],
        trigger_time="2000-01-01",
    )

    tile_id = alert_instance.tile_id[0]

    dbrowser.objname = tile_id
    coadded_pathlist_hh = dbrowser.search(
        pattern="coadd_scaled*com.fits"
    )

    if len(coadded_pathlist_hh) == 0:
        print(f"No files found: {alert_instance.objname}")
        continue

    object_name = str(alert_instance.objname)
    remote_object_dir = f"{remote_base_path}/{object_name}"

    print(f"Creating remote directory: {remote_host}:{remote_object_dir}")

    subprocess.run(
        [
            "ssh",
            "-p", str(port),
            remote_host,
            "mkdir", "-p", remote_object_dir,
        ],
        check=True,
    )

    for path in coadded_pathlist_hh:
        local_path = Path(path)

        if not local_path.is_file():
            print(f"Local file does not exist: {local_path}")
            continue

        remote_file_path = (
            f"{remote_base_path}/{object_name}/{local_path.name}"
        )

        print(
            f"Sending:\n"
            f"  {local_path}\n"
            f"  -> {remote_host}:{remote_file_path}"
        )

        subprocess.run(
            [
                "scp",
                "-P", str(port),
                str(local_path),
                f"{remote_host}:{remote_file_path}",
            ],
            check=True,
        )
# %%
tbl_target_path_updated = '/home/hhchoi1022/RQ_tile_filter_availability_by_ID_updated.csv'

tbl_target.write(tbl_target_path_updated)

subprocess.run(
    [
        "scp",
        "-P", str(port),
        str(tbl_target_path_updated),
        f"{account}@{server_address}:/mnt/dataset/KS4/7DS/RQ_tile_filter_availability_by_ID_updated.csv",
    ],
    check=True,
)
# %%
tbl_target
# %%

from ezphot.skycatalog import SkyCatalog
all_tiles = set(tbl_target[tbl_target['num_coadded_hh'] > 0]['tile_id'])
tile = list(all_tiles)[0]
for tile in all_tiles:
    skycat = SkyCatalog(objname = tile)
    catalog_path = skycat.filepath
    subprocess.run(
        [
            "scp",
            "-P", str(port),
            str(catalog_path),
            f"{account}@{server_address}:/mnt/dataset/KS4/7DS/{tile}.cat",
        ],
        check=True,
    )
# %%