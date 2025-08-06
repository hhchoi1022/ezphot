
# Written on 2025-06-30


#%%
from tippy.imageojbects import *
from tippy.methods import TIPSubtraction, TIPStacking
from tippy.utils import *
from tippy.helper import Helper
import numpy as np
tile_ids_S250206dm = list([
    "T01259", "T01159", "T01467", "T01361", "T01160", "T01360", "T01258", "T01362",
    "T01256", "T01363", "T01161", "T01257", "T01469", "T01260", "T01468", "T01158",
    "T01465", "T01576", "T01064", "T01358", "T01359", "T01364", "T01065", "T01261",
    "T01466", "T01162", "T01066", "T01463", "T01577", "T00974", "T01063", "T01578",
    "T01067", "T01262", "T01470", "T01691", "T01255", "T01357", "T00975", "T01575",
    "T00973", "T01163", "T01157", "T01579", "T01690", "T01692", "T01156", "T01464",
    "T01689", "T00972", "T01693", "T01580", "T01462", "T01574", "T01068", "T00976",
    "T01062", "T01573", "T01254", "T01365", "T01694", "T01572", "T01355", "T01164",
    "T01471", "T01809", "T01356", "T01810", "T01263", "T01811", "T01061", "T01688",
    "T01687", "T01461", "T01155"
    ])

tile_ids_S250206dm = list([
 "T01257", "T01469", "T01260", "T01468",
    "T01465", "T01576", "T01064", "T01358", "T01359", "T01364", "T01065", "T01261",
    "T01466", "T01162", "T01066", "T01463", "T01577", "T00974", "T01063", "T01578",
    "T01067", "T01262", "T01470", "T01691", "T01255", "T01357", "T00975", "T01575",
    "T00973", "T01163", "T01157", "T01579", "T01690", "T01692", "T01156", "T01464",
    "T01689", "T00972", "T01693", "T01580", "T01462", "T01574", "T01068", "T00976",
    "T01062", "T01573", "T01254", "T01365", "T01694", "T01572", "T01355", "T01164",
    "T01471", "T01809", "T01356", "T01810", "T01263", "T01811", "T01061", "T01688",
    "T01687", "T01461", "T01155"
    ])
#%%
helper = Helper()
sdt_data_querier = SDTData()
# %%
stacking = TIPStacking()
DIA = TIPSubtraction()

tile_id = 'T01158'
for tile_id in tile_ids_S250206dm:
    imginfo_all = sdt_data_querier.show_scidestdata(
        targetname = tile_id,
        show_only_numbers = False,
        key = 'filter',
        pattern = 'calib*100.com.fits'
    )
    for filter_ in ['g', 'r']:
        if filter_ not in imginfo_all.keys():
            print(f'No {filter_} filter images found for tile {tile_id}. Skipping...')
            continue
        imginfo = imginfo_all[filter_]
        telinfo = helper.estimate_telinfo(imginfo[0])
        target_imglist = [ScienceImage(img, telinfo = telinfo, load = True) for img in imginfo]
        target_img = stacking.select_quality_images(target_imglist, max_numbers = 3)[0]
        reference_img, _ = DIA.get_referenceframe_from_image(target_img)
    
        for target_img in target_imglist:
            result = DIA.find_transients(target_img, 
                                        [reference_img],
                                        visualize = True,
                                        show_transient_numbers = 1,
                                        reject_variable_sources = True
                                        )
#%%
all_catalog = result[0][0]
candidate_catalog = result[1][0]
transient_catalog = result[2][0]
subframe_target_img = result[3][0]
subframe_reference_img = result[4][0]
subframe_subtracted_img = result[5][0]
transient_tbl = transient_catalog.data
candidate_tbl = candidate_catalog.data
#%%
import time
for i in range(len(transient_tbl)):
    ra = transient_tbl['X_WORLD'][i]
    dec = transient_tbl['Y_WORLD'][i]
    DIA.show_transient_positions(
        subframe_target_img, 
        subframe_reference_img,
        subframe_subtracted_img,
        ra,
        dec,
        title = f'Transient {i+1} - {transient_tbl["NUMBER"][i]}'
    )
    time.sleep(1)
#%%
from tippy.catalog import TIPCatalog
#candidate_catalog.show_source(241.624307, -70.3271458)
DIA.show_transient_positions(
    subframe_target_img, 
    subframe_reference_img,
    subframe_subtracted_img,
    241.624307,
    -70.3271458
)
#all_catalog = TIPCatalog('/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T01158/7DT15/g/calib_7DT15_T01158_20250210_075804_g_100.com.fits.cat', catalog_type = 'all', load = True)
#candidate_catalog = TIPCatalog('/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T01158/7DT09/r/sub_coadd_calib_7DT09_T01158_20250207_074709_r_100.com_subframe_0.fits.transient', catalog_type = 'transient', load = True)
#%%
C = transient_catalog.search_sources(241.624307, -70.3271458)[0]
#%%
#subframe_subtracted_img = ScienceImage('/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T01158/7DT03/r/sub_coadd_calib_7DT03_T01158_20250208_044740_r_100.com_subframe_0.fits', telinfo = telinfo, load = True)
from tippy.methods import TIPAperturePhotometry
TIPAperturePhotometry = TIPAperturePhotometry()
A = TIPAperturePhotometry.circular_photometry(
    target_img = subframe_subtracted_img,
    x_arr = transient_catalog.data['X_WORLD'],
    y_arr = transient_catalog.data['Y_WORLD'],
    unit = 'coord',
    aperture_diameter_arcsec = [5,7,10],
    target_bkgrms = None,
    save = True,
    visualize = True,
    save_fig = True
)
#%%
from tippy.methods import TIPPhotometricCalibration
TIPPhotometricCalibration = TIPPhotometricCalibration()
B = TIPPhotometricCalibration.apply_zp(
    target_img = subframe_subtracted_img,
    target_catalog = A,
    save = True
)
D = B.search_sources(241.624307, -70.3271458)[0]

#%%
C['MAGERR_APER_2']
#D['MAGERR_APER_2']
# %%
# from tippy.utils import ImageQuerier

tile_ids_S250206dm = list([
    "T01259", "T01159", "T01467", "T01361", "T01160", "T01360", "T01258", "T01362",
    "T01256", "T01363", "T01161", "T01257", "T01469", "T01260", "T01468", "T01158",
    "T01465", "T01576", "T01064", "T01358", "T01359", "T01364", "T01065", "T01261",
    "T01466", "T01162", "T01066", "T01463", "T01577", "T00974", "T01063", "T01578",
    "T01067", "T01262", "T01470", "T01691", "T01255", "T01357", "T00975", "T01575",
    "T00973", "T01163", "T01157", "T01579", "T01690", "T01692", "T01156", "T01464",
    "T01689", "T00972", "T01693", "T01580", "T01462", "T01574", "T01068", "T00976",
    "T01062", "T01573", "T01254", "T01365", "T01694", "T01572", "T01355", "T01164",
    "T01471", "T01809", "T01356", "T01810", "T01263", "T01811", "T01061", "T01688",
    "T01687", "T01461", "T01155"
    ])

for tile_id in tile_ids_S250206dm:
    imginfo_all = sdt_data_querier.show_scidestdata(
        targetname = tile_id,
        show_only_numbers = False,
        key = 'filter',
        pattern = 'calib*com.fits'
    )
    print(f'Tile ID: {tile_id}, Number of images: {len(imginfo_all)}')
    
    if 'g' in imginfo_all.keys():
        imginfo = imginfo_all['g']
        target_img = ScienceImage(imginfo[0], telinfo = telinfo, load = True)
        image_qurier = ImageQuerier(catalog_key = f'SkyMapper/SMSS4/{target_img.filter}') 
        #reference_path = target_img.savedir / f'{target_img.objname}_ref.fits'
        reference_img = image_qurier.query(
            width = int(target_img.naxis1 * 1.2),
            height = int(target_img.naxis2 * 1.2),
            ra = target_img.ra,
            dec = target_img.dec,
            pixelscale = np.mean(target_img.pixelscale),
            telinfo = target_img.telinfo,
            save_path = None,
            objname = target_img.objname,
        )
        reference_img.register()

    if 'r' in imginfo_all.keys():
        imginfo = imginfo_all['r']
        target_img = ScienceImage(imginfo[0], telinfo = telinfo, load = True)
        
        image_qurier = ImageQuerier(catalog_key = f'SkyMapper/SMSS4/{target_img.filter}') 
        #reference_path = target_img.savedir / f'{target_img.objname}_ref.fits'
        reference_img = image_qurier.query(
            width = int(target_img.naxis1 * 1.2),
            height = int(target_img.naxis2 * 1.2),
            ra = target_img.ra,
            dec = target_img.dec,
            pixelscale = np.mean(target_img.pixelscale),
            telinfo = target_img.telinfo,
            save_path = None,
            objname = target_img.objname,
        )
        reference_img.register()

#%%
reference_img = ReferenceImage(reference_path, telinfo = target_img.telinfo, load = True)
#%%
target_img.show()
reference_img.show()
result = DIA.find_transients(target_img, 
                             [reference_img],
                             visualize = True,
                             show_transient_numbers = 30,
                             reject_variable_sources = True
                             )
# %%
from tippy.catalog import TIPCatalog
candidate_catalog = result[0][0]
transient_catalog = result[1][0]
subframe_target_img = result[2][0]
subframe_reference_img = result[3][0]
subframe_subtracted_img = result[4][0]
candidate_tbl = candidate_catalog.data
transient_tbl = transient_catalog.data
#%%
show_tbl = candidate_tbl
for i in range(len(show_tbl)):
    ra = show_tbl['X_WORLD'][i]
    dec = show_tbl['Y_WORLD'][i]

    DIA.show_transient_positions(
        subframe_target_img, 
        subframe_reference_img,
        subframe_subtracted_img,
        ra,
        dec
    )
    
#%%
candidate_catalog.show_source(229.4253039, -67.0045685)

# %%
DIA.show_transient_positions(
    subframe_target_img, 
    subframe_reference_img,
    subframe_subtracted_img,
    229.4253039,
    -67.0045685
)
# %%
imginfo_all = sdt_data_querier.show_scidestdata(
    targetname = tile_id,
    show_only_numbers = False,
    key = 'filter',
    pattern = 'sub*100.com.fits'
)
# %%
