

#%%
from ezphot.methods import *
from ezphot.imageobjects import *
from ezphot.helper import Helper
from ezphot.utils import DataBrowser
from ezphot.skycatalog import *
import psutil, os
from pympler import asizeof

from tqdm import tqdm
import gc
import time
import numpy as np
#%%
tile_ids = ['T04071',
            'T05354',
            'T08344',
            'T15265',
            'T14988',
            'T16088',
            'T19432',
            'T19150']
#%%
tile_ids = [
    # 'T04071',
    # 'T05354',
    # 'T08344',
    # 'T15265',
    # 'T14988',
    # 'T16088',
    # 'T19432',
    # 'T19150',
    # 'T16205',
    # 'T16484',
    # 'T20116',
    # 'T20675',
    'T22064',
    'T20118',
    'T20397',
    'T22066']
    
tile_ids_too = [
    # 'T03751',
    # 'T03573',
    # 'T02955',
    # 'T04069',
    # 'T06782',
    # 'T08346',
    # 'T06569'
]

tile_ids = tile_ids + tile_ids_too

#%%
from ezphot.utils import SDTDataQuerier
import time
sdtdataquerier = SDTDataQuerier()

for tile in tile_ids:
    #sdtdataquerier.show_scisourcedata(targetname = tile)
    try:
        sdtdataquerier.sync_scidata(targetname = tile)
    except:
        pass
    time.sleep(10)
#%%
event_start_date = '2025-11-10'


filters = ['m400', 'm425', 'm450', 'm475', 'm500', 'm525', 'm550', 'm575', 'm600', 'm625', 'm650', 'm675', 'm700', 'm725', 'm750', 'm775', 'm800', 'm825', 'm850', 'm875', 'g', 'r', 'i']
#filters = ['g', 'r', 'i']
def process_image(filter):
    try:
        target_imgset = db.search(pattern = 'calib*.com.fits', return_type = 'science')
        target_imgset.select_images(filter = filter, obs_start = event_start_date)
        target_imglist = target_imgset.target_images
        target_img = target_imglist[0]
        target_imgset.select_images(filter = filter, obs_end = event_start_date)
        target_refimglist = target_imgset.target_images

        target_ivpmask = target_img.calculate_invalidmask(
            verbose = True,
            visualize = False,
            save_fig = False)
        
        target_srcmask = target_img.calculate_sourcemask(
            target_srcmask = target_ivpmask,
            save = False,
            verbose = True,
            visualize = False,
            save_fig = False)
        
        target_bkg = target_img.calculate_bkg(
            target_srcmask = target_srcmask,
            target_ivpmask = target_ivpmask,
            is_2D_bkg = True,
            box_size = 256,
            filter_size = 3,
            save = False,
            verbose = True,
            visualize = False,
            save_fig = False)
        
        target_bkgrms = target_img.calculate_bkgrms(
            target_srcmask = target_srcmask,
            target_ivpmask = target_ivpmask,
            box_size = 256,
            filter_size = 3,
            save = False,
            verbose = True,
            visualize = False,
            save_fig = False)
        
        target_img.header['EGAIN'] *= 100
        target_catalog = target_img.photometry_sex(
            target_bkg = target_bkg,
            target_bkgrms = target_bkgrms,
            detection_sigma = 5,
            aperture_diameter_arcsec = [5,7,10],
            aperture_diameter_seeing = [3.5,4.5],
            saturation_level = 60000,
            kron_factor = 2.5,
            save = False,
            verbose = True,
            visualize = False,
            save_fig = False
        )
        
        target_img.photometric_calibration(
            target_catalog = target_catalog,
            catalog_type = 'GAIAXP',
            max_distance_second = 1.0,
            calculate_color_terms = True,
            calculate_mag_terms = True,
            mag_lower = 13,
            mag_upper = 15,
            visualize = False,
            dynamic_mag_range = False,
            save = True,
            save_fig = False)


        is_hips2fits = False
        if len(target_refimglist) == 0:
            if filter in ['g', 'r', 'i']:
                from ezphot.utils import ImageQuerier
                imagequerier = ImageQuerier(f'SkyMapper/SMSS4/{filter}')
                ra = target_img.ra
                dec = target_img.dec
                is_exist = imagequerier.check_coverage(ra = ra, dec = dec, radius_deg = 1)
                if is_exist[f'SkyMapper/SMSS4/{filter}']:
                    reference_img = imagequerier.query(
                        width = target_img.naxis1,
                        height = target_img.naxis2,
                        ra = ra,
                        dec = dec,
                        pixelscale = target_img.pixelscale[0],
                        telinfo = target_img.telinfo,
                        save_path = target_img.savedir / f'{target_img.objname}_ref_{filter}.fits',
                        objname = target_img.objname)
                    reference_img = reference_img.to_scienceimage()
                    reference_img.header['TELESCOP'] = '7DT'
                    reference_img.header['GAIN'] = 2750
                    is_hips2fits = True
                else:
                    return filter, False
            else:
                return filter, False
                 
        elif len(target_refimglist) > 1:
            ref_seeinglist = [ref_img.seeing for ref_img in target_refimglist]
            reference_img = target_refimglist[np.argmin(ref_seeinglist)]
        else:
            reference_img = target_refimglist[0]
        print(filter, "target seeing/depth:", target_imglist[0].seeing, target_imglist[0].depth, "reference seeing/depth:", reference_img.seeing, reference_img.depth)
        
        
        reference_ivpmask = reference_img.calculate_invalidmask(
            verbose = True,
            visualize = False,
            save_fig = False)
        
        reference_srcmask = reference_img.calculate_sourcemask(
            target_srcmask = reference_ivpmask,
            save = False,
            verbose = True,
            visualize = False,
            save_fig = False)
        
        reference_bkg = reference_img.calculate_bkg(
            target_srcmask = reference_srcmask,
            target_ivpmask = reference_ivpmask,
            is_2D_bkg = True,
            box_size = 256,
            filter_size = 3,
            save = False,
            verbose = True,
            visualize = False,
            save_fig = False)
        
        reference_bkgrms = reference_img.calculate_bkgrms(
            target_srcmask = reference_srcmask,
            target_ivpmask = reference_ivpmask,
            box_size = 256,
            filter_size = 3,
            save = False,
            verbose = True,
            visualize = False,
            save_fig = False)
        
        if not is_hips2fits:
            reference_img.header['EGAIN'] *= 100

        reference_catalog = reference_img.photometry_sex(
            target_bkg = reference_bkg,
            target_bkgrms = reference_bkgrms,
            detection_sigma = 5,
            aperture_diameter_arcsec = [5,7,10],
            aperture_diameter_seeing = [3.5,4.5],
            saturation_level = 60000,
            kron_factor = 2.5,
            save = False,
            verbose = True,
            visualize = False,
            save_fig = False)
        
        reference_img.photometric_calibration(
            target_catalog = reference_catalog,
            catalog_type = 'GAIAXP',
            max_distance_second = 1.0,
            calculate_color_terms = True,
            calculate_mag_terms = True,
            mag_lower = 13,
            mag_upper = 16,
            dynamic_mag_range = False,
            fwhm_sigma = 8,
            visualize = False,
            save = True,
            save_fig = False)
        
        if reference_img.seeing > target_img.seeing:
            convim = 'i'
        else:
            convim = 't'
        
        result = subtract.find_transients(
            target_img = target_img,
            reference_imglist = [reference_img],
            target_bkg = target_bkg,
            
            detection_sigma = 5,
            aperture_diameter_arcsec = [5,7,10],
            aperture_diameter_seeing = [3.5,4.5],
            kron_factor = 2.5,
            catalog_type = 'GAIAXP',
            
            target_transient_number = 5,
            reject_variable_sources = False,
            negative_detection = True,
            reverse_subtraction = False,

            save = True,
            verbose = True,
            visualize = False,
            save_transient_figure = True,
            save_candidate_figure = True,
            show_transient_numbers = 100,
            show_candidate_numbers = 100,
            convim = convim,
            nrx = 1,
            nry = 1,
            nsx = 10,
            nsy = 10,
            ko = 3,
            bgo = 1,
            r = 10)
    
        return filter, True
    except:
        return filter, False
    
    
#%%

for tile in tile_ids:
    # tile = tile_ids[0]
    db = DataBrowser('scidata')
    db.observatory = '7DT'
    db.objname = tile
    target_imgset = db.search(pattern = 'calib*.com.fits', return_type = 'science')
    from ezphot.methods import Subtract
    subtract = Subtract()

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=23) as executor:
        futures = [executor.submit(process_image, filter) for filter in filters]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                print(result)
    time.sleep(5)
    # %%