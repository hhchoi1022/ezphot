

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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
import matplotlib.pyplot as plt

from ezphot.helper import Helper
from ezphot.dataobjects import LightCurve
from ezphot.skycatalog import SkyCatalog
from ezphot.dataobjects import Catalog
from ezphot.dataobjects import CatalogSet
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
from tqdm import tqdm
from ezphot.methods import Subtract
#%% Load the data
databrowser = DataBrowser('scidata')
databrowser.observatory = 'RASA36'
databrowser.objname = 'NGC1566'
databrowser.telkey = 'RASA36_KL4040_HIGH_1x1'
target_imgset = databrowser.search(pattern='Calib*.fits', return_type='science')
target_imglist = target_imgset.target_images
target_img = target_imglist[0]

#%%
# Set telescope information
# Define the image processing function
# 76 images -> 9min 41s
def imgprocess(target_img):
    # run the expensive steps here
    try:
        # Fill nan to mean value
        target_srcmask = target_img.calculate_sourcemask(
            save = False,
            verbose = False,
            visualize = False,
            save_fig = False)

        target_bkg = target_img.calculate_bkg(
            target_srcmask = target_srcmask,
            target_ivpmask = None,
            is_2D_bkg = True,
            box_size = 64,
            filter_size = 3,
            save = True,
            verbose = False,
            visualize = False,
            save_fig = False)
        
        target_bkgrms = target_img.calculate_bkgrms(
            target_srcmask = target_srcmask,
            target_ivpmask = None,
            box_size = 64,
            filter_size = 3,
            save = True,
            verbose = False,
            visualize = False,
            save_fig = False)
        
        target_catalog = target_img.photometry_sex(
            target_bkg = target_bkg,
            target_bkgrms = target_bkgrms,
            detection_sigma = 5,
            aperture_diameter_arcsec = [6,9,12],
            aperture_diameter_seeing = [3.5,4.5],
            saturation_level = 60000,
            kron_factor = 2.5,
            
            save = True,
            verbose = False,
            visualize = False,
            save_fig = False)
        
        target_img, calibrated_catalog, reference_catalog, update_kwargs = target_img.photometric_calibration(
            target_catalog = target_catalog,
            catalog_type = 'APASS',
            max_distance_second = 3,
            calculate_color_terms = True,
            calculate_mag_terms = True,
            mag_lower = 12,
            mag_upper = 15,
            dynamic_mag_range = False,
            classstar_lower = 0.5,
            elongation_upper = 1.7,
            elongation_sigma = 5,
            fwhm_lower = 1,
            fwhm_upper = 10,
            fwhm_sigma = 5,
            flag_upper = 1,
            maskflag_upper = 1,
            inner_fraction = 0.8,
            isolation_radius = 10,
            save = True,
            verbose = False,
            visualize = False,
            save_fig = False,
            save_refcat = False,
        )
    except:
        return target_img.path

#%%
import time

def chunk_list(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def process_batch(batch_images, batch_index, max_workers=8, stagger_delay=0.5):
    print(f"\nStarting batch {batch_index+1} with {len(batch_images)} images...")

    failed_imglist = []
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit jobs with staggered delays to reduce IO contention
        futures = []
        for i, img in enumerate(batch_images):
            # Add small delay between submissions to stagger IO operations
            if i > 0:
                time.sleep(stagger_delay)
            future = executor.submit(imgprocess, img)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_index+1}"):
            try:
                result = future.result()
                if result is not None:
                    failed_imglist.append(result)
            except Exception as e:
                print(f"[ERROR in batch {batch_index+1}] {e}")
    end_time = time.time()
    print(f"Batch {batch_index+1} completed in {end_time - start_time:.2f} seconds")
    # Clean up memory between batches
    gc.collect()
    return failed_imglist

# ? Main loop over batches
batch_size = 64  # Reduced batch size to reduce IO pressure
failed_imglist = []

for batch_index, batch in enumerate(chunk_list(target_imglist, batch_size)):
    # Reduced max_workers and added stagger_delay to reduce IO contention
    batch_results = process_batch(batch, batch_index, max_workers=16, stagger_delay=0.3)
    failed_imglist.extend(batch_results)
#%%
databrowser = DataBrowser('scidata')
databrowser.objname = 'NGC1566'
databrowser.observatory = 'RASA36'
databrowser.telkey = 'RASA36_KL4040_MERGE_1x1'
# Stack the images
imginfo_all = databrowser.search(pattern='Calib*.fits', return_type='imginfo')
imginfo_groups = imginfo_all.group_by(['filter', 'telescop']).groups
#%%
#%%
from ezphot.helper import Helper
from ezphot.methods import Stack

helper = Helper()
stack = Stack()
#%%
stack_imglist = []
stack_bkgrmslist = []
failed_imglist = []
for imginfo_group in imginfo_groups:
    imginfo_group = helper.group_table(imginfo_group, 'mjd', 0.1)
    imginfo_subgroups = imginfo_group.group_by('group').groups
    for imginfo_subgroup in imginfo_subgroups:
        target_imglist = [ScienceImage(path=row['file'], load=True) for row in imginfo_subgroup]
        target_imglist = stack.select_quality_images(target_imglist, seeing_limit = 6, depth_limit = 15, ellipticity_limit = 0.6, max_numbers = len(target_imglist), visualize = False)
        if len(target_imglist) == 0:
            print(f"[WARNING] No images found. skipping stacking.")
            continue
        target_bkglist = [target_img.bkgmap for target_img in target_imglist]
        target_bkgrmslist = [target_img.bkgrms for target_img in target_imglist]
        try:
            stack_img, stack_bkgrms = stack.stack_multiprocess(
                target_imglist = target_imglist,
                target_bkglist = target_bkglist,
                target_bkgrmslist = target_bkgrmslist,
                target_outpath = None,
                bkgrms_outpath = None,
                combine_type = 'median',
                n_proc = 8,
                clip_type = 'extrema',
                sigma = 3.0,
                nlow = 1,
                nhigh = 1,
                resample = True,
                resample_type = 'LANCZOS3',
                center_ra = None,
                center_dec = None,
                pixel_scale = None,
                x_size = None,
                y_size = None,
                scale = True,
                scale_type = 'min',
                zp_key = 'ZP_APER_1',
                convolve = False,
                seeing_key = 'SEEING',
                kernel = 'gaussian',
                verbose = False,
            )
            # Clean up memory
            for target_img in target_imglist:
                target_img.data = None
            for target_bkg in target_bkglist:
                target_bkg.data = None
            for target_bkgrms in target_bkgrmslist:
                target_bkgrms.data = None
            stack_img.data = None
            stack_bkgrms.data = None
                
            stack_imglist.append(stack_img)
            stack_bkgrmslist.append(stack_bkgrms)
        except:
            print(f"[ERROR] Stacking failed, skipping stacking.")
            failed_imglist.extend(target_imglist)
            continue
#%%
def stackprocess(target_img):
    # run the expensive steps here
    try:
        # Fill nan to mean value
        target_bkg = target_img.bkgmap
        target_bkgrms = target_img.bkgrms
        target_catalog = target_img.photometry_sex(
            target_bkg = target_bkg,
            target_bkgrms = target_bkgrms,
            sex_params = dict(BACK_TYPE = 'MANUAL'),
            detection_sigma = 5,
            aperture_diameter_arcsec = [6,9,12],
            aperture_diameter_seeing = [3.5,4.5],
            saturation_level = 60000,
            kron_factor = 2.5,
            
            save = True,
            verbose = False,
            visualize = False,
            save_fig = True)
        
        target_img, calibrated_catalog, reference_catalog, update_kwargs = target_img.photometric_calibration(
            target_catalog = target_catalog,
            catalog_type = 'APASS',
            max_distance_second = 3,
            calculate_color_terms = True,
            calculate_mag_terms = True,
            mag_lower = 12,
            mag_upper = 16,
            dynamic_mag_range = False,
            classstar_lower = 0.5,
            elongation_upper = 1.7,
            elongation_sigma = 5,
            fwhm_lower = 1,
            fwhm_upper = 10,
            fwhm_sigma = 5,
            flag_upper = 1,
            maskflag_upper = 1,
            inner_fraction = 0.8,
            isolation_radius = 10,
            magnitude_key = 'MAG_APER_2',
            save = True,
            verbose = False,
            visualize = False,
            save_fig = False,
            save_refcat = True,
        )
    except:
        return target_img.path

#%%
databrowser = DataBrowser('scidata')
databrowser.observatory = 'RASA36'
databrowser.objname = 'NGC1566'
databrowser.telkey = 'RASA36_KL4040_HIGH_1x1'
stack_imgset = databrowser.search(pattern='Calib*com.fits', return_type='science')
stack_imglist = stack_imgset.target_images
#%%
# run in multiprocess and check failed images
failed_stacked_img = []
with ProcessPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(stackprocess, stack_img) for stack_img in stack_imglist]
    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = future.result()
        except Exception as e:
            failed_stacked_img.append(stack_img)
            print(f"[ERROR] {e}")
#%%
databrowser = DataBrowser('scidata')
databrowser.observatory = 'RASA36'
databrowser.objname = 'NGC1566'
databrowser.telkey = 'RASA36_KL4040_HIGH_1x1'
catalogset = databrowser.search(pattern='Calib*com.fits.cat', return_type='catalog')
catalogset.select_sources(ra = 64.9725, dec= -54.948081)
# %%
lc = LightCurve(catalogset)
lc.plt_params.figure_figsize = (10, 6)
lc.plt_params.xlim= [59500, 59700]
lc.plt_params.ylim = [16, 11]
lc.plot(ra = 64.9725, dec= -54.948081, flux_key = 'MAGSKY_APER_2')
# %% SUBTRACTION
databrowser = DataBrowser('scidata')
databrowser.observatory = 'RASA36'
databrowser.objname = 'NGC1566'
databrowser.telkey = 'RASA36_KL4040_MERGE_1x1'
stack_imgset = databrowser.search(pattern='Calib*com.fits', return_type='science')
stack_imglist = stack_imgset.target_images
#%%
for stack_img in stack_imglist:
    cutout_img = stack_img.cutout(x = 64.9725, y = -54.948081, size_pixel = 2000, coord_type = 'coord')
    cutout_img.write()
#%%
databrowser.telkey = 'RASA36_KL4040_MERGE_1x1'
stack_imgset = databrowser.search(pattern='cutout_*com.fits', return_type='science')
stack_imglist = stack_imgset.target_images
reference_img = stack_imglist[0].get_referenceframe(overlap_threshold = 0)[0]
# target_img = reference_img.to_scienceimage()
# target_img.to_referenceimage().write()
# cutout_ref_img = reference_img.cutout(x = 64.9725, y = -54.948081, size_pixel = 2000, coord_type = 'coord')
# 
#%%
# cutout_ref_img.write()
# cutout_ref_img.register()
# reference_img.deregister()
# reference_img.remove()
# for stack_img in stack_imglist:
#     stack_img.data += 10
#     stack_img.write()
#%%
DIA = Subtract()
#%%
failed_images = []
successful_results = []

def subtract_process(target_img, reference_img):
    try:
        result = DIA.find_transients(
            target_img=target_img, 
            reference_imglist=[reference_img],
            target_bkg=None,
            detection_sigma = 5,
            aperture_diameter_arcsec = [6,9,12],
            aperture_diameter_seeing = [3.5,4.5],
            kron_factor = 2.5,
            save = True,
            verbose = False,
            visualize = False,
            save_transient_figure = False,
            save_candidate_figure = False,
            iu = 60000,
            il = -10000,
            tu = 60000,
            tl = -10000,
        )
        del reference_img
        del target_img
        return result
    except Exception as e:
        return None  # Return failed image as indicator
#%%

successful_results = []
failed_images = []
for target_img in stack_imglist:
    result = subtract_process(target_img, reference_img)
    if result is not None:
        successful_results.append(result)
    else:
        failed_images.append(target_img)
# %%
databrowser = DataBrowser('scidata')
databrowser.objname = 'NGC1566'
databrowser.observatory = 'RASA36'
catalogset = databrowser.search(pattern='sub*.cat', return_type='catalog')
catalogset.select_sources(ra = 64.9725, dec= -54.948081)
# %%
lc = LightCurve(catalogset)
lc.plt_params.figure_figsize = (10, 6)
lc.plt_params.xlim= [59500, 59700]
lc.plt_params.ylim = [20, 11]
lc.FILTER_OFFSET['r'] = 1
lc.plot(ra = 64.9725, dec= -54.948081, flux_key = 'MAGSKY_APER_2')
# %%
