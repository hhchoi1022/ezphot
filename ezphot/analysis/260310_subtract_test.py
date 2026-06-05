

#%%
'''
In this code, we are going to test the subtraction method in ezphot.
1. Check the subtraction performance with different seeing conditions.
2. Check the subtraction performance with different background levels.
'''
#%% Prepare reference frame
trigger_date = '2026-03-01'
from bridge.connector import GWPortalConnector
# gwportal_connector = GWPortalConnector('raw')
# raw_tbl = gwportal_connector.query(obs_end_date = trigger_date, tile_name = 'T01222')
#%%
from ezphot.utils import DataBrowser
db = DataBrowser('scidata')
db.objname = 'T01222'
db.filter = 'm475'
target_imgset = db.search('7DT*.fits', return_type = 'science')
target_imgset.select_images(obs_end = trigger_date)
file_pathlist = [target_img.path for target_img in target_imgset.target_images]
#%%
import gc
from ezphot.imageobjects import ScienceImage
def single_process(filepath):
    try:
        target_img = ScienceImage(filepath)
        # mbias = target_img.get_masterframe(imagetyp = 'BIAS', max_days = 50)[0]
        # mdark = target_img.get_masterframe(imagetyp = 'DARK', max_days = 50)[0]
        # mflat = target_img.get_masterframe(imagetyp = 'FLAT', max_days = 50)[0]
        # target_img = target_img.correct_bdf(bias_image = mbias, dark_image = mdark, flat_image = mflat, save = True, verbose = False)
        # target_img = target_img.platesolve(overwrite = True, verbose = False)
        target_circularmask = target_img.calculate_circularmask(x_position = 114.0822898, y_position = -69.5412758, radius_arcsec = 200, visualize = False, verbose = False)
        target_srcmask = target_img.calculate_sourcemask(target_srcmask = target_circularmask, sigma = 5, mask_radius_factor = 3, save= True, verbose = False, visualize = False)
        target_bkg = target_img.calculate_bkg(target_srcmask = target_srcmask, save = True, verbose = False, visualize = False, is_2D_bkg = True)
        target_bkgrms = target_img.calculate_bkgrms_from_propagation(target_bkg = target_bkg, mbias = mbias, mdark = mdark, mflat = mflat, save = True, verbose = False, visualize = False)
        target_catalog = target_img.photometry_sex(
            target_bkg = target_bkg,
            target_bkgrms = target_bkgrms,
            detection_sigma = 1.5,
            save = True,
            verbose = False,
            visualize = False,
            save_fig = False
        )
        if target_img.filter in ['g','r','i']:
            mag_lower = 12
            mag_upper = 15
        elif target_img.filter in ['m450','m475','m500','m525','m550','m575','m600','m625']:
            mag_lower = 11
            mag_upper = 14
        else:
            mag_lower = 10
            mag_upper = 13
        
        target_img, target_catalog, reference_catalog, updated_kwargs = target_img.photometric_calibration(
            target_catalog = target_catalog,
            catalog_type = 'GAIAXP',
            catalog_version = 'v1',
            max_distance_second = 2.5,
            min_number_of_sources = 10,
            calculate_color_terms = True,
            calculate_mag_terms = True,
            mag_lower = mag_lower,
            mag_upper = mag_upper,
            snr_lower = 10,
            snr_upper = 150,
            dynamic_mag_range = False,
            classstar_lower = 0.8,
            elongation_upper = 1.7,
            elongation_sigma = 5,
            fwhm_lower_arcsec = 1,
            fwhm_upper_arcsec = 10,
            fwhm_sigma = 5,
            flag_upper = 1,
            maskflag_upper = 1,
            ra_deg = None,
            dec_deg = None,
            radius_arcsec = 1500,
            inner_fraction = 0.7, # Fraction of the images
            isolation_radius_arcsec = 10.0,
            
            magnitude_key = 'MAG_AUTO',
            magnitudeerr_key = 'MAGERR_AUTO',
            fwhm_key = 'FWHM_WORLD',
            ra_key = 'X_WORLD',
            dec_key = 'Y_WORLD',
            classstar_key = 'CLASS_STAR',
            elongation_key = 'ELONGATION',
            flag_key = 'FLAGS',
            maskflag_key = 'IMAFLAGS_ISO',

            # Other parameters
            update_header = True,
            save = True,
            verbose = False,
            visualize = False,
            save_fig = False,
            save_refcat = True
        )
        
        del target_img, mbias, mdark, mflat, target_srcmask, target_bkg, target_bkgrms, target_catalog, reference_catalog
        gc.collect()
    except:
        pass

from tqdm import tqdm
import multiprocessing as mp

with mp.Pool(processes=16) as pool:
    batch_results = list(tqdm(pool.imap(single_process, tqdm(file_pathlist, desc = 'Processing...')), total=len(file_pathlist)))

#%%

from ezphot.utils import DataBrowser
db = DataBrowser('scidata')
db.objname = 'T01222'
db.filter = 'm475'
target_imgset = db.search('7DT*.fits', return_type = 'science')
target_imgset.select_images(obs_start = trigger_date)
target_imgsetlist = target_imgset.divide_images()
#%%
for target_imgset in target_imgsetlist:
    target_imglist, target_bkgrmslist = target_imgset.prepare_stack(
        n_proc = 4,
        scale = True,
        zp_key = 'ZP_APER_2',
        convolve = False,
        reproject = True,
        verbose = False,
        save = True,
        clear = True
    )
    
    stacked_img, stacked_bkgrms = target_imgset.stack(n_proc = 16, combine_type = 'median', remove_intermediate = True)
    stacked_catalog = stacked_img.photometry_sex(
        target_bkg = None,
        target_bkgrms = stacked_bkgrms,
        detection_sigma = 1.5,
        save = True,
        verbose = False,
        visualize = True,
        save_fig = False
    )
    
    if stacked_img.filter in ['g','r','i']:
        mag_lower = 12
        mag_upper = 15
    elif stacked_img.filter in ['m450','m475','m500','m525','m550','m575','m600','m625']:
        mag_lower = 11
        mag_upper = 14
    else:
        mag_lower = 10
        mag_upper = 13
        
    stacked_img, stacked_catalog, reference_catalog, updated_kwargs = stacked_img.photometric_calibration(
        target_catalog = stacked_catalog,
        catalog_type = 'GAIAXP',
        catalog_version = 'v1',
        max_distance_second = 2.5,
        min_number_of_sources = 10,
        calculate_color_terms = True,
        calculate_mag_terms = True,
        mag_lower = mag_lower,
        mag_upper = mag_upper,
        snr_lower = 10,
        snr_upper = 150,
        dynamic_mag_range = False,
        classstar_lower = 0.8,
        elongation_upper = 1.7,
        elongation_sigma = 5,
        fwhm_lower_arcsec = 1,
        fwhm_upper_arcsec = 10,
        fwhm_sigma = 5,
        flag_upper = 100,
        maskflag_upper = 1,
        ra_deg = None,
        dec_deg = None,
        radius_arcsec = 1500,
        inner_fraction = 0.7, # Fraction of the images
        isolation_radius_arcsec = 10.0,
        
        magnitude_key = 'MAG_AUTO',
        magnitudeerr_key = 'MAGERR_AUTO',
        fwhm_key = 'FWHM_WORLD',
        ra_key = 'X_WORLD',
        dec_key = 'Y_WORLD',
        classstar_key = 'CLASS_STAR',
        elongation_key = 'ELONGATION',
        flag_key = 'FLAGS',
        maskflag_key = 'IMAFLAGS_ISO',

        # Other parameters
        update_header = True,
        save = True,
        verbose = True,
        visualize = True,
        save_fig = False,
        save_refcat = True
    )
#%%

from ezphot.utils import DataBrowser
db = DataBrowser('scidata')
db.objname = 'T01222'
db.filter = 'm475'
target_imgset = db.search('*com.fits', return_type = 'science')
#%%
target_img = target_imgset.target_images[0]
reference_img = target_imgset.target_images[-1]
# %%

from ezphot.methods import Subtract
# %%
subtract = Subtract()
# %%
result_1=  subtract.find_transients(
    target_img = target_img,
    reference_imglist = [reference_img],
    
    detection_sigma = 1.5,
    aperture_diameter_arcsec = [5, 7, 10],
    aperture_diameter_seeing = [3.5, 4.5],
    kron_factor = 1.5,
    catalog_type = 'GAIAXP',
    reject_variable_sources = False,
    negative_detection = True,
    reverse_subtraction = False,
    save_transient_figure = True,
    save_candidate_figure = True,
    show_transient_numbers = 100,
    show_candidate_numbers = 100,
    save = False,
    verbose = True,
    visualize = True,
    bgo = 2,
    ko = 2,
    nrx = 2,
    nry = 2,
    r = 21
)
# %%
reference_img = target_imgset.target_images[-2]
result_2 = subtract.find_transients(
    target_img = target_img,
    reference_imglist = [reference_img],
    
    detection_sigma = 1.5,
    aperture_diameter_arcsec = [5, 7, 10],
    aperture_diameter_seeing = [3.5, 4.5],
    kron_factor = 1.5,
    catalog_type = 'GAIAXP',
    reject_variable_sources = False,
    negative_detection = True,
    reverse_subtraction = False,
    save_transient_figure = True,
    save_candidate_figure = True,
    show_transient_numbers = 100,
    show_candidate_numbers = 100,
    save = False,
    verbose = True,
    visualize = True
)
# %%