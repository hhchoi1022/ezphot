
#%%
from bridge.alertmonitor import AlertProcessor
from bridge.objects import Alert
from astropy.time import Time
alert_instance = Alert(objname = 'T19136')
alert_instance.trigger_time = Time('2001-01-01')
processor = AlertProcessor()
processor.load_files_gwportal(alert_instance = alert_instance)
# processor.load_files_ezphot(alert_instance = alert_instance, file_pattern = '*com.fits')
processor.target_images = [target_img for target_img in processor.target_images if target_img.is_saved == False]
processor.pipeline_before_stacking(alert_instance)
processor.stacking(alert_instance)
# processor.pipeline_after_stacking(alert_instance)
#%%
# shift_map_filter = {
#     'm400': 5,
#     'm425': 15,
#     'm500': -10,
#     'm525': -15,
#     'm550': -35,
#     'm575': -25,
#     'm600': -50,
#     'm625': -35,
#     'm650': -15,
#     'm675': -50,
#     'm700': -50,
#     'm725': -10,
#     'm750': -35,
#     'm775': -45,
#     'm800': 40,
#     'm825': 15,
#     'm875': -20}

#%%
def process_one_image(img_path):
    """
    Run full photometry + calibration for one ScienceImage.
    """
    from ezphot.imageobjects import ScienceImage

    target_img = ScienceImage(img_path)
    target_img.write()

    # --- masks ---
    target_ivpmask = target_img.calculate_invalidmask(
        save=False,
        verbose=False,
        visualize=False,
        save_fig=False,
    )

    target_srcmask = target_img.calculate_sourcemask(
        target_srcmask=None,
        sigma=5,
        mask_radius_factor=3,
        visualize=False,
        save=True,
    )

    target_bkg = target_img.calculate_bkg(
        target_srcmask=target_srcmask,
        target_ivpmask=target_ivpmask,
        box_size=256,
        filter_size=3,
        correct_global_offset=True,
        visualize=False,
        save=True,
    )

    # --- photometry ---
    target_catalog = target_img.photometry_sex(
        target_bkg=target_bkg,
        target_bkgrms=target_img.bkgrms,
        target_mask=None,
        sex_params=None,
        detection_sigma=2.5,
        aperture_diameter_arcsec=[5, 7, 10],
        aperture_diameter_seeing=[3.5, 4.5, 5.5],
        annulus_width_arcsec=None,
        saturation_level=60000,
        kron_factor=2.5,
        save=True,
        verbose=False,
        visualize=False,
        save_fig=False,
    )
    
    filter = target_img.filter
    # --- magnitude range ---
    if filter in ['g', 'r']:
        mag_lower = 13
        mag_upper = 16.5
    elif filter in ['i']:
        mag_lower = 12
        mag_upper = 16
    elif filter in ['m425', 'm450', 'm475', 'm500', 'm525', 'm550', 'm575', 'm600', 'm625', 'm650']:
        mag_lower = 11.5
        mag_upper = 15.5
    else:
        mag_lower = 10.5
        mag_upper = 14.5

    # --- calibration ---
    target_img, target_catalog, target_refcat, update_kwargs = target_img.photometric_calibration(
        target_catalog=target_catalog,
        # catalog_type='GAIAXP_CORR_LAMOST',
        # catalog_version=f'v{shift_map_filter[filter]}',
        catalog_type='GAIAXP',
        catalog_version='v1',
        
        max_distance_second=2.5,
        min_number_of_sources=10,
        calculate_color_terms=True,
        calculate_mag_terms=True,

        mag_lower=mag_lower,
        mag_upper=mag_upper,
        dynamic_mag_range=False,
        classstar_lower=0.8,
        elongation_upper=1.7,
        elongation_sigma=5,
        fwhm_lower_arcsec=1,
        fwhm_upper_arcsec=10,
        fwhm_sigma=5,
        flag_upper=1,
        maskflag_upper=1,
        ra_deg=None,
        dec_deg=None,
        radius_arcsec=1500,
        inner_fraction=0.7,
        isolation_radius_arcsec=10.0,

        magnitude_key='MAG_AUTO',
        magnitudeerr_key='MAG_AUTO',
        fwhm_key='FWHM_WORLD',
        ra_key='X_WORLD',
        dec_key='Y_WORLD',
        classstar_key='CLASS_STAR',
        elongation_key='ELONGATION',
        flag_key='FLAGS',
        maskflag_key='IMAFLAGS_ISO',

        update_header=True,
        save=True,
        verbose=False,
        visualize=True,
        save_fig=True,
        save_refcat=True,
    )

    # target_catalog = target_img.apply_mag_terms(target_catalog = target_catalog, 
    #                                             verbose = True,
    #                                             save = False)
    # target_catalog = target_img.apply_color_terms(target_catalog = target_catalog, 
    #                                               verbose = True,
    #                                               save = False)
    
    
    return target_img.path
#%%
import glob
img_paths = glob.glob('/home/hhchoi1022/tract7dt/images/*')
img_path = img_paths[0]
#%%
from ezphot.utils import DataBrowser

tile_id = 'T22272'

dbrowser = DataBrowser('scidata')
dbrowser.observatory = '7DT'
dbrowser.objname = tile_id

target_imgset = dbrowser.search(pattern='*com.fits', return_type='science')

# 🚨 IMPORTANT: pass only paths
img_paths = [img.path for img in target_imgset.target_images]
img_path = img_paths[5]
#%%

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

if __name__ == "__main__":
    n_workers = 24

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_one_image, p) for p in img_paths]

        for fut in as_completed(futures):
            try:
                result = fut.result()
                print(f"[DONE] {result}")
            except Exception as e:
                print(f"[FAIL] {e}")

# %%
