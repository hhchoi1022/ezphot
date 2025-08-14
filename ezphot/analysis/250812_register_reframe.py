#%%
from ezphot.utils import DataBrowser
from ezphot.imageobjects import *
from ezphot.methods import *
import glob
import gc
#%%
filepathlist = glob.glob('/home/hhchoi1022/data/refdata/undefined/*KCT*NGC1566*.fits')
target_imglist = []
for filepath in filepathlist:
    target_img = ReferenceImage(filepath)
    target_imglist.append(target_img)
# %%
do_platesolve = True
platesolve_kwargs = dict(
    overwrite = True,
    verbose = True,
    )

stack_aperturephotometry_kwargs = dict(
    sex_params = None,
    detection_sigma = 5,
    aperture_diameter_arcsec = [7,12,15],
    aperture_diameter_seeing = [3.5, 4.5], # If given, use seeing to calculate aperture size
    saturation_level = 35000,
    kron_factor = 1.5,
    save = True,
    verbose = True,
    visualize = True,
    save_fig = True,
)

stack_photometriccalibration_kwargs = dict(
    catalog_type = 'APASS',
    max_distance_second = 5,
    calculate_color_terms = True,
    calculate_mag_terms = True,
    classstar_lower = 0.5,
    elongation_upper = 3,
    elongation_sigma = 5,
    mag_lower = 11,
    mag_upper = 13,
    dynamic_mag_range = True,
    fwhm_lower = 1,
    fwhm_upper = 15,
    fwhm_sigma = 5,
    flag_upper = 1,
    maskflag_upper = 1,
    inner_fraction = 0.8, # Fraction of the images
    isolation_radius = 15.0,
    magnitude_key = 'MAG_APER_3',
    fwhm_key = 'FWHM_IMAGE',
    x_key = 'X_IMAGE',
    y_key = 'Y_IMAGE',
    classstar_key = 'CLASS_STAR',
    elongation_key = 'ELONGATION',
    flag_key = 'FLAGS',
    maskflag_key = 'IMAFLAGS_ISO',
    save = True,
    verbose = True,
    visualize = True,
    save_fig = True,
    save_refcat = True,
)

platesolve = Platesolve()
aperphot = AperturePhotometry()
photcal = PhotometricCalibration()

def stackprocess(stack_img):
    stacked_status = dict()

    if do_platesolve:
        try:
            target_img = platesolve.solve_astrometry(
                target_img=stack_img,
                **platesolve_kwargs,
            )
            stack_img = platesolve.solve_scamp(
                target_img=stack_img,
                scamp_sexparams=None,
                scamp_params=None,
                **platesolve_kwargs
            )[0]
            stacked_status['platesolve'] = True
        except Exception as e:
                stacked_status['platesolve'] = e
                
    stack_bkgrms = Errormap(path = stack_img.savepath.bkgrmspath, emaptype = 'bkgrms', load = True)
    if not stack_bkgrms.is_exists:
        stack_bkgrms = None
    try:
        # Perform aperture photometry
        sex_catalog = aperphot.sex_photometry(
            target_img = stack_img,
            target_bkg = None,
            target_bkgrms = stack_bkgrms,
            target_mask = None, ################################ HERE, Central stars are masked for ZP caclculation
            **stack_aperturephotometry_kwargs
        )
        stacked_status['aperture_photometry'] = True
    except Exception as e:
        stacked_status['aperture_photometry'] = e

    try:
        stack_img, calib_catalog, filtered_catalog = photcal.photometric_calibration(
            target_img = stack_img,
            target_catalog = sex_catalog,
            **stack_photometriccalibration_kwargs
        )
        stacked_status['photometric_calibration'] = True
    except Exception as e:
        stacked_status['photometric_calibration'] = e
        
    # Clean up memory
    #stack_img.data = None
    #stack_bkgrms.data = None
    gc.collect()
    return stack_img, stacked_status
# %%
for target_img in target_imglist:
    stack_img, stacked_status = stackprocess(target_img)
    stack_img.register()
# %%