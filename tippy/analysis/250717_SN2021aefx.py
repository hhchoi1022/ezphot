

#%%
from tippy.photometry import *
from tippy.image import *
from tippy.helper import Helper
from tippy.utils import SDTData
from tippy.helper import TIPDataBrowser

from tqdm import tqdm
import gc
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
#matplotlib.use('Agg') 
#%%
# Initialize the classes
Preprocessor = TIPPreprocess()
PlateSolver = TIPPlateSolve()
MaskGenerator = TIPMasking()
BkgGenerator = TIPBackground()
ErrormapGenerator = TIPErrormap()
AperturePhotometry = TIPAperturePhotometry()
PSFPhotometry = TIPPSFPhotometry()
PhotometricCalibration = TIPPhotometricCalibration()
Stacking = TIPStacking()
helper = Helper()

print(Preprocessor)
print(PlateSolver)
print(MaskGenerator)
print(BkgGenerator)
print(ErrormapGenerator)
print(AperturePhotometry)
print(PSFPhotometry)
print(Stacking)
#%% Load the data
databrowser = TIPDataBrowser('scidata')
databrowser.observatory = 'KCT'
databrowser.objname = 'NGC1566'
databrowser.keys
target_imglist = databrowser.search(pattern='Calib*120.fits', return_type='science')
#%%
### CONFIOGURATION FOR SINGLE IMAGE PROCESSING
visualize = False
verbose = False
save_fig = True
save = True


do_platesolve = False
platesolve_kwargs = dict(
    overwrite = True,
    verbose = verbose,
    )

do_generatemask = True
mask_kwargs = dict(
    sigma = 5,
    radius_factor = 1.5,
    saturation_level = 35000,
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig
)

do_circularmask = True
circlemask_kwargs = dict(
    x_position = 65.0013660,
    y_position = -54.9378261,
    radius_arcsec = 150,
    unit = 'deg',
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig
)

do_objectmask = False
objectmask_kwargs = dict(
    x_position = 245.8960980,
    y_position = -26.5227669,
    radius_arcsec = 600,
    unit = 'deg',
    save = False,
    verbose = verbose,
    visualize = visualize,
    save_fig = True
)

do_generatebkg = True
bkg_kwargs = dict(
    box_size = 64,
    filter_size = 3,
    n_iterations = 0,
    mask_sigma = 3,
    mask_radius_factor = 3,
    mask_saturation_level = 35000,
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig
)

do_generateerrormap = True
errormap_from_propagation = False
errormap_kwargs = dict(
    save = True,
    verbose = verbose,
    visualize = visualize,
    save_fig = True
)
errormap_from_sourcemask = True
errormap_mask_kwargs = dict(
    box_size = 64,
    filter_size = 3,
    errormap_type = 'bkgrms',
    mode = 'sep',
    n_iterations = 0,
    mask_sigma = 3,
    mask_radius_factor = 3,
    mask_saturation_level = 35000,
    bkg_estimator = 'median',
    save =save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig
)

do_aperturephotometry = True
aperturephotometry_kwargs = dict(
    sex_params = None,
    detection_sigma = 1.5,
    aperture_diameter_arcsec = [5,7,10],
    saturation_level = 35000,
    kron_factor = 2.5,
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig,
)

do_photometric_calibration = True
photometriccalibration_kwargs = dict(
    catalog_type = 'APASS',
    max_distance_second = 1.0,
    calculate_color_terms = True,
    calculate_mag_terms = True,
    mag_lower = None,
    mag_upper = None,
    snr_lower = 30,
    snr_upper = 500,
    classstar_lower = 0.7,
    elongation_upper = 3,
    elongation_sigma = 5,
    fwhm_lower = 2,
    fwhm_upper = 15,
    fwhm_sigma = 5,
    flag_upper = 1,
    maskflag_upper = 1,
    inner_fraction = 0.95, # Fraction of the images
    isolation_radius = 10.0,
    magnitude_key = 'MAG_APER_1',
    flux_key = 'FLUX_APER_1',
    fluxerr_key = 'FLUXERR_APER_1',
    fwhm_key = 'FWHM_IMAGE',
    x_key = 'X_IMAGE',
    y_key = 'Y_IMAGE',
    classstar_key = 'CLASS_STAR',
    elongation_key = 'ELONGATION',
    flag_key = 'FLAGS',
    maskflag_key = 'IMAFLAGS_ISO',
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig,
    save_refcat = True,
)

### CONFIGURATION FOR STACKING
do_stacking = True
stacking_kwargs = dict(
    combine_type = 'mean',
    n_proc = 8,
    # Clip parameters
    clip_type = 'extrema',
    sigma = 3.0,
    nlow = 1,
    nhigh = 1,
    # Resample parameters
    resample = True,
    resample_type = 'LANCZOS3',
    center_ra = None,
    center_dec = None,
    pixel_scale = None,
    x_size = None,
    y_size = None,
    # Scale parameters
    scale = True,
    scale_type = 'min',
    zp_key = 'ZP_APER_1',
    # Convolution parameters
    convolve = False,
    seeing_key = 'SEEING',
    kernel = 'gaussian',
    # Other parameters
    verbose = verbose,
    save = save
    )

### CONFIGURATION FOR STACKED IMAGE PROCESSING

stack_aperturephotometry_kwargs = dict(
    sex_params = None,
    detection_sigma = 1.5,
    aperture_diameter_arcsec = [5,7,10],
    saturation_level = 35000,
    kron_factor = 2.5,
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig,
)

stack_photometriccalibration_kwargs = dict(
    catalog_type = 'GAIAXP',
    max_distance_second = 2.0,
    calculate_color_terms = True,
    calculate_mag_terms = True,

    mag_lower = None,
    mag_upper = None,
    snr_lower = 30,
    snr_upper = 500,
    classstar_lower = 0.3,
    elongation_upper = 3,
    elongation_sigma = 5,
    fwhm_lower = 2,
    fwhm_upper = 15,
    fwhm_sigma = 5,
    flag_upper = 1,
    maskflag_upper = 1,
    inner_fraction = 0.95, # Fraction of the images
    isolation_radius = 10.0,
    magnitude_key = 'MAG_APER_1',
    flux_key = 'FLUX_APER_1',
    fluxerr_key = 'FLUXERR_APER_1',
    fwhm_key = 'FWHM_IMAGE',
    x_key = 'X_IMAGE',
    y_key = 'Y_IMAGE',
    classstar_key = 'CLASS_STAR',
    elongation_key = 'ELONGATION',
    flag_key = 'FLAGS',
    maskflag_key = 'IMAFLAGS_ISO',
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig,
    save_starcat = False,
    save_refcat = True,
)

#%%
# Set telescope information
# Define the image processing function
# 76 images -> 9min 41s
def imgprocess(target_img):
    #target_img = ScienceImage(path = target_path, telinfo = telinfo, load = True)
    # mbias_path = Preprocessor.get_masterframe_from_image(target_img, imagetyp = 'BIAS', max_days = 60)[0]
    # mdark_path = Preprocessor.get_masterframe_from_image(target_img, imagetyp = 'DARK', max_days = 60)[0]
    # mflat_path = Preprocessor.get_masterframe_from_image(target_img, imagetyp = 'FLAT', max_days = 60)[0]
    # mbias = MasterImage(path = mbias_path['file'], telinfo = telinfo, load = True)
    # mdark = MasterImage(path = mdark_path['file'], telinfo = telinfo, load = True)
    # mflat = MasterImage(path = mflat_path['file'], telinfo = telinfo, load = True)
    mbias = None
    mdark = None
    mflat = None

    status = dict()
    status['image'] = target_img.path
    status['platesolve'] = None
    if do_platesolve:
        try:
            target_img = PlateSolver.solve_astrometry(
                target_img=target_img,
                **platesolve_kwargs,
            )
            target_img = PlateSolver.solve_scamp(
                target_img=target_img,
                scamp_sexparams=None,
                scamp_params=None,
                **platesolve_kwargs
            )[0]
            status['platesolve'] = True
        except Exception as e:
            status['platesolve'] = e

    status['mask'] = None
    target_srcmask = None
    if do_generatemask:
        try:
            # Mask the object frames
            target_srcmask = MaskGenerator.mask_sources(
                target_img = target_img,
                target_mask = None,
                **mask_kwargs
            )
            status['mask'] = True
        except:
            status['mask'] = e

    status['circular_mask'] = None
    if do_circularmask:
        try:
            # Generate the circular mask
            target_srcmask = MaskGenerator.mask_circle(
                target_img = target_img,
                target_mask = target_srcmask,
                mask_type = 'source',
                **circlemask_kwargs
            )
            status['circular_mask'] = True

        except Exception as e:
            status['circular_mask'] = e

    status['object_mask'] = None
    target_objectmask = None
    if do_objectmask:
        try:
            # Generate the circular mask
            target_objectmask = MaskGenerator.mask_circle(
                target_img = target_img,
                target_mask = None,
                mask_type = 'invalid',
                **objectmask_kwargs
            )
            status['object_mask'] = True

        except Exception as e:
            status['object_mask'] = e

    status['background'] = None
    target_bkg = None
    if do_generatebkg:
        try:
            # Generate the background
            target_bkg, bkg_instance = BkgGenerator.calculate_sep(
                target_img = target_img,
                target_mask = target_srcmask,
                ** bkg_kwargs
            )
            status['background'] = True
        except Exception as e:
            status['background'] = e

    status['errormap'] = None
    target_bkgrms = None
    if do_generateerrormap:
        try:
            if errormap_from_propagation:
                if target_bkg is None:
                    target_bkg, bkg_instance = BkgGenerator.calculate_sep(
                        target_img = target_img,
                        target_mask = target_srcmask,
                        ** bkg_kwargs)
                target_bkgrms = ErrormapGenerator.calculate_from_bkg(
                    target_bkg = target_bkg,
                    egain = target_img.egain,
                    readout_noise = target_img.telinfo['readnoise'],
                    mbias_img = mbias,
                    mdark_img = mdark,
                    mflat_img = mflat,
                    **errormap_kwargs
                )
                status['errormap'] = True
            if errormap_from_sourcemask:
                target_bkgrms = ErrormapGenerator.calculate_from_sourcemask(
                    target_img = target_img,
                    target_mask = target_srcmask,
                    **errormap_mask_kwargs
                )[0]
                status['errormap'] = True
        except Exception as e:
            status['errormap'] = e

    if do_aperturephotometry:
        try:
            # Perform aperture photometry
            sex_catalog = AperturePhotometry.sex_photometry(
                target_img = target_img,
                target_bkg = target_bkg,
                target_bkgrms = target_bkgrms,
                target_mask = target_objectmask, ################################ HERE, Central stars are masked for ZP caclculation
                **aperturephotometry_kwargs
            )
            status['aperture_photometry'] = True
        except Exception as e:
            status['aperture_photometry'] = e

    if do_photometric_calibration and do_aperturephotometry:
        try:
            target_img, calib_catalog, filtered_catalog = PhotometricCalibration.photometric_calibration(
                target_img = target_img,
                target_catalog = sex_catalog,
                **photometriccalibration_kwargs
            )
            status['photometric_calibration'] = True
        except Exception as e:
            status['photometric_calibration'] = e

    # Clean up memory
    del target_srcmask, target_objectmask
    target_img.data = None
    target_bkg.data = None
    target_bkgrms.data = None
    gc.collect()
    return target_img, target_bkg, target_bkgrms, calib_catalog, status
#%%
# Process the images (Masking, Background, Errormap, Aperture Photometry, Photometric Calibration)
with ProcessPoolExecutor(max_workers=40) as executor:
    futures = [executor.submit(imgprocess, target_img) for target_img in target_imglist]

    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = future.result()
            if result is not None:
                target_img, target_bkg, target_bkgrms, target_catalog, status = result
            else:
                print(f"[WARNING] Skipped a result due to None")
        except Exception as e:
            print(f"[ERROR] {e}")
#%%
# Stack the images
imginfo_all = databrowser.search(pattern='Calib*120.fits', return_type='imginfo')
imginfo_groups = imginfo_all.group_by(['filter', 'telescop']).groups
#%%
stack_imglist = []
stack_bkgrmslist = []
failed_imglist = []
for imginfo_group in imginfo_groups:
    imginfo_group = helper.group_table(imginfo_group, 'mjd', 0.2)
    imginfo_subgroups = imginfo_group.group_by('group').groups
    for imginfo_subgroup in imginfo_subgroups:
        target_imglist = [ScienceImage(path=row['file'], telinfo=telinfo, load=True) for row in imginfo_subgroup]
        target_imglist = [target_img for target_img in target_imglist if 'ZP_APER' in target_img.header.keys()]
        target_bkglist = [Background(path = target_img.savepath.bkgpath, load=True) for target_img in target_imglist]
        target_bkgrmslist = [Errormap(path = target_img.savepath.bkgrmspath, emaptype = 'bkgrms', load=True) for target_img in target_imglist]

        if len(target_imglist) == 0:
            print(f"[WARNING] No images found. skipping stacking.")
            continue
        try:
            stack_img, stack_bkgrms = Stacking.stack_multiprocess(
                target_imglist = target_imglist,
                target_bkglist = target_bkglist,
                target_bkgrmslist = target_bkgrmslist,
                target_outpath = None,
                bkgrms_outpath = None,
                **stacking_kwargs
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

def stackprocess(stack_path, stack_bkgrmspath, telinfo):
    stack_img = ScienceImage(path=stack_path, telinfo=telinfo, load=True)
    stack_bkgrms = Errormap(path=stack_bkgrmspath, emaptype='bkgrms', load=True)
    stacked_status = dict()
    try:
        # Perform aperture photometry
        aperturephotometry_kwargs = stack_aperturephotometry_kwargs.copy()
        aperturephotometry_kwargs['aperture_diameter_arcsec'].extend([2.5* stack_img.seeing, 3.5*stack_img.seeing])
        sex_catalog = AperturePhotometry.sex_photometry(
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
        stack_img, calib_catalog, filtered_catalog = PhotometricCalibration.photometric_calibration(
            target_img = stack_img,
            target_catalog = sex_catalog,
            **stack_photometriccalibration_kwargs
        )
        stacked_status['photometric_calibration'] = True
    except Exception as e:
        stacked_status['photometric_calibration'] = e
        
    # Clean up memory
    stack_img.data = None
    stack_bkgrms.data = None
    gc.collect()
    return stack_img, stack_bkgrms, calib_catalog, stacked_status

arglist = [(stack_img.path, stack_bkgrms.path, telinfo) for stack_img, stack_bkgrms in zip(stack_imglist, stack_bkgrmslist)]
with ProcessPoolExecutor(max_workers=10) as executor:
    failed_stacked_path = []
    futures = [executor.submit(stackprocess, *args) for args in arglist]
    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = future.result()
        except Exception as e:
            failed_stacked_path.append(args[0])
            print(f"[ERROR] {e}")
        
# # %%
# helper = Helper()
# data_qurier = SDTData()
# stack_images = data_qurier.show_scidestdata(
#     'T01462',
#     False,
#     'filter',
#     'calib*100.com.fits'
# )
# stack_pathlist_all = [stack_imgpath for stack_pathlist in stack_images.values() for stack_imgpath in stack_pathlist]
# stack_imglist = [ScienceImage(path=path, telinfo=telinfo, load=True) for path in stack_pathlist_all]
# stack_bkgrmslist = [Errormap(path=img.savepath.bkgrmspath, emaptype='bkgrms', load=True) for img in stack_imglist]


# from tippy.photometry import TIPSubtraction
# DIA = TIPSubtraction()

# for stack_img, stack_bkgrms in zip(stack_imglist, stack_bkgrmslist):
#     refinfo_tbl = DIA.get_referenceframe_from_image(stack_img)
#     if refinfo_tbl is None or len(refinfo_tbl) == 0:
#         print(f"[WARNING] No reference frame found for {stack_img.path}, skipping DIA.")
#         if stack_img.filter.startswith('m'):
#             print('No reference frame is found. Skipping DIA')
#         else:
        
#     reference_img = ReferenceImage(refinfo_tbl['file'], telinfo=stack_img.telinfo, load=True)
#     reference_img.header.remove('SEEING')
# %%
import time
print('Current time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
# %%
