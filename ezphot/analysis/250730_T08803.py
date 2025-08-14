

#%%
from ezphot.methods import *
from ezphot.imageobjects import *
from ezphot.helper import Helper
from ezphot.utils import SDTDataQuerier
from ezphot.utils import DataBrowser
import psutil, os
from pympler import asizeof

from tqdm import tqdm
import gc
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
#matplotlib.use('Agg') 
#%%
# Initialize the classes
Preprocessor = Preprocess()
PlateSolver = TIPPlateSolve()
MaskGenerator = TIPMasking()
BkgGenerator = BackgroundEstimator()
ErrormapGenerator = TIPErrormap()
AperturePhotometry = AperturePhotometry()
PSFPhotometry = TIPPSFPhotometry()
PhotometricCalibration = PhotometricCalibration()
Stacking = Stack()
helper = Helper()

print(Preprocessor)
print(PlateSolver)
print(MaskGenerator)
print(BkgGenerator)
print(ErrormapGenerator)
print(AperturePhotometry)
print(PSFPhotometry)
print(Stacking)
databrowser = DataBrowser('scidata')
databrowser.observatory = '7DT'
databrowser.filter = 'm400'
databrowser.objname = 'T08803'
databrowser.keys

#%% Load the data
target_imglist = databrowser.search(pattern='calib*100.fits', return_type='science')
#%%
### CONFIOGURATION FOR SINGLE IMAGE PROCESSING
visualize = True
verbose = True
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
    mask_radius_factor = 1.5,
    saturation_level = 40000,
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig
)

do_circularmask = False
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
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig
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
    detection_sigma = 5,
    aperture_diameter_arcsec = [5,7,10],
    aperture_diameter_seeing = [3.5, 4.5], # If given, use seeing to calculate aperture size
    saturation_level = 35000,
    kron_factor = 2.5,
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig,
)

do_photometric_calibration = True
photometriccalibration_kwargs = dict(
    catalog_type = 'GAIAXP',
    max_distance_second = 3,
    calculate_color_terms = True,
    calculate_mag_terms = True,
    classstar_lower = 0.5,
    elongation_upper = 3,
    elongation_sigma = 5,
    fwhm_lower = 1,
    fwhm_upper = 15,
    fwhm_sigma = 5,
    flag_upper = 1,
    maskflag_upper = 1,
    inner_fraction = 0.8, # Fraction of the images
    isolation_radius = 15.0,
    magnitude_key = 'MAG_AUTO',
    flux_key = 'FLUX_AUTO',
    fluxerr_key = 'FLUXERR_AUTO',
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
    combine_type = 'median',
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
    zp_key = 'ZP_APER_4',
    # Convolution parameters
    convolve = False,
    seeing_key = 'SEEING',
    kernel = 'gaussian',
    # Other parameters
    verbose = verbose,
    save = save
)
#%%
# Set telescope information
# Define the image processing function
# 76 images -> 9min 41s
def imgprocess(target_img):
    # run the expensive steps here
    Preprocessor = Preprocess()
    PlateSolver = TIPPlateSolve()
    MaskGenerator = TIPMasking()
    BkgGenerator = BackgroundEstimator()
    ErrormapGenerator = TIPErrormap()
    AperturePhotometry = AperturePhotometry()
    PSFPhotometry = TIPPSFPhotometry()
    PhotometricCalibration = PhotometricCalibration()
    Stacking = Stack()
    helper = Helper()
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
            target_bkg, bkg_instance = BkgGenerator.estimate_with_sep(
                target_img = target_img,
                target_mask = target_srcmask,
                ** bkg_kwargs
            )
            status['background'] = True
        except Exception as e:
            status['background'] = e

    status['errormap'] = None
    target_bkgrms = None
    bkg_instance = None
    target_bkg_tmp = None
    if do_generateerrormap:
        try:
            if errormap_from_propagation:
                if target_bkg is None:
                    target_bkg, bkg_instance = BkgGenerator.estimate_with_sep(
                        target_img = target_img,
                        target_mask = target_srcmask,
                        ** bkg_kwargs)
                target_bkgrms = ErrormapGenerator.calculate_bkgrms_from_propagation(
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
                target_bkgrms, target_bkg_tmp, bkg_instance = ErrormapGenerator.calculate_from_sourcemask(
                    target_img = target_img,
                    target_mask = target_srcmask,
                    **errormap_mask_kwargs
                )
                status['errormap'] = True
        except Exception as e:
            status['errormap'] = e

    sex_catalog = None
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

    calib_catalog = None
    filtered_catalog = None
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

    del target_img
    del target_bkg
    del target_bkgrms
    del target_srcmask
    del target_objectmask
    del bkg_instance
    del target_bkg_tmp
    del mbias
    del mdark
    del mflat
    del sex_catalog
    del calib_catalog
    del filtered_catalog
    gc.collect()   

    return status
#%%
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
from tqdm import tqdm

def chunk_list(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def process_batch(batch_images, batch_index, max_workers=16):
    print(f"\nStarting batch {batch_index+1} with {len(batch_images)} images...")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(imgprocess, img) for img in batch_images]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_index+1}"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"[ERROR in batch {batch_index+1}] {e}")
    
    # Clean up memory between batches
    gc.collect()
    return results

# ? Main loop over batches
batch_size = 16
all_results = []

for batch_index, batch in enumerate(chunk_list(target_imglist, batch_size)):
    batch_results = process_batch(batch, batch_index, max_workers=16)
    all_results.extend(batch_results)

#%%
### CONFIGURATION FOR STACKED IMAGE PROCESSING

stack_aperturephotometry_kwargs = dict(
    sex_params = None,
    detection_sigma = 5,
    aperture_diameter_arcsec = [7,12,15],
    aperture_diameter_seeing = [3.5, 4.5], # If given, use seeing to calculate aperture size
    saturation_level = 35000,
    kron_factor = 2.5,
    save = save,
    verbose = verbose,
    visualize = visualize,
    save_fig = save_fig,
)

stack_photometriccalibration_kwargs = dict(
    catalog_type = 'APASS',
    max_distance_second = 5,
    calculate_color_terms = True,
    calculate_mag_terms = True,
    mag_lower = None,
    mag_upper = None,
    snr_lower = 20,
    snr_upper = 500,
    classstar_lower = 0.5,
    elongation_upper = 3,
    elongation_sigma = 5,
    fwhm_lower = 1,
    fwhm_upper = 15,
    fwhm_sigma = 5,
    flag_upper = 1,
    maskflag_upper = 1,
    inner_fraction = 0.8, # Fraction of the images
    isolation_radius = 15.0,
    magnitude_key = 'MAG_AUTO',
    flux_key = 'FLUX_APER_4',
    fluxerr_key = 'FLUXERR_APER_4',
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
#%%
databrowser = DataBrowser('scidata')
databrowser.observatory = 'RASA36'
databrowser.objname = 'NGC1566'
databrowser.telkey = 'RASA36_KL4040_HIGH_1x1'
# Stack the images
imginfo_all = databrowser.search(pattern='Calib*60.fits', return_type='imginfo')
imginfo_groups = imginfo_all.group_by(['filter', 'telescop']).groups
 #%%
stack_imglist = []
stack_bkgrmslist = []
failed_imglist = []
for imginfo_group in imginfo_groups:
    imginfo_group = helper.group_table(imginfo_group, 'mjd', 0.1)
    imginfo_subgroups = imginfo_group.group_by('group').groups
    telinfo = helper.estimate_telinfo(imginfo_subgroups[0][0]['file'])
    for imginfo_subgroup in imginfo_subgroups:
        target_imglist = [ScienceImage(path=row['file'], telinfo=telinfo, load=True) for row in imginfo_subgroup]
        target_imglist = Stacking.select_quality_images(target_imglist, seeing_limit = 8, depth_limit = 15, ellipticity_limit = 1.0, max_numbers = len(target_imglist), visualize = visualize)
        if len(target_imglist) == 0:
            print(f"[WARNING] No images found. skipping stacking.")
            continue
        target_bkglist = [Background(path = target_img.savepath.bkgpath, load=True) for target_img in target_imglist]
        target_bkgrmslist = [Errormap(path = target_img.savepath.bkgrmspath, emaptype = 'bkgrms', load=True) for target_img in target_imglist]
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
#%%
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
#%%
stack_imglist = databrowser.search(pattern='Calib*60.com.fits', return_type='science')
stack_bkgrmslist = [Errormap(path=stack_img.savepath.bkgrmspath, emaptype='bkgrms', load=True) for stack_img in stack_imglist]
telinfo = helper.estimate_telinfo(stack_imglist[0].path)
#%%
arglist = [(stack_img.path, stack_bkgrms.path, helper.estimate_telinfo(stack_img.path)) for stack_img, stack_bkgrms in zip(stack_imglist, stack_bkgrmslist)]
with ProcessPoolExecutor(max_workers=10) as executor:
    failed_stacked_path = []
    futures = [executor.submit(stackprocess, *args) for args in arglist]
    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = future.result()
        except Exception as e:
            print(f"[ERROR] {e}")
# %%
import time
print('Current time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
# %%
from ezphot.dataobjects import LightCurve
from ezphot.catalog import Catalog
from ezphot.catalog import CatalogDataset
# %%
catalogdataset = CatalogDataset()
catalogdataset.search_catalogs('NGC1566', 'Calib*60.com.fits.cat', folder = '/home/hhchoi1022/data/scidata/RASA36/RASA36_KL4040_HIGH_1x1')
catalogdataset.select_sources(ra = 64.9725, dec= -54.948081)
# %%
lc = LightCurve(catalogdataset)
lc.merge_catalogs()
#%%
for catalog in lc.source_catalogs.catalogs:
    catalog.load_target_img()
# %%
lc.plt_params.figure_figsize = (10, 6)
lc.plt_params.xlim= [59500, 59700]
lc.plt_params.ylim = [20, 11]
lc.plot(ra = 64.9729, dec= -54.948381, matching_radius_arcsec = 10,)
# %% Reference image
databrowser.observatory = 'RASA36'
databrowser.objname = 'NGC1566'
databrowser.telkey = 'RASA36_KL4040_HIGH_1x1'
reference_singleimglist = databrowser.search(pattern='Calib*60.fits', return_type='science')


# %%
reference_singleimglist_selected = Stacking.select_quality_images(
    reference_singleimglist,
    max_obsdate = '2021-10-20',
    max_numbers = 60,
    visualize = True,
    weight_ellipticity = 1,
    weight_seeing= 1,
    weight_depth = 1,
    seeing_limit = 4,
    depth_limit = 18.0
    
)
# %%
reference_singlebkglist_selected = [Background(path = img.savepath.bkgpath, load = True) for img in reference_singleimglist_selected]
reference_singlebkgrmslist_selected = [Errormap(path = img.savepath.bkgrmspath, emaptype = 'bkgrms', load = True) for img in reference_singleimglist_selected] 
reference_img, reference_bkgrms = Stacking.stack_multiprocess(
    target_imglist = reference_singleimglist_selected,
    target_bkglist = reference_singlebkglist_selected,
    target_bkgrmslist = reference_singlebkgrmslist_selected,
    **stacking_kwargs
)
# %%
telinfo = helper.estimate_telinfo(reference_img.path)
reference_img, reference_bkgrms, _, _ = stackprocess(reference_img.path, reference_bkgrms.path, telinfo)
# %%
reference_img = reference_img.to_referenceimage()
#%%
reference_img.register()
# %%
DIA_kwargs = dict(
    detection_sigma = 5,
    target_transient_number = 5,
    reject_variable_sources = True,
    negative_detection = True,
    reverse_subtraction = False,
    save = True,
    verbose = False,
    visualize = False, #False
    show_transient_numbers = 10)

databrowser = DataBrowser('scidata')
databrowser.observatory = 'RASA36'
databrowser.telkey = 'RASA36_KL4040_HIGH_1x1'
databrowser.objname = 'NGC1566'
stacked_imglist = databrowser.search(pattern='Calib*60.com.fits', return_type='science')
from ezphot.methods import TIPSubtraction
DIA = TIPSubtraction()
#%%
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

failed_images = []
successful_results = []

def subtract_process(target_img):
    try:
        reference_img = DIA.get_referenceframe_from_image(target_img)[0]
        result = DIA.find_transients(
            target_img=target_img, 
            reference_imglist=[reference_img],
            target_bkg=None,
            **DIA_kwargs
        )
        del reference_img
        del target_img
        return result
    except Exception as e:
        target_img.data = None  # Optional: clear large data
        target_img._error = str(e)  # Store the error message if needed
        return target_img  # Return failed image as indicator

#%%
# Run with multiprocessing
c = []
with ProcessPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(subtract_process, img) for img in stacked_imglist]
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        if hasattr(result, "_error"):  # failed image
            failed_images.append(result)
        else:  # successful
            c.append(result)
# %%
from ezphot.dataobjects import LightCurve
from ezphot.catalog import Catalog
from ezphot.catalog import CatalogDataset
catalogdataset = CatalogDataset()
catalogdataset.search_catalogs('NGC1566', 'sub*transient', folder = '/home/hhchoi1022/data/scidata/RASA36/RASA36_KL4040_HIGH_1x1')
catalogdataset.select_sources(ra = 64.9725, dec= -54.948081)
# %%
lc = LightCurve(catalogdataset)
lc.merge_catalogs()
#%%
# %%
lc.plt_params.figure_figsize = (10, 6)
lc.plt_params.xlim= [59500, 59700]
lc.plt_params.ylim = [20, 11]
lc.plot(ra = 64.9729, dec= -54.948381, matching_radius_arcsec = 10,)
# %%
