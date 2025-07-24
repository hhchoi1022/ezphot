#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tippy.image import *
from tippy.catalog import *
from tippy.photometry import *
from tippy.routine import *
from tippy.helper import *
from tippy.utils import *
import multiprocessing
from tqdm import tqdm
import numpy as np
import gc


#%% LOAD DATA
helper = Helper()
sdtdata = SDTData()
# Prepare
target_name = 'NGC6121'
targetdata = sdtdata.show_scidestdata(targetname=target_name, pattern = '*.fits')
targetdata.pop('tmp')
def get_imginfo_process(key):
    from tippy.image import ScienceImage
    helper = Helper()
    imginfo = helper.get_imginfo(targetdata[key])
    calib_indices = np.isin(imginfo['imgtype'], ['BIAS', 'ZERO', 'DARK', 'FLAT'])
    objframes_info = imginfo[~calib_indices]
    return key, objframes_info
with multiprocessing.Pool(processes=8) as pool:
    results = list(tqdm(pool.imap_unordered(get_imginfo_process, targetdata.keys()), total=len(targetdata)))

imginfo_all = dict()
for key, imginfo in results:
    imginfo_selected_idx = (imginfo['object'] == target_name) & (imginfo['exptime'] == '100.0')
    imginfo_all[key] = imginfo[imginfo_selected_idx]

for key, imginfo in imginfo_all.items():
    print(f"{key} with {len(imginfo)} object frames")


#%% Visualize the images
imginfo_filter = imginfo_all['m425']
target_imglist = [ScienceImage(path = imginfo['file'], telinfo = helper.estimate_telinfo(imginfo['file']), load = True) for imginfo in imginfo_filter]
telinfo = helper.estimate_telinfo(target_imglist[0].path)

for target_img in target_imglist:
    target_img.show(downsample= 6)


#%% # Initialize the classes
Preprocessor = TIPPreprocess()
PlateSolver = TIPPlateSolve()
MaskGenerator = TIPMasking()
BkgGenerator = TIPBackground()
ErrormapGenerator = TIPErrormap()
AperturePhotometry = TIPAperturePhotometry()
PSFPhotometry = TIPPSFPhotometry()
PhotometricCalibration = TIPPhotometricCalibration()
Stacking = TIPStacking()

print(Preprocessor)
print(PlateSolver)
print(MaskGenerator)
print(BkgGenerator)
print(ErrormapGenerator)
print(AperturePhotometry)
print(PSFPhotometry)
print(Stacking)
#%% Prepare the master frames & configurations

# Set master frames
init_file = str(list(imginfo_all.values())[0][0]['file'])
telinfo = helper.estimate_telinfo(init_file)
init_img = ScienceImage(path = init_file, telinfo = telinfo, load = False)

mbias_path = Preprocessor.get_masterframe_from_image(
    target_img = init_img,
    imagetyp = 'BIAS',
    max_days = 15)[0]['file']
mdark_path = Preprocessor.get_masterframe_from_image(
    target_img = init_img,
    imagetyp = 'DARK',
    max_days = 15)[0]['file']
mbias = MasterImage(path=mbias_path, telinfo = telinfo, load=True)
mdark = MasterImage(path=mdark_path, telinfo = telinfo, load=True)

# Sort the keys of imginfo_all
sorted_keys = sorted(imginfo_all.keys())
# Rebuild the dict in sorted order
imginfo_all = {key: imginfo_all[key] for key in sorted_keys}    

do_platesolve = False
platesolve_kwargs = dict(
    overwrite = True,
    verbose = True,
    )

do_generatemask = True
mask_kwargs = dict(
    sigma = 5,
    radius_factor = 2,
    saturation_level = 50000,
    save = False,
    verbose = True,
    visualize = False,
    save_fig = True
)

do_circularmask = True
circlemask_kwargs = dict(
    x_position = 245.8960980,
    y_position = -26.5227669,
    radius_arcsec = 350,
    unit = 'deg',
    save = True,
    verbose = True,
    visualize = False,
    save_fig = True
)

do_objectmask = True

objectmask_kwargs = dict(
    x_position = 245.8960980,
    y_position = -26.5227669,
    radius_arcsec = 600,
    unit = 'deg',
    save = False,
    verbose = True,
    visualize = False,
    save_fig = True
)

do_generatebkg = True
bkg_kwargs = dict(
    box_size = 256,
    filter_size = 3,
    n_iterations = 0,
    mask_sigma = 5,
    mask_radius_factor = 3,
    mask_saturation_level = 60000,
    save = True,
    verbose = True,
    visualize = False,
    save_fig = True
)

do_generateerrormap = True
errormap_from_propagation = True,
errormap_kwargs = dict(
    save = True,
    verbose = True,
    visualize = False,
    save_fig = True
)

do_aperturephotometry = True
aperturephotometry_kwargs = dict(
    sex_params = None,
    detection_sigma = 5,
    aperture_diameter_arcsec = [6.0, 9.0, 12.0],
    saturation_level = 60000,
    kron_factor = 2.5,
    save = True,
    verbose = True,
    visualize = True,
    save_fig = True,
)

do_photometric_calibration = True
photometriccalibration_kwargs = dict(
    catalog_type = 'GAIAXP',
    max_distance_second = 1.0,
    calculate_color_terms = True,
    calculate_mag_terms = True,

    mag_lower = None,
    mag_upper = None,
    snr_lower = 10,
    snr_upper = 500,
    classstar_lower = 0.7,
    elongation_upper = 1.5,
    elongation_sigma = 5,
    fwhm_lower = 2,
    fwhm_upper = 15,
    fwhm_sigma = 5,
    flag_upper = 1,
    maskflag_upper = 1,
    inner_fraction = 0.8, # Fraction of the images
    isolation_radius = 20.0,

    save = True,
    verbose = True,
    visualize = True,
    save_fig = True,
    save_starcat = False,
    save_refcat = True,
    mag_key = 'MAG_APER_1'
)


#%% Define the image processing function

def imgprocess(target_path, telinfo):
    target_img = ScienceImage(path = target_path, telinfo = telinfo, load = True)
    mflat_path = Preprocessor.get_masterframe_from_image(target_img, imagetyp = 'FLAT', max_days = 60)[0]
    mflat = MasterImage(path = mflat_path['file'], telinfo = telinfo, load = True)

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
    target_centermask = None
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
        except Exception as e:
            status['errormap'] = e

    if do_aperturephotometry:
        try:
            # Perform aperture photometry
            target_img, sex_catalog = AperturePhotometry.sex_photometry(
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
            target_img, calib_catalog = PhotometricCalibration.photometric_calibration(
                target_img = target_img,
                target_catalog = sex_catalog,
                **photometriccalibration_kwargs
            )
            status['photometric_calibration'] = True
        except Exception as e:
            status['photometric_calibration'] = e

    del target_srcmask, target_objectmask
    target_img.data = None
    gc.collect()
    return target_img, target_bkg, target_bkgrms, status


#%% Process the images (Masking, Background, Errormap, Aperture Photometry, Photometric Calibration)

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

arglist = [(str(row['file']), telinfo) for imginfo_list in imginfo_all.values() for row in imginfo_list]
#arglist = [(str(row['file']), telinfo) for row in imginfo_all['m600']]

results_by_filter = {
    filtername: {
        'target_img': [],
        'target_bkg': [],
        'target_bkgrms': [],
        'status': []
    } for filtername in imginfo_all.keys()
}

with ProcessPoolExecutor(max_workers=15) as executor:
    futures = [executor.submit(imgprocess, *args) for args in arglist]

    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = future.result()
            if result is not None:
                target_img, target_bkg, target_bkgrms, status = result
                filtername = target_img.filter
                results_by_filter[filtername]['target_img'].append(target_img)
                results_by_filter[filtername]['target_bkg'].append(target_bkg)
                results_by_filter[filtername]['target_bkgrms'].append(target_bkgrms)
                results_by_filter[filtername]['status'].append(status)
            else:
                print(f"[WARNING] Skipped a result due to None")
        except Exception as e:
            print(f"[ERROR] {e}")

#%% Check the results
for key, value in results_by_filter.items():
    print(f"{key} with {len(value['target_img'])} images")
    
    for status in value['status']:
        print('Platesolve:', status.get('platesolve', 'N/A'))
        print('Mask:', status.get('mask', 'N/A'))
        print('Circular Mask:', status.get('circular_mask', 'N/A'))
        print('Object Mask:', status.get('object_mask', 'N/A'))
        print('Background:', status.get('background', 'N/A'))
        print('Errormap:', status.get('errormap', 'N/A'))
        print('Aperture Photometry:', status.get('aperture_photometry', 'N/A'))
        print('Photometric Calibration:', status.get('photometric_calibration', 'N/A'))
    print("\n")

#%% Prepare the stacking parameters

do_stacking = True
stacking_kwargs = dict(
    combine_type = 'median',
    n_proc = 8,
    # Clip parameters
    clip_type = None,
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
    verbose = True,
    save = True
    )

#%% Stack the images

targetdata = sdtdata.show_scidestdata(targetname=target_name, pattern = 'calib*.fits')
with multiprocessing.Pool(processes=8) as pool:
    results = list(tqdm(pool.imap_unordered(get_imginfo_process, targetdata.keys()), total=len(targetdata)))
#%%
imginfo_all = dict()
for key, imginfo in results:
    imginfo_selected_idx = (imginfo['object'] == target_name) & (imginfo['exptime'] == '100.0')
    imginfo_all[key] = imginfo[imginfo_selected_idx]

#%%
stack_imglist = []
stack_bkgrmslist = []

for filter, imginfo_filter in imginfo_all.items():
    target_pathlist = imginfo_filter['file']
    target_imglist = [ScienceImage(path=row['file'], telinfo=telinfo, load=True) for row in target_pathlist]
    target_bkglist = [Background(target_img.savepath.bkgpath, load=True) for row in target_imglist]
    target_bkgrmslist = [Errormap(path=target_img.savepath.bkgrmspath, emaptype='bkgrms', load=True) for target_img in target_imglist]
    if len(target_imglist) == 0:
        print(f"[WARNING] No images found for {key}, skipping stacking.")
        continue

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
        
    stack_imglist.append(stack_img)
    stack_bkgrmslist.append(stack_bkgrms)

#%%
target_imgdict = dict()
for filter_, imginfo in imginfo_all.items():
    files = [row['file'] for row in imginfo]
    images = [ScienceImage(path=file, telinfo=telinfo, load=True) for file in files]
    target_imgdict[filter_] = images    
#%%

for filter_, images in target_imgdict.items():
    print(f"{filter_} with {len(images)} images")
    for img in images:
        print('Depth: {}, Seeing: {}, ZP: {}, SKYSIG: {}'.format(img.header['UL5_APER'], img.header['SEEING'], img.header['ZP_APER'], img.info.SKYSIG))
        print('"Depth: {}, Seeing: {}, ZP: {}, SKYSIG: {}'.format(img.header['UL5_4'], img.header['SEEING'], img.header['ZP_4'], img.header['SKYSIG']))
    print("\n")
#%%
do_psfphotometry = True

psfmodel_kwargs = dict(
    detection_sigma = 5.0,
    minarea_pixels = 5,
    fwhm_estimate_pixel = 4.0,
    saturation_level = 40000,
    psf_size = 25,
    num_grids = 1,
    oversampling = 1,
    eccentricity_upper = 0.4,
    verbose = True,
    visualize = True)

psfphotometry_kwargs = dict(
    detection_sigma = 5,
    minarea_pixels = 8,
    deblend_nlevels = 32,
    deblend_contrast = 0.005,
    fwhm_estimate_pixel = 5.0,
    n_iterations = 2,
    visualize = True,
    verbose = True,
    save = True,
    apply_aperture_correction = True)
    
do_photometric_calibration = True
photometriccalibration_kwargs = dict(
    catalog_type = 'GAIAXP',
    max_distance_second = 1.0,
    calculate_color_terms = True,
    calculate_mag_terms = True,

    mag_lower = None,
    mag_upper = None,
    snr_lower = 50,
    snr_upper = 1000,
    classstar_lower = 0.7,
    elongation_upper = 1.5,
    elongation_sigma = 5,
    fwhm_lower = 2,
    fwhm_sigma = 3,
    flag_upper = 1,
    inner_fraction = 0.9,
    isolation_radius = 30.0,

    save = True,
    verbose = True,
    visualize = True,
    save_fig = True,
    save_starcat = False,
    save_refcat = True,
    mag_key = 'MAG_PSF_CORR'
)
#%%
#for stack_img, stack_bkgrms in zip(stack_imglist, stack_bkgrmslist):
def psfphot_process(stacked_imgpath, stacked_bkgrmspath):
    stack_img = ScienceImage(path = stacked_imgpath, telinfo = telinfo, load = True)
    stack_bkgrms = Errormap(path = stacked_bkgrmspath, emaptype = 'bkgrms', load = True)
    
    status = dict()
    status['image'] = stacked_imgpath
    status['psfmodel'] = None
    if do_psfphotometry:
        try:
            epsf_model_dict = PSFPhotometry.build_epsf_model_psfex(
                target_img= stack_img,
                target_bkg= None,
                target_bkgrms= stack_bkgrms,
                **psfmodel_kwargs)
        except Exception as e:
            status['psfmodel'] = e
            print(f"[ERROR] PSF model generation failed for {stack_img.path}: {e}")
        
        try:
            psf_catalog = PSFPhotometry.psf_photometry(
                target_img = stack_img,
                target_bkg = None,
                target_bkgrms = stack_bkgrms,
                epsf_model_dict = epsf_model_dict,
                sources = None,
                target_mask = None,
                **psfphotometry_kwargs
            )
        except Exception as e:
            status['psf_photometry'] = e
            print(f"[ERROR] PSF photometry failed for {stack_img.path}: {e}")
        
    if do_photometric_calibration and do_psfphotometry:
        try:
            stack_img, filtered_psf_catalog = PhotometricCalibration.photometric_calibration(
                target_img = stack_img,
                target_catalog = psf_catalog,
                **photometriccalibration_kwargs
            )
        except Exception as e:
            status['photometric_calibration'] = e
            print(f"[ERROR] Photometric calibration failed for {stack_img.path}: {e}")
    
    return stack_img, status

# ## Source masking
#%%
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

#arglist = [(str(row['file']), telinfo) for imginfo_list in imginfo_all.values() for row in imginfo_list]
arglist = [(stacked_img.path, stack_bkgrms.path) for stacked_img, stacked_bkgrms in zip(stack_imglist, stack_bkgrmslist)]

results = []
with ProcessPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(psfphot_process, *args) for args in arglist]

    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = future.result()
            if result is not None:
                results.append(result)
            else:
                print(f"[WARNING] Skipped a result due to None")
        except Exception as e:
            print(f"[ERROR] {e}")
# %%

stacked_imglist = list(init_img.savepath.savedir.parent.parent.rglob('calib*100.com.fits'))
stacked_bkgrmslist = list(init_img.savepath.savedir.parent.parent.rglob('calib*100.com.fits.bkgrms'))
stacked_imglist.sort()
stacked_bkgrmslist.sort()
stacked_img = stacked_imglist[0]
stacked_bkgrms = stacked_bkgrmslist[0]
#%%
for stacked_img, stacked_bkgrms in zip(stacked_imglist, stacked_bkgrmslist):
    psf_img, status = psfphot_process(stacked_img, stacked_bkgrms)

# %%
