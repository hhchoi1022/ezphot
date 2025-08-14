#%%
from ezphot.imageojbects import *
from ezphot.catalog import *
from ezphot.methods import *
from ezphot.routine import *
from ezphot.helper import *
import multiprocessing
from tqdm import tqdm
import numpy as np
import gc
#%% # Initialize the classes
Preprocessor = Preprocess()
PlateSolver = TIPPlateSolve()
MaskGenerator = TIPMasking()
BkgGenerator = BackgroundEstimator()
ErrormapGenerator = TIPErrormap()
AperturePhotometry = AperturePhotometry()
PSFPhotometry = TIPPSFPhotometry()
PhotometricCalibration = TIPPhotometricCalibration()
Stacking = Stack()

print(Preprocessor)
print(PlateSolver)
print(MaskGenerator)
print(BkgGenerator)
print(ErrormapGenerator)
print(AperturePhotometry)
print(PSFPhotometry)
print(Stacking)
#%%
target_path = ''
target_img = ScienceImage(path = target_path, telinfo = telinfo, load = True)
#%% MASKING
target_srcmask = MaskGenerator.mask_sources(
    target_img,
    sigma = 5,
    mask_radius_factor = 3,
    saturation_level = 60000,
    save = False,
    verbose = True,
    visualize = True,
    save_fig = False
)
target_srcmask = MaskGenerator.mask_circle(
    target_img,
    target_mask = target_srcmask,
    x_position = 245.8960980,
    y_position = -26.5227669,
    radius_arcsec = 350,
    unit = 'deg',
    save = True,
    verbose = True,
    visualize = True,
    save_fig = False,
)

target_objectmask = MaskGenerator.mask_circle(
    target_img,
    target_mask = None,
    x_position = 245.8960980,
    y_position = -26.5227669,
    radius_arcsec = 600,
    unit = 'deg',
    save = True,
    verbose = True,
    visualize = True,
    save_fig = False,
)

#%% BACKGROUND GENERATION
target_bkg, _ = BkgGenerator.estimate_with_sep(
    target_img = target_img,
    target_mask = target_srcmask,
    box_size = 256,
    filter_size = 3,
    n_iterations = 0,
    mask_sigma = 3.0,
    mask_radius_factor = 3,
    mask_saturation_level = 60000,
    save = True,
    verbose = True,
    visualize = True,
    save_fig = False,
)

#%% BACKGROUND ERROR MAP GENERATION
mbias_path = Preprocessor.get_masterframe_from_image(
    target_img = target_img,
    imagetyp = 'BIAS',
    max_days = 15)[0]['file']
mdark_path = Preprocessor.get_masterframe_from_image(
    target_img = target_img,
    imagetyp = 'DARK',
    max_days = 15)[0]['file']
mflat_path = Preprocessor.get_masterframe_from_image(
    target_img = target_img,
    imagetyp = 'FLAT',
    max_days = 30)[0]['file']

mbias = MasterImage(path=mbias_path, telinfo = telinfo, load=True)
mdark = MasterImage(path=mdark_path, telinfo = telinfo, load=True)
mflat = MasterImage(path=mflat_path, telinfo = telinfo, load=True)

target_bkgrms = ErrormapGenerator.calculate_bkgrms_from_propagation(
    target_bkg = target_bkg,
    mbias_img = mbias,
    mdark_img = mdark,
    mflat_img = mflat,
    mflaterr_img = None,
    ncombine = None,
    save = True,
    verbose = True,
    visualize = True,
    save_fig = False,
)

#%% APERTURE PHOTOMETRY
_, sex_catalog = AperturePhotometry.sex_photometry(
    target_img,
    target_bkg,
    target_bkgrms,
    target_mask = target_objectmask,
    sex_params = None,
    detection_sigma = 5,
    aperture_diameter_arcse = 5.0,
    saturation_level = 60000,
    kron_factor = 2.5,
    save = True,
    verbose = True,
    visualize = True,
    save_fig = False,
)
#%% PHOTOMETRIC CALIBRATION
target_img, calibrated_catalog = PhotometricCalibration.photometric_calibration(
    target_img = target_img,
    target_catalog = sex_catalog,
    catalog_type = 'GAIAXP',
    max_distance_second = 1.0,
    calculate_color_terms = True,
    calculate_mag_terms = True,
    
    mag_lower = None,
    mag_upper = None,
    snr_lower = 10,
    snr_upper = 100,
    classstar_lower = 0.8,
    elongation_upper = 30,
    elongation_sigma = 5,
    fwhm_lower = 2,
    fwhm_upper = 15,
    fwhm_sigma = 3,
    flag_upper = 1,
    inner_fraction = 0.7,
    isolation_radius = 5.0,
    
    save = True,
    verbose = True,
    visualize = True,
    save_fig = False,
    save_starcat = False,
    save_refcat = False,
)

#%% PSF MODELING
epsf_model_dict = PSFPhotometry.build_epsf_model_psfex(
    target_img= target_img,
    target_bkg= target_bkg,
    target_bkgrms= target_bkgrms,
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


#%% PSF PHOTOMETRY
psf_catalog = PSFPhotometry.psf_photometry(
    target_img = target_img,
    target_bkg = target_bkg,
    target_bkgrms = target_bkgrms,
    epsf_model_dict = epsf_model_dict[(0,0)],
    sources = None,
    target_mask = None,
    # Detection parameters
    detection_sigma = 5.0,
    minarea_pixels = 5,
    deblend_nlevels = 32,
    deblend_contrast = 0.003,

    # PSF Photometry parameters
    fwhm_estimate_pixel = 5.0,
    n_iterations = 1,

    visualize = True,
    verbose = True,
    save = True,
    apply_aperture_correction = True,
)


#%% PHOTOMETRIC CALIBRATION FOR PSF CATALOG
calibrated_psf_catalog, filtered_psf_catalog = PhotometricCalibration.photometric_calibration(
    target_img = target_img,
    target_catalog = psf_catalog,
    catalog_type = 'GAIAXP',
    max_distance_second = 1.0,
    calculate_color_terms = True,
    calculate_mag_terms = True,
    
    mag_lower = None,
    mag_upper = None,
    snr_lower = 30,
    snr_upper = 200,
    classstar_lower = 0.8,
    elongation_upper = 30,
    elongation_sigma = 5,
    fwhm_lower = 2,
    fwhm_sigma = 3,
    flag_upper = 1,
    inner_fraction = 0.9,
    isolation_radius = 30.0,
    
    save = True,
    verbose = True,
    visualize = True,
    save_fig = False,
    save_starcat = False,
    save_refcat = False,
    mag_key = 'MAG_PSF',
)


# %%
