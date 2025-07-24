

#%%
from tippy.photometry import *
from tippy.imageojbects import *
from tippy.helper import Helper
from tippy.utils import SDTData
from tqdm import tqdm
import gc
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
#%%
matplotlib.use('Agg') 

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

print(Preprocessor)
print(PlateSolver)
print(MaskGenerator)
print(BkgGenerator)
print(ErrormapGenerator)
print(AperturePhotometry)
print(PSFPhotometry)
print(Stacking)
#%%
### CONFIOGURATION FOR SINGLE IMAGE PROCESSING

do_platesolve = False
platesolve_kwargs = dict(
    overwrite = True,
    verbose = True,
    )

do_generatemask = True
mask_kwargs = dict(
    sigma = 5,
    radius_factor = 1.5,
    saturation_level = 50000,
    save = False,
    verbose = True,
    visualize = False,
    save_fig = True
)

do_circularmask = False
circlemask_kwargs = dict(
    x_position = 245.8960980,
    y_position = -26.5227669,
    radius_arcsec = 350,
    unit = 'deg',
    save = False,
    verbose = True,
    visualize = False,
    save_fig = True
)

do_objectmask = False
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
    box_size = 64,
    filter_size = 3,
    n_iterations = 0,
    mask_sigma = 3,
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
    save = False,
    verbose = True,
    visualize = False,
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
    snr_lower = 15,
    snr_upper = 500,
    classstar_lower = 0.7,
    elongation_upper = 3,
    elongation_sigma = 5,
    fwhm_lower = 2,
    fwhm_upper = 15,
    fwhm_sigma = 5,
    flag_upper = 1,
    maskflag_upper = 1,
    inner_fraction = 0.8, # Fraction of the images
    isolation_radius = 20.0,
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
    save = True,
    verbose = True,
    visualize = False,
    save_fig = True,
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
    verbose = True,
    save = True
    )

### CONFIGURATION FOR STACKED IMAGE PROCESSING

stack_aperturephotometry_kwargs = dict(
    sex_params = None,
    detection_sigma = 5,
    aperture_diameter_arcsec = [6.0, 9.0, 12.0],
    saturation_level = 60000,
    kron_factor = 2.5,
    save = False,
    verbose = True,
    visualize = True,
    save_fig = True,
)

stack_photometriccalibration_kwargs = dict(
    catalog_type = 'GAIAXP',
    max_distance_second = 1.0,
    calculate_color_terms = True,
    calculate_mag_terms = True,

    mag_lower = None,
    mag_upper = None,
    snr_lower = 15,
    snr_upper = 500,
    classstar_lower = 0.7,
    elongation_upper = 3,
    elongation_sigma = 5,
    fwhm_lower = 2,
    fwhm_upper = 15,
    fwhm_sigma = 5,
    flag_upper = 1,
    maskflag_upper = 1,
    inner_fraction = 0.8, # Fraction of the images
    isolation_radius = 20.0,
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
    save = True,
    verbose = True,
    visualize = False,
    save_fig = True,
    save_starcat = False,
    save_refcat = True,
)

#%% Query object frames
tile_id = 'T01318'
helper = Helper()
data_qurier = SDTData()
print(f"Syncing data for target: {tile_id}")
data_qurier.sync_scidata(targetname = tile_id)
# Load the data
target_images = data_qurier.show_scidestdata(
    tile_id,
    False,
    'filter',
    'calib*100.fits'
)
target_imglist_all = [target_imgpath for target_imglist in target_images.values() for target_imgpath in target_imglist]
imginfo_all = data_qurier.get_imginfo(target_imglist_all)
imginfo_groups = imginfo_all.group_by(['filter', 'telescop']).groups
print(f'Tile ID: {tile_id}, Number of images: {len(imginfo_all)}, Number of groups: {len(imginfo_groups)}')
#%%

# Set telescope information
target_path = imginfo_groups[0]['file'][0]
telinfo = helper.estimate_telinfo(target_path)
#%%
# Define the image processing function
# 76 images -> 9min 41s
def imgprocess(target_path, telinfo):
    target_img = ScienceImage(path = target_path, telinfo = telinfo, load = True)
    mbias_path = Preprocessor.get_masterframe_from_image(target_img, imagetyp = 'BIAS', max_days = 60)[0]
    mdark_path = Preprocessor.get_masterframe_from_image(target_img, imagetyp = 'DARK', max_days = 60)[0]
    mflat_path = Preprocessor.get_masterframe_from_image(target_img, imagetyp = 'FLAT', max_days = 60)[0]
    mbias = MasterImage(path = mbias_path['file'], telinfo = telinfo, load = True)
    mdark = MasterImage(path = mdark_path['file'], telinfo = telinfo, load = True)
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
    del mbias, mdark, mflat
    target_img.data = None
    target_bkg.data = None
    target_bkgrms.data = None
    gc.collect()
    return target_img, target_bkg, target_bkgrms, calib_catalog, status

# Process the images (Masking, Background, Errormap, Aperture Photometry, Photometric Calibration)
imginfo_all.sort('filter')
arglist = [(str(row['file']), telinfo) for row in imginfo_all]
results_by_filter = {
    str(filtername): {
        'target_img': [],
        'target_bkg': [],
        'target_bkgrms': [],
        'target_catalog': [],
        'status': []
    } for filtername in set(imginfo_all['filter'])
}
with ProcessPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(imgprocess, *args) for args in arglist]

    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = future.result()
            if result is not None:
                target_img, target_bkg, target_bkgrms, target_catalog, status = result
                filtername = target_img.filter
                results_by_filter[filtername]['target_img'].append(target_img)
                results_by_filter[filtername]['target_bkg'].append(target_bkg)
                results_by_filter[filtername]['target_catalog'].append(status.get('calib_catalog', None))
                results_by_filter[filtername]['target_bkgrms'].append(target_bkgrms)
                results_by_filter[filtername]['status'].append(status)
            else:
                print(f"[WARNING] Skipped a result due to None")
        except Exception as e:
            print(f"[ERROR] {e}")
            
# Stack the images
imginfo_groups = imginfo_all.group_by(['filter', 'telescop']).groups
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
            print(f"[WARNING] No images foun, skipping stacking.")
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
        
#%%
from tippy.catalog import TIPCatalog
from tippy.catalog import TIPCatalogDataset
from tippy.dataobjects import PhotometricSpectrum
# %%
target_catalogs_filter = data_qurier.show_scidestdata(
    tile_id,
    False,
    'filter',
    'calib*com.fits.cat'
)
target_catalogpaths_all = [target_catalogpath for target_cataloglist in target_catalogs_filter.values() for target_catalogpath in target_cataloglist]
target_catalogs_all = [TIPCatalog(path=target_catalogpath, catalog_type = 'all', load=True) for target_catalogpath in target_catalogpaths_all]
for target_catalog in target_catalogs_all:
    target_path = target_catalog.find_corresponding_fits()
    target_img = ScienceImage(path=target_path, telinfo=telinfo, load=True)
    target_catalog.load_target_img(target_img)
    
# %%
dataset = TIPCatalogDataset(target_catalogs_all)
spectrum = PhotometricSpectrum(dataset)
# %%
spectrum.update_data(data_keys = ['MAGSKY_APER', 'MAGSKY_APER_1', 'MAGSKY_APER_2', 'MAGERR_APER', 'MAGERR_APER_1', 'MAGERR_APER_2'])
#%%
ra = 99.1083333
dec = -68.8058333
spectrum.plot(
    ra = ra,
    dec = dec,
    matching_radius_arcsec = 1.0,
    flux_key = 'MAGSKY_APER_1',
    fluxerr_key = 'MAGERR_APER_1',
    overplot_gaiaxp = True,
    overplot_ps1 = True,
    overplot_sdss = True,
)
# %%
spectrum.source_catalogs.catalogs[-5].show_source(ra, dec, matching_radius_arcsec = 4.5)

#%%
for catalog in spectrum.source_catalogs.catalogs:
    target_img = catalog.target_img
    print('Filter: ', target_img.filter, 'ZP_APER_1: ', target_img.header['ZP_APER_1'], 'telname: ', target_img.header['telescop'], 'Obsdate: ', target_img.obsdate)
# %%
