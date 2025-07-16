

#%%
from tippy.photometry import *
from tippy.image import *
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
DIA = TIPSubtraction()
sdt_data_querier = SDTData()
helper = Helper()

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
    detection_sigma = 1.5,
    aperture_diameter_arcsec = [5,7,10],
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
    detection_sigma = 1.5,
    aperture_diameter_arcsec = [5,7,10],
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

#%% Define the tile IDs for S250206dm
tile_ids_S250206dm = list([
 "T01161", "T01257", "T01469", "T01260", "T01468", "T01158",
    "T01465", "T01576", "T01064", "T01358", "T01359", "T01364", "T01065", "T01261",
    "T01466", "T01162", "T01066", "T01463", "T01577", "T00974", "T01063", "T01578",
    "T01067", "T01262", "T01470", "T01691", "T01255", "T01357", "T00975", "T01575",
    "T00973", "T01163", "T01157", "T01579", "T01690", "T01692", "T01156", "T01464",
    "T01689", "T00972", "T01693", "T01580", "T01462", "T01574", "T01068", "T00976",
    "T01062", "T01573", "T01254", "T01365", "T01694", "T01572", "T01355", "T01164",
    "T01471", "T01809", "T01356", "T01810", "T01263", "T01811", "T01061", "T01688",
    "T01687", "T01461", "T01155"
    ])
tile_ids_S250206dm = list(['T01570'])

#%% Query object frames

for tile_id in tile_ids_S250206dm:
    # Load the data
    sdt_data_querier.sync_scidata(targetname = tile_id)

    target_images = sdt_data_querier.show_scidestdata(
        tile_id,
        False,
        'filter',
        'calib*100.fits'
    )
    target_imglist_all = [target_imgpath for target_imglist in target_images.values() for target_imgpath in target_imglist]
    imginfo_all = sdt_data_querier.get_imginfo(target_imglist_all)
    imginfo_groups = imginfo_all.group_by(['filter', 'telescop']).groups
    print(f'Tile ID: {tile_id}, Number of images: {len(imginfo_all)}, Number of groups: {len(imginfo_groups)}')
    
    # Set telescope information
    target_path = imginfo_groups[0]['file'][0]
    telinfo = helper.estimate_telinfo(target_path)
    
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
    with ProcessPoolExecutor(max_workers=20) as executor:
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
            
#%% SUBTRACTION WITH SMSS HIPS2FITS IMAGE
tile_ids_S250206dm = list([
    "T01259", "T01159", "T01467", "T01361", "T01160", "T01360", "T01258", "T01362",
    "T01256", "T01363", "T01161", "T01257", "T01469", "T01260", "T01468", "T01158",
    "T01465", "T01576", "T01064", "T01358", "T01359", "T01364", "T01065", "T01261",
    "T01466", "T01162", "T01066", "T01463", "T01577", "T00974", "T01063", "T01578",
    "T01067", "T01262", "T01470", "T01691", "T01255", "T01357", "T00975", "T01575",
    "T00973", "T01163", "T01157", "T01579", "T01690", "T01692", "T01156", "T01464",
    "T01689", "T00972", "T01693", "T01580", "T01462", "T01574", "T01068", "T00976",
    "T01062", "T01573", "T01254", "T01365", "T01694", "T01572", "T01355", "T01164",
    "T01471", "T01809", "T01356", "T01810", "T01263", "T01811", "T01061", "T01688",
    "T01687", "T01461", "T01155"
    ])
tile_ids_S250206dm = list([ "T01570"])
#%%
unprocessed_tile_ids_S250206dm = list()
for tile_id in tile_ids_S250206dm:
    imginfo_all = sdt_data_querier.show_scidestdata(
        targetname = tile_id,
        show_only_numbers = False,
        key = 'filter',
        pattern = 'calib*100.com.fits'
    )
    try:
        for filter_ in ['g', 'r']:
            if filter_ not in imginfo_all.keys():
                print(f'No {filter_} filter images found for tile {tile_id}. Skipping...')
                continue
            
            imginfo = imginfo_all[filter_]
            telinfo = helper.estimate_telinfo(imginfo[0])
            target_imglist = [ScienceImage(img, telinfo = telinfo, load = True) for img in imginfo]
            target_img = Stacking.select_quality_images(target_imglist, max_numbers = 1)[0]
            reference_img, _ = DIA.get_referenceframe_from_image(target_img)
        
            for target_img in target_imglist:
                result = DIA.find_transients(target_img, 
                                            [reference_img],
                                            visualize = True,
                                            show_transient_numbers = 1,
                                            reject_variable_sources = True
                                            )
    except Exception as e:
        print(f"[ERROR] {e}")
        unprocessed_tile_ids_S250206dm.append(tile_id)
#%%
tile_ids_S250206dm = list([
    "T01259", "T01159", "T01467", "T01361", "T01160", "T01360", "T01258", "T01362",
    "T01256", "T01363", "T01161***", "T01257***", "T01469", "T01260", "T01468", "T01158",
    "T01465", "T01576", "T01064", "T01358", "T01359", "T01364", "T01065", "T01261",
    "T01466", "T01162", "T01066", "T01463", "T01577", "T00974", "T01063", "T01578",
    "T01067", "T01262", "T01470", "T01691", "T01255", "T01357", "T00975", "T01575",
    "T00973", "T01163", "T01157", "T01579", "T01690", "T01692", "T01156", "T01464",
    "T01689", "T00972", "T01693", "T01580", "T01462", "T01574", "T01068", "T00976",
    "T01062", "T01573", "T01254", "T01365", "T01694", "T01572", "T01355", "T01164",
    "T01471", "T01809", "T01356", "T01810", "T01263", "T01811", "T01061", "T01688",
    "T01687", "T01461", "T01155"
    ])
idx = 74
tile_id =  'T01357'   #tile_ids_S250206dm[idx]
print(f"Processing tile ID: {tile_id}")
file_dict = sdt_data_querier.show_scidestdata(
    targetname = tile_id,
    show_only_numbers = False,
    key = 'filter',
    pattern = '*calib*100.com.fits'
)
cat_dict = sdt_data_querier.show_scidestdata(
    targetname = tile_id,
    show_only_numbers = False,
    key = 'filter',
    pattern = '*.transient')
telinfo = helper.estimate_telinfo(file_dict['g'][0])

from tippy.catalog import TIPCatalog
from tippy.catalog import TIPCatalogDataset
all_filepaths = []
for filter_, filepaths in cat_dict.items():
    for filepath in filepaths:
        all_filepaths.append(filepath)
transientcatalog_list = [TIPCatalog(path = path, catalog_type = 'transient', load = True) for path in all_filepaths]
transientcatalog_dataset = TIPCatalogDataset(transientcatalog_list)
merged_transient_tbl, metadata = transientcatalog_dataset.merge_catalogs(max_distance_arcsec = 5)
real_transients = merged_transient_tbl
real_transients = merged_transient_tbl[merged_transient_tbl['n_detections'] > 1]
print(f"Number of real transients: {len(real_transients)}")
for i, catalog in enumerate(transientcatalog_list):
    print(f'Idx = {i}, N_detections: {catalog.nsources}, Depth: {catalog.target_img.depth}, Seeing: {catalog.target_img.seeing}, Filter: {catalog.target_img.filter}')
#%%
idx = 4
target_pathlist = [catalog.target_img.savepath.savedir / catalog.target_img.savepath.savepath.name.replace('sub_', '') for catalog in transientcatalog_list]
target_imglist = [ScienceImage(path = path, telinfo = telinfo, load = True) for path in target_pathlist]
target_img = target_imglist[idx]
reference_img = DIA.get_referenceframe_from_image(target_img)[0]
subtracted_img = transientcatalog_list[idx].target_img
print(target_img.obsdate, target_img.depth, target_img.seeing, target_img.filter)
# %%
# for transient in real_transients:
#     DIA.show_transient_positions(
#         science_img = target_img,
#         reference_img = reference_img,
#         subtracted_img = subtracted_img,
#         x = transient['ra'],
#         y = transient['dec'],
#         title = f"Transient at {transient['ra']:.6f}, {transient['dec']:.6f} with N_detections: {transient['n_detections']}",
#     )
# %%
ra, dec = 234.6855127, -68.794466 #cfkg for T01357 (Faint, but no-detection)
#ra, dec = 232.0696807, -67.8969778 #cflm for T01464 (Faint, but no-detection)
#ra, dec = 241.624307, -70.3271458 #cgem for T01158
#ra, dec = 229.4253039, -67.0045685 #chrk for T01571
#ra, dec = 228.3214072, -69.9576424 #dcnj for T01253
#ra, dec = 224.4553223,	-69.6647536 #dfxb for T01252
# ra, dec = 239.3351,	-68.6674 # cfqv or T01358 (Non-detection)
#ra,dec = 241.33278,	-65.70439 #cixq for T01806 ??????
# ra, dec = 238.38315,	-68.94083 #cfqk for T01358 (Non-detection)
# ra, dec = 241.8987,-68.98864 # cxxe for T01359 (Faint, but no detection)
# ra, dec = 233.22453, -67.62472 #cupm for T01462 (Negative detection)
# ra, dec = 240.6089648, -66.2845662 #cexu for T01688
# ra, dec = 233.5478514, -67.9823167 #cflk for T01462 (Clear detection (variable?))
# ra, dec = 227.763408, -67.0095581 #coau for T01570 (Non detection)
# ra, dec = 228.1596284, -66.9544428 #cgqi for T01570 (Non detection)
# ra, dec = 245.3187737, -67.4976052 #cong for T01466
# ra, dec = 250.75132, -68.94839 #clea for T01362
# ra, dec = 239.65318,	-67.034 #cimd for T01574
# ra, dec = 226.81001, -67.10678 #cnyl for T01570 (Non detection)
# ra, dec = 229.601605,-67.7274796 #cnjf for T01461
# ra, dec = 242.1756452, -67.5020564 #cgsg for T01465
# ra, dec = 240.4565808, -69.027386 #coso for T01359 (Non detection)
# ra, dec = 242.5642418,	-72.8343142 # chmn for T00887

#ra, dec = 233.322342, -68.007909 # Transient with T01462

for target_img, target_catalog in zip(target_imglist, transientcatalog_list):
    subtracted_img = target_catalog.target_img
    DIA.show_transient_positions(
        science_img = target_img,
        reference_img = reference_img,
        subtracted_img = subtracted_img,
        x = ra,
        y = dec,
        title = f"Transient at {ra:.6f}, {dec:.6f}",
    )
#%%
transientcatalog_dataset.search_sources(ra = ra, dec = dec, radius = 5)
# %% Forced aperture photometry for the transient
tile_id = 'T01462'
imginfo_all = sdt_data_querier.show_scidestdata(
    targetname = tile_id,
    show_only_numbers = False,
    key = 'filter',
    pattern = 'calib*100.com.fits'
)
#%%
all_filelist = []
for filter_, filelist in imginfo_all.items():
    for file in filelist:
        all_filelist.append(file)
#%%
target_imglist = [ScienceImage(path = target_path, telinfo = telinfo, load = True) for target_path in all_filelist]
target_bkgrmslist = [Errormap(path = target_img.savepath.bkgrmspath, emaptype = 'bkgrms', load = True) for target_img in target_imglist]
#%%
#%%
ra = 233.322342
dec = -68.007909
for target_img, target_bkgrms in zip(target_imglist, target_bkgrmslist):
    circ_catalog = AperturePhotometry.circular_photometry(
        target_img = target_img,
        x_arr = [ra],
        y_arr = [dec],
        aperture_diameter_arcsec = [5, 7, 10],
        target_bkg = None,
        unit = 'coord',
        target_bkgrms = target_bkgrms,
        save = True,
    )
    result_catalog = PhotometricCalibration.apply_zp(transient_catalog.target_img, circ_catalog, True)

# %%
