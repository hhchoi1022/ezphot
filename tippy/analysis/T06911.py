

#%%
from tippy.photometry import *
from tippy.image import *
from tippy.helper import Helper
from tippy.utils import SDTData
from tqdm import tqdm
import gc
#%% Query object frames
helper = Helper()
data_qurier = SDTData()
target_images = data_qurier.show_scidestdata(
    'T06911',
    False,
    pattern = '*100.fits')
target_imglist_all = [target_imgpath for target_imglist in target_images.values() for target_imgpath in target_imglist]
imginfo_all = data_qurier.get_imginfo(
    target_imglist_all,
)
target_path = target_imglist_all[0]
telinfo = helper.estimate_telinfo(target_path)
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
# %%
imginfo_groups = imginfo_all.group_by(['filter', 'telescop']).groups
#%% Set master frames

# %% Process single images to generate background, errormap

do_platesolve = False
platesolve_kwargs = dict(
    overwrite = True,
    verbose = True,
    )

do_generatemask = True
mask_kwargs = dict(
    sigma = 5,
    radius_factor = 3,
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
    aperture_diameter_arcsec = [5, 7, 10],
    saturation_level = 60000,
    kron_factor = 2.5,
    save = False,
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
    snr_lower = 15,
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
    visualize = True,
    save_fig = True,
    save_refcat = True,
)


#%% Define the image processing function
def imgprocess(target_path, telinfo):
    target_img = ScienceImage(path = target_path, telinfo = telinfo, load = True)
    mbias_path = Preprocessor.get_masterframe_from_image(target_img, imagetyp = 'BIAS', max_days = 1000)[0]
    mdark_path = Preprocessor.get_masterframe_from_image(target_img, imagetyp = 'DARK', max_days = 1000)[0]
    mflat_path = Preprocessor.get_masterframe_from_image(target_img, imagetyp = 'FLAT', max_days = 1000)[0]
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
        except Exception as e:
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

    calib_catalog = None
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
            target_img, calib_catalog, ref_catalog = PhotometricCalibration.photometric_calibration(
                target_img = target_img,
                target_catalog = sex_catalog,
                **photometriccalibration_kwargs
            )
            status['photometric_calibration'] = True
        except Exception as e:
            status['photometric_calibration'] = e

    # del target_srcmask, target_objectmask
    # target_img.data = None
    # if target_bkg is not None:
    #     target_bkg.data = None
    # if target_bkgrms is not None:
    #     target_bkgrms.data = None
    # if calib_catalog is not None:
    #     calib_catalog.data = None
    # gc.collect()
    return target_path, status

#%% Process the images (Masking, Background, Errormap, Aperture Photometry, Photometric Calibration)

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

imginfo_all.sort('filter')
arglist = [(str(row['file']), telinfo) for row in imginfo_all]# if row['filter'] == 'm875']
#%%
results = dict()
with ProcessPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(imgprocess, *args) for args in arglist]

    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = future.result()
            if result is not None:
                target_path, status = result
                results[target_path] = status
            else:
                print(f"[WARNING] Skipped a result due to None")
        except Exception as e:
            print(f"[ERROR] {e}")

#%% Check the results
for file_, status in results.items():
    print('Platesolve:', status.get('platesolve', 'N/A'))
    print('Mask:', status.get('mask', 'N/A'))
    print('Circular Mask:', status.get('circular_mask', 'N/A'))
    print('Object Mask:', status.get('object_mask', 'N/A'))
    print('Background:', status.get('background', 'N/A'))
    print('Errormap:', status.get('errormap', 'N/A'))
    print('Aperture Photometry:', status.get('aperture_photometry', 'N/A'))
    print('Photometric Calibration:', status.get('photometric_calibration', 'N/A'))
    print("\n")
#%%

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
#%%
ra = 222.2123987 
dec = -40.1460314
#%%
from tippy.catalog import TIPCatalog
filter_ = 'm850'
filepath_sample = imginfo_all[imginfo_all['filter'] == filter_]['file'][0]
target_img_sample = ScienceImage(filepath_sample, telinfo, load=False)
catalog_sample = TIPCatalog(target_img_sample.savepath.catalogpath, catalog_type = 'all')
catalog_sample.show_source(ra, dec)
#%% Stack the images
imginfo_groups = imginfo_all.group_by(['filter', 'telescop']).groups
#%%
stack_imglist = []
stack_bkgrmslist = []
for imginfo_group in imginfo_groups:
    imginfo_group = helper.group_table(imginfo_group, 'mjd', 0.2)
    imginfo_subgroups = imginfo_group.group_by('group').groups
    for imginfo_subgroup in imginfo_subgroups:
        target_imglist = [ScienceImage(path=row['file'], telinfo=telinfo, load=False) for row in imginfo_subgroup]
        target_bkglist = [Background(path = target_img.savepath.bkgpath, load=False) for target_img in target_imglist]
        target_bkgrmslist = [Errormap(path = target_img.savepath.bkgrmspath, emaptype = 'bkgrms', load=False) for target_img in target_imglist]

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

stack_aperturephotometry_kwargs = dict(
    sex_params = None,
    detection_sigma = 5,
    aperture_diameter_arcsec = [5.0, 7.0, 10.0],
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
    elongation_upper = 1.5,
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
    visualize = True,
    save_fig = True,
    save_starcat = False,
    save_refcat = True,
)
#%%
stacked_status = dict()
for stack_img, stack_bkgrms in zip(stack_imglist, stack_bkgrmslist):
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
        target_img, calib_catalog, ref_catalog = PhotometricCalibration.photometric_calibration(
            target_img = stack_img,
            target_catalog = sex_catalog,
            **stack_photometriccalibration_kwargs
        )
        stacked_status['photometric_calibration'] = True
    except Exception as e:
        stacked_status['photometric_calibration'] = e
# %%
for stack_img in stack_imglist:
    print(stack_img.header['SEEING'])
    print(stack_img.header['UL5_APER_1'])
# %%
target_images = data_qurier.show_scidestdata(
    'T06911',
    False,
    pattern = '*100.com.fits')
target_images = glob.glob('/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T06911/*/*/*100.com.fits')
stack_imglist = []
#for filter_, target_path in target_images.items():
for target_path in target_images:
    stack_imglist.append(ScienceImage(path=target_path, telinfo=telinfo, load=True))
#%%
#stack_imglist = [ScienceImage(path=row['file'], telinfo=telinfo, load=False) for row in imginfo_all]
import glob
from tippy.catalog import TIPCatalogDataset
from tippy.catalog import TIPCatalog
stacked_cataloglist = [TIPCatalog(stacked_img.savepath.catalogpath, catalog_type = 'all') for stacked_img in stack_imglist]

cataloglist_gppy = glob.glob('/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T06911/*/*/*/*com.phot.cat')
stacked_cataloglist_gppy = [TIPCatalog(catalog_path, catalog_type = 'all') for catalog_path in cataloglist_gppy]
for stack_catalog_gppy in stacked_cataloglist_gppy:
    stack_path_gppy = stack_catalog_gppy.find_corresponding_fits()
    stack_img_gppy = ScienceImage(path=stack_path_gppy, telinfo=telinfo, load=True)
    stack_catalog_gppy.load_target_img(stack_img_gppy)
    filter_ = stack_catalog_gppy.info.filter
    mag_keys = ['MAG_AUTO', 'MAG_APER', 'MAG_APER_1', 'MAG_APER_2', 'MAG_APER_3', 'MAG_APER_4', 'MAG_APER_5']
    magerr_keys = ['MAGERR_AUTO', 'MAGERR_APER', 'MAGERR_APER_1', 'MAGERR_APER_2', 'MAGERR_APER_3', 'MAGERR_APER_4', 'MAGERR_APER_5']
    stack_catalog_gppy.data.rename_columns(['ALPHA_J2000', 'DELTA_J2000'], ['X_WORLD', 'Y_WORLD'])
    for mag_key, magerr_key in zip(mag_keys, magerr_keys):
        magsky_key_prev = f'{mag_key}_{filter_}'
        keys_split = mag_key.split('MAG_')
        magsky_key_new = f'MAGSKY_{keys_split[1]}'
        magerr_key_prev = f'{magerr_key}_{filter_}'
        magerr_key_new = f'MAGSKYERR_{keys_split[1]}'
        stack_catalog_gppy.data.rename_columns([magsky_key_prev, magerr_key_prev], [magsky_key_new, magerr_key_new])

#Only return medium band filters
stacked_cataloglist = [catalog for catalog in stacked_cataloglist if catalog.info.filter.startswith('m')]
stacked_cataloglist_gppy = [catalog for catalog in stacked_cataloglist_gppy if catalog.info.filter.startswith('m')]
dataset = TIPCatalogDataset(stacked_cataloglist)
dataset_gppy = TIPCatalogDataset(stacked_cataloglist_gppy)
for stacked_catalog in stacked_cataloglist:
    stack_path = stacked_catalog.find_corresponding_fits()
    stack_img = ScienceImage(path=stack_path, telinfo=telinfo, load=True)
    stacked_catalog.load_target_img(stack_img)
# %%
from tippy.dataobjects import PhotometricSpectrum
# %%
P = PhotometricSpectrum(source_catalogs = dataset)
P_gp = PhotometricSpectrum(source_catalogs = dataset_gppy)
# %%
P.update_data(data_keys = ['MAGSKY_AUTO', 'MAGSKY_APER', 'MAGSKY_APER_1', 'MAGSKY_APER_2', 'MAGERR_AUTO', 'MAGERR_APER', 'MAGERR_APER_1', 'MAGERR_APER_2'])
P_gp.update_data(data_keys = ['MAGSKY_AUTO', 'MAGSKY_APER_1', 'MAGSKY_APER_2', 'MAGSKY_APER_3', 'MAGSKY_APER_4', 'MAGSKY_APER_5', 'MAGSKYERR_AUTO', 'MAGSKYERR_APER_1', 'MAGSKYERR_APER_2', 'MAGSKYERR_APER_3', 'MAGSKYERR_APER_4', 'MAGSKYERR_APER_5'])
#%%
detected_sources = P.data[P.data['n_detections'] > 16]

#%%
i =  4
coord = detected_sources['coord'][i]
ra = coord.ra.value
dec = coord.dec.value
#ra = 222.2123987 
#dec = -40.1460314
key = 'AUTO'
P.plot(ra = ra,
       dec = dec,
       flux_key = f'MAGSKY_{key}', 
       fluxerr_key = f'MAGERR_{key}', 
       matching_radius_arcsec = 1.0,
       overplot_gaiaxp = True,
       color_key = 'FILTER'
       )
# %%
key = 'AUTO'
ax = P_gp.plot(
       ra = ra,
       dec = dec,
       flux_key = f'MAGSKY_{key}', 
       fluxerr_key = f'MAGSKYERR_{key}', 
       matching_radius_arcsec = 1.0,
       overplot_gaiaxp = True,
       color_key = 'FILTER'
)
#%%
P.source_catalogs.catalogs[6].show_source(ra, dec, 4, 50, 5)
# %%
P_gp.source_catalogs.catalogs[7].show_source(ra, dec, 4, 50, 5)
# %%
P_gp.source_catalogs.catalogs[6].search_sources(ra, dec, 'coord', 2)
# %%
