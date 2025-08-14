#%%

from ezphot.methods import *
from ezphot.imageobjects import *
from ezphot.helper import Helper
from ezphot.utils import SDTDataQuerier
from tqdm import tqdm
import gc
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
#%%
# Use 'Agg' backend for matplotlib to avoid GUI issues in non-interactive environments
matplotlib.use('Agg') 

#%%
# Initialize the classes
Preprocessor = Preprocess()
PlateSolver = Platesolve()
MaskGenerator = TIPMasking()
BkgGenerator = BackgroundEstimator()
ErrormapGenerator = TIPErrormap()
AperturePhotometry = AperturePhotometry()
PSFPhotometry = PSFPhotometry()
PhotometricCalibration = PhotometricCalibration()
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
    mask_radius_factor = 1.5,
    saturation_level = 50000,
    save = False,
    verbose = True,
    visualize = False,
    save_fig = True
)

do_circularmask = True
circlemask_kwargs = dict(
    x_position = 233.8476795,
    y_position = 12.0483362,
    radius_arcsec = 150,
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
    aperture_diameter_arcsec = [5,7,10],
    aperture_diameter_seeing = [2.5, 3.5],
    saturation_level = 60000,
    kron_factor = 1.5,
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
    mag_lower = 13,
    mag_upper = 15,
    dynamic_mag_range = True,
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
    aperture_diameter_arcsec = [5,7,10],
    saturation_level = 60000,
    kron_factor = 1.5,
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
    mag_lower = 13,
    mag_upper = 15,
    dynamic_mag_range = True,
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

#%% Query object frames
from ezphot.helper import DataBrowser
helper = Helper()
data_qurier = SDTDataQuerier()
browser = DataBrowser('scidata')
browser.objname = 'T22956'
browser.filter = 'g'
target_imgset = browser.search('calib*100.fits', 'science')
target_imglist = target_imgset.target_images
target_path = target_imglist[0].path
telinfo = target_imglist[0].telinfo
#telinfo = helper.estimate_telinfo(target_path)
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

    status['circular_mask'] = None
    if do_circularmask:
        try:
            # Generate the circular mask
            target_srcmask = MaskGenerator.mask_circle(
                target_img = target_img,
                target_mask = None,
                mask_type = 'source',
                **circlemask_kwargs
            )
            status['circular_mask'] = True

        except Exception as e:
            status['circular_mask'] = e


    status['mask'] = None
    target_srcmask = None
    if do_generatemask:
        try:
            # Mask the object frames
            target_srcmask = MaskGenerator.mask_sources(
                target_img = target_img,
                target_mask = target_srcmask,
                **mask_kwargs
            )
            status['mask'] = True
        except:
            status['mask'] = e

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
#%%
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
#%%
def stackprocess(stack_path, stack_bkgrmspath, telinfo):
    stack_img = ScienceImage(path=stack_path, telinfo=telinfo, load=True)
    stack_bkgrms = Errormap(path=stack_bkgrmspath, emaptype='bkgrms', load=True)
    stacked_status = dict()
    try:
        # Perform aperture photometry
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

    calib_catalog = None
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

stack_images = data_qurier.show_scidestdata(
    tile_id,
    False,
    'filter',
    'calib*100.com.fits'
)
target_imglist_all = [target_imgpath for target_imglist in stack_images.values() for target_imgpath in target_imglist]
stack_imglist = [ScienceImage(path=path, telinfo=telinfo, load=True) for path in target_imglist_all]
stack_bkgrmslist = [Errormap(path=stack_img.savepath.bkgrmspath, emaptype='bkgrms', load=True) for stack_img in stack_imglist]
#%%
arglist = [(stack_img.path, stack_bkgrms.path, telinfo) for stack_img, stack_bkgrms in zip(stack_imglist, stack_bkgrmslist)]
with ProcessPoolExecutor(max_workers=20) as executor:
    failed_stacked_path = []
    futures = [executor.submit(stackprocess, *args) for args in arglist]
    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = future.result()
        except Exception as e:
            failed_stacked_path.append(args[0])
            print(f"[ERROR] {e}")
#%%
from ezphot.catalog import Catalog, CatalogDataset

if __name__ == "__main__":
    source_catalogs_LC = CatalogDataset()
    source_catalogs_LC.search_catalogs(
        target_name = 'T22956',
        search_key = '100.com.fits.cat'
     )    
# %%
if __name__ == "__main__":
    fit_filter_key = ['g', 'r', 'i', 'm425', 'm650', 'm775', 
                      'm475', 'm525', 'm550', 'm575', 'm600', 'm625', 'm650', 
                      'm675', 'm700', 'm725', 'm750', 'm775', 'm800', 'm825', 'm850', 'm875']
    ra = 233.857430764 # SN2025fvw
    dec = 12.0577222937
    source_catalogs_LC.select_catalogs(filter = fit_filter_key)
    #source_catalogs_LC.exclude_catalogs(telname = '7DT10')
    source_catalogs_LC.select_sources(ra, dec, radius =  60)
#%%
if __name__ == "__main__":
    from ezphot.dataobjects import LightCurve
    self = LightCurve(source_catalogs_LC)
    self.merge_catalogs()
    self.extract_source_info(ra, dec)
# %%
if __name__ == "__main__":
    source =self.data[0]

    flux_key = 'MAGSKY_APER_2'
    fluxerr_key = 'MAGERR_APER_2'
    matching_radius_arcsec = 5
    color_key: str = 'filter'#'OBSDATE'
    overplot_gaiaxp = False
    overplot_sdss = False
    overplot_ps1 = False
    self.plt_params.ylim = [26, 7]
    #self.plt_params.xlim = [60757, 60780]
    self.plt_params.figure_figsize = (8, 10)
    self.plt_params.label_position = 'upper right'
    # self.FILTER_OFFSET['m850'] = +2
#%%
if __name__ == "__main__":
    from astropy.time import Time
    figs, axs, matched_sources = self.plot(ra, 
                          dec, 
                          flux_key=flux_key, 
                          color_key = color_key, 
                          matching_radius_arcsec=matching_radius_arcsec,
                          overplot_gaiaxp=overplot_gaiaxp,
                          overplot_sdss = overplot_sdss,
                          overplot_ps1 = overplot_ps1)
    ztf_g_mag = [16.9141, 15.334]
    ztf_g_magerr = [0.0708192, 0.0231074]
    ztf_g_obsdate = [2460762.0002315, 2460764.9897801]
    ztf_g_obsdate_mjd = [Time(obsdate, format='jd').mjd for obsdate in ztf_g_obsdate]
    ztf_r_mag = [18.9175 + 1.0]
    ztf_r_magerr = [0.0909718]
    ztf_r_obsdate = [2460759.842419]
    ztf_r_obsdate_mjd = [Time(obsdate, format='jd').mjd for obsdate in ztf_r_obsdate]
    ps_r_mag = [17.24 +1.0]
    ps_r_magerr = [0.03]
    ps_r_obsdate = [2460761.02748]
    ps_r_obsdate_mjd = [Time(obsdate, format='jd').mjd for obsdate in ps_r_obsdate]
    # atlas_o_ul = [18.94]
    # atlas_o_obsdate = [2460759.80766]
    # atlas_o_obsdate_mjd = [Time(obsdate, format='jd').mjd for obsdate in atlas_o_obsdate]
    ztf_r_ul = [20.31]
    ztf_r_obsdate = [2460758.9986343]
    ztf_r_obsdate_mjd = [Time(obsdate, format='jd').mjd for obsdate in ztf_r_obsdate]
    axs[0].errorbar(ztf_g_obsdate_mjd, np.array(ztf_g_mag)-2, yerr=ztf_g_magerr, ms = 10, fmt='*', mfc='green', mec = 'black', label='ZTF g')
    axs[0].errorbar(ztf_r_obsdate_mjd, ztf_r_mag , yerr=ztf_r_magerr, ms = 10, fmt='*', mfc='red',  mec = 'black', label='ZTF r')
    axs[0].errorbar(ps_r_obsdate_mjd, ps_r_mag , yerr=ps_r_magerr, ms = 10, fmt='^', mfc='red',  mec = 'black', label='PS1 r')
    # axs[0].errorbar(atlas_o_obsdate_mjd, atlas_o_ul, yerr=0, ms = 10, fmt='v', mfc='orange',  mec = 'black', label='ATLAS-O [UL]')
    axs[0].errorbar(ztf_r_obsdate_mjd, ztf_r_ul, yerr=0, ms = 10, fmt='v', mfc='red',  mec = 'black', label='ZTF r [UL]')
    axs[0].legend(loc='upper left', ncol = 2, fontsize=12)
#%%
figs[0]
 # %%
from astropy.table import Table
flux_key = 'MAGSKY_APER_2'
fluxerr_key = 'MAGERR_APER_2'
obs_tbl = self.data
# Initialize empty lists to store combined values
obsdates = []
mags = []
magerrs = []
filters = []

# Iterate over all columns to gather data by filter
for row in obs_tbl:
    
    # Append data
    obsdates.append(row['mjd'])
    mags.append(row[flux_key])
    magerrs.append(row[fluxerr_key])
    filters.append(row['filter'])

obsdates.append(Time(ztf_r_obsdate, format = 'jd').mjd[0])
mags.append(ztf_r_mag[0])
magerrs.append(ztf_r_magerr[0])
filters.append('r')
obsdates.append(Time(ps_r_obsdate, format = 'jd').mjd[0])
mags.append(ps_r_mag[0])
magerrs.append(ps_r_magerr[0])
filters.append('r')
obsdates.append(Time(ztf_g_obsdate, format = 'jd').mjd[0])
mags.append(ztf_g_mag[0])
magerrs.append(ztf_g_magerr[0])
filters.append('g')
obsdates.append(Time(ztf_g_obsdate, format = 'jd').mjd[1])
mags.append(ztf_g_mag[1])
magerrs.append(ztf_g_magerr[1])
filters.append('g')

# Create the table
fit_tbl = Table()
fit_tbl['obsdate'] = obsdates
fit_tbl['mag'] = mags
fit_tbl['e_mag'] = magerrs
fit_tbl['filter'] = filters
fit_tbl['magsys'] = ['AB'] * len(fit_tbl)
#%%
clean_idx = [True] * len(fit_tbl)
for column in fit_tbl.colnames:
    try:
        clean_idx &= ~fit_tbl[column].mask
    except:
        pass
fit_tbl = fit_tbl[clean_idx]
filter_tbl = fit_tbl.group_by('filter').groups
#%% FIT TO FIREBALL MODEL
import numpy as np
from scipy.optimize import curve_fit
exp_fit_tbl = fit_tbl[fit_tbl['obsdate'] < 60770]
def exp_model(t, A, tau, B, t_ref):
    return A * np.exp(-(t - t_ref) / tau) + B

fit_data = {}
for f in np.unique(exp_fit_tbl['filter']):
    mask = exp_fit_tbl['filter'] == f
    t = np.array(exp_fit_tbl['obsdate'][mask])
    m = np.array(exp_fit_tbl['mag'][mask])
    e = np.array(exp_fit_tbl['e_mag'][mask])
    if len(t) < 3:
        continue
    t_ref = t[0]  # Fixed anchor
    try:
        # Wrap model for curve_fit with fixed t_ref
        def model_to_fit(t, A, tau, B):
            return exp_model(t, A, tau, B, t_ref)
        popt, pcov = curve_fit(model_to_fit, t, m, sigma=e, absolute_sigma=True,
                               p0=(1.0, 5.0, np.median(m)))
        fit_data[f] = (t, m, e, popt, t_ref)
    except RuntimeError:
        print(f"[WARNING] Fit failed for filter {f}")

# Extend time range by ±3 days
obsdate_min = min(fit_tbl['obsdate'])
obsdate_max = max(fit_tbl['obsdate'])
t_fit = np.linspace(obsdate_min - 3, obsdate_max + 3, 300)

for f, (t, m, e, popt, t_ref) in fit_data.items():
    A, tau, B = popt
    m_fit = exp_model(t_fit, A, tau, B, t_ref)
    if f == 'r':
        m_fit += 1
    elif f.startswith('m'):
        m_fit += self.FILTER_OFFSET.get(f, 0)
    axs[0].plot(t_fit, m_fit, c = self.FILTER_COLOR.get(f), linestyle ='--', linewidth = 0.5, label=f'{f} exp fit')
#%%
figs[0]

#%% Spectral template fitting
helper = Helper()
color_key, offset_key, filter_key_sncosmo, _, name_key = helper.load_filt_keys()
#%%
# from pathlib import Path
# all_filters = glob.glob('/home/hhchoi1022/code/ezphot/ezphot/configuration/7DT_filters/*fitted.csv')
# for file_ in all_filters:
#     filename = Path(file_)
#     stem = filename.stem
#     filter_ = stem.split('_fitted')[0]
#     if len(filter_) > 2:
#         filter_ = f'm{filter_[:3]}'
#     else:
#         filter_ = filter_
#     new_filename = filename.parent / f'{filter_}.csv'
#     os.rename(filename, new_filename)
#     print(f'Renaming {filename} to {new_filename}')
#%% Filter registration
import sncosmo
import glob
from astropy.io import ascii
from astropy import units as u
#%%
import os
import glob
import sncosmo
from astropy.io import ascii
import astropy.units as u

def registerfilter(responsefile, name, force=True):
    tbl = ascii.read(responsefile, format='csv')
    tbl.sort('wavelength')
    tbl.remove_column('col0')
    tbl = tbl[(tbl['wavelength'] > 200 ) & (tbl['wavelength'] < 900)] # Filter out wavelengths outside the range of 2000-10000 nm
    band = sncosmo.Bandpass(tbl['wavelength'], tbl['transmission'], wave_unit=u.nm, name=name)
    sncosmo.register(band, force=force)

# Register all filters in the specified directory
list_responsefile = glob.glob('./filter_transmission/7DT/*.csv')
for responsefile in list_responsefile:
    filename = os.path.basename(responsefile)
    filter_name = filename.replace('.csv', '')
    try:
        registerfilter(responsefile, filter_name, force=True)
    except Exception as e:
        print(f"Error registering filter {filter_name}: {e}")
#%% Data 
filterkeylist = [] 
for filter_ in fit_tbl['filter']:
    if filter_key_sncosmo.get(filter_) is None:
        filterkeylist.append(filter_)
    else:
        filter_key = filter_key_sncosmo[filter_]
        filterkeylist.append(filter_key)

fit_tbl['filter_sncosmo'] = filterkeylist
show_tbl = fit_tbl
#%%
formatted_fit_tbl = helper.SNcosmo_format(fit_tbl['obsdate'], fit_tbl['mag'], fit_tbl['e_mag'], fit_tbl['filter_sncosmo'], magsys = fit_tbl['magsys'], zp = 25)
# formatted_fit_tbl = formatted_fit_tbl[(formatted_fit_tbl['mjd'] < 60825)]
formatted_fit_tbl = formatted_fit_tbl[((formatted_fit_tbl['mjd'] < 60767) & 
                                      (formatted_fit_tbl['mjd'] > 60766)) |
                                      ((formatted_fit_tbl['mjd'] < 60791) &
                                      (formatted_fit_tbl['mjd'] > 60790))]
# formatted_fit_tbl = formatted_fit_tbl[(formatted_fit_tbl['band'] == 'sdss::g') | 
#                                       (formatted_fit_tbl['band'] == 'sdss::r') |
#                                       (formatted_fit_tbl['band'] == 'sdss::i') ]
#%%
helper.group_table(formatted_fit_tbl, 'mjd', 0.5)
#%%SNcosmo
#chi-square
def calc_chisq(formatted_fit_tbl, filt_):
    chisquare = 0
    data = formatted_fit_tbl[formatted_fit_tbl['band'] == filt_]
    for obsdate in data['mjd']:
        flux = fitted_model.bandflux(filt_,obsdate, zp = 25, zpsys = 'ab')
        obs_flux = data[data['mjd'] == obsdate]['flux'][0]
        obs_fluxerr = data[data['mjd'] == obsdate]['fluxerr'][0]
        delflux = obs_flux - flux
        pull = delflux/obs_fluxerr
        chisquare += pull**2
    reduced_chisq = chisquare/(len(data)-2)
    return reduced_chisq

#source = sncosmo.get_source('salt3')
model = sncosmo.Model(source='salt3')
dust = sncosmo.CCM89Dust()
import sfdmap
#dustmap = sfdmap.SFDMap("./sfddata-master")
#ebv = dustmap.ebv(ra, dec)
# model.add_effect(dust, 'mw', 'obs')
#model.set(mwebv = ebv)
# model.set(mwr_v = 3.1)
# model.add_effect(dust, 'host', 'rest')
# model.set(hostebv = 0.097)
# model.set(hostr_v = 2.3)
#model.set(z = 0.005017)
result , fitted_model= sncosmo.fit_lc(
    formatted_fit_tbl, model,
    #['t0', 'amplitude'], # hsiao
    ['t0', 'x0', 'x1', 'c', 'z'], #salt2 or salt3
    bounds = {'z': (0.0005, 0.1)}
    )

figtext =  ''
for band in set(formatted_fit_tbl['band']):
    chisq = round(calc_chisq(formatted_fit_tbl, band),2)
    figtext +=f"$[reduced\ \  \chi^2$]{band}={chisq}\n"
sncosmo.plot_lc(formatted_fit_tbl, model=fitted_model, errors=result.errors, figtext = figtext, ncol = 3,  xfigsize = 10, tighten_ylim=False)

#%%stretch parameter & delmag
import numpy as np
result.parameters
z = result.parameters[0]
t0 = result.parameters[1]
x0 = result.parameters[2]
x1 = result.parameters[3]
c = result.parameters[4]
e_x1 = result.errors['x1']
e_c = result.errors['c']

def saltt3_to_salt2(x1,c):
    x1_salt2 = ((0.985/0.138)*x1 - c - 0.005 - (0.985/0.138))*0.138/0.985/1.028
    c_salt2 = (c - 0.002 * x1_salt2 - 0.013)/0.985
    return x1_salt2, c_salt2
#x1, c = saltt3_to_salt2(x1, c)

param_stretch = 0.98+ 0.091*x1+ 0.003*x1**2- 0.00075*x1**3
e_param_stretch = np.sqrt((0.091*e_x1)**2+(0.003*2*e_x1)**2+(0.00075*3*e_x1)**2)
delmag = 1.09- 0.161*x1+ 0.013*x1**2- 0.00130*x1**3
e_delmag = np.sqrt((0.161*e_x1)**2+(0.013*2*e_x1)**2+(0.00130*3*e_x1)**2)
t_max = result.parameters[1]

t_range = np.arange(t_max - 20, t_max + 45, 0.01)
Bmag_range = fitted_model.bandmag('bessellb', 'ab', t_range)
nan_mask = np.isnan(Bmag_range)
Bmag_range = Bmag_range[~nan_mask]
t_range = t_range[~nan_mask]
t_max = t_range[np.argmin(Bmag_range)]
magB_idx = np.argmin(Bmag_range)
magB_max = Bmag_range[magB_idx]
timeB_max = t_range[magB_idx]
Vmag_range = fitted_model.bandmag('bessellv', 'ab', t_range)
magV_idx = np.argmin(Vmag_range)
magV_max = Vmag_range[magV_idx]
timeV_max = t_range[magV_idx]
Rmag_range = fitted_model.bandmag('bessellr', 'ab', t_range)
magR_idx = np.argmin(Rmag_range)
magR_max = Rmag_range[magR_idx]
timeR_max = t_range[magR_idx]
gmag_range = fitted_model.bandmag('sdssg', 'ab', t_range)
magg_idx = np.argmin(gmag_range)
magg_max = gmag_range[magg_idx]
timeg_max = t_range[magg_idx]
rmag_range = fitted_model.bandmag('sdssr', 'ab', t_range)
magr_idx = np.argmin(rmag_range)
magr_max = rmag_range[magr_idx]
timer_max = t_range[magr_idx]
imag_range = fitted_model.bandmag('sdssi', 'ab', t_range)
magi_idx = np.argmin(imag_range)
magi_max = imag_range[magi_idx]
timei_max = t_range[magi_idx]

magB15_time = timeB_max + 15
Bmag_later = fitted_model.bandmag('bessellb', 'ab', magB15_time)
mB_15 = magB_max - Bmag_later

maxpoint = fit_tbl[np.abs(fit_tbl['obsdate']-int(t_max))<1]
e_magerr_max = np.median(maxpoint['e_mag'])

nu = magB_max - (-19.31) + 0.13 * x1 - 1.77 * c # Guy et al. 2007, SALT2 paper +- 0.03, +- 0.013, +- 0.16
e_nu = np.sqrt(e_magerr_max**2 + 0.03**2 + (np.sqrt((0.013/0.13)**2+(e_x1/x1)**2))**2 + (np.sqrt((0.16/1.77)**2+(np.abs(e_c/c))**2))**2)
#nu = magB_max - (-19.31 + 5*np.log10(const_hubble/70)) + 1.52*(param_stretch-1) - 1.57 * c # P. Astier, et al. 2005
#e_nu = np.sqrt(e_magB_max**2 + 0.03**2 + (np.sqrt((0.14/1.52)**2+(e_param_stretch/param_stretch)**2))**2 + (np.sqrt((0.15/1.57)**2+(np.abs(e_c/c))**2))**2)

const_hubble = 70 

#nu = magB_max - (-19.218) + 1.295*(param_stretch-1) - 3.181*c # Guy et al. 2010
distance = 10**((nu +5)/5) # unit = pc
e_distance = 10**((nu+e_nu +5)/5) - distance
#%%
print(f'distance = {distance}+-{e_distance}')
print(f't_max_all = t_max_all = {t_max}')
print(f't_max_filt = B={timeB_max}, V = {timeV_max}, R = {timeR_max}, g={timeg_max}, r = {timer_max}, i = {timei_max}')
print(f'mag_max = B={magB_max}, V = {magV_max}, R = {magR_max}, g={magg_max}, r = {magr_max}, i = {magi_max}')
print(f'ABSmag_max = B={magB_max-nu}, V = {magV_max-nu}, R = {magR_max-nu}, g={magg_max-nu}, r = {magr_max-nu}, i = {magi_max-nu}')
print(f'stretch_param = {param_stretch}+-{e_param_stretch}')
print(f'delmag = {mB_15}+-{e_delmag}')
print(f'magB_max = {magB_max}')
print(f'ABSmagB_max = {magB_max-nu}+-{e_magerr_max}')
print(f'nu = {nu}+-{e_nu}')
#%%
source =self.data[0]

flux_key = 'MAGSKY_AUTO'
fluxerr_key = 'MAGERR_AUTO'
matching_radius_arcsec = 5
color_key: str = 'filter'#'OBSDATE'
overplot_gaiaxp = False
overplot_sdss = False
overplot_ps1 = False
self.plt_params.ylim = [26, 7]
self.plt_params.xlim = [60757, 60815]
self.plt_params.figure_figsize = (8, 12)
# self.FILTER_OFFSET['m450'] = -2
# self.FILTER_OFFSET['m650'] = -1
# self.FILTER_OFFSET['m850'] = +2
figs, axs, matched_sources = self.plot(ra, 
                    dec, 
                    flux_key=flux_key, 
                    color_key = color_key, 
                    matching_radius_arcsec=matching_radius_arcsec,
                    overplot_gaiaxp=overplot_gaiaxp,
                    overplot_sdss = overplot_sdss,
                    overplot_ps1 = overplot_ps1)
ztf_g_mag = [16.9141, 15.334]
ztf_g_magerr = [0.0708192, 0.0231074]
ztf_g_obsdate = [2460762.0002315, 2460764.9897801]
ztf_g_obsdate_mjd = [Time(obsdate, format='jd').mjd for obsdate in ztf_g_obsdate]
ztf_r_mag = [18.9175 + 1.0]
ztf_r_magerr = [0.0909718]
ztf_r_obsdate = [2460759.842419]
ztf_r_obsdate_mjd = [Time(obsdate, format='jd').mjd for obsdate in ztf_r_obsdate]
ps_r_mag = [17.24 +1.0]
ps_r_magerr = [0.03]
ps_r_obsdate = [2460761.02748]
ps_r_obsdate_mjd = [Time(obsdate, format='jd').mjd for obsdate in ps_r_obsdate]
axs[0].errorbar(ztf_g_obsdate_mjd, ztf_g_mag, yerr=ztf_g_magerr, ms = 10, fmt='*', mfc='green', mec = 'black', label='ZTF g')
axs[0].errorbar(ztf_r_obsdate_mjd, ztf_r_mag , yerr=ztf_r_magerr, ms = 10, fmt='*', mfc='red',  mec = 'black', label='ZTF r')
axs[0].errorbar(ps_r_obsdate_mjd, ps_r_mag , yerr=ps_r_magerr, ms = 10, fmt='^', mfc='red',  mec = 'black', label='PS1 r')
axs[0].legend(loc='upper left', ncol = 2, fontsize=12)
# %%
for filter_ in fit_filter_key:
    mag = fitted_model.bandmag(filter_, 'ab', t_range)
    mag_offset = self.FILTER_OFFSET.get(filter_)
    axs[0].plot(t_range, mag + mag_offset, c=self.FILTER_COLOR.get(filter_), linestyle='-', linewidth=1.5, label=f'{filter_} model')
# %%
figs[0]
# %%
if __name__ == "__main__":
    from ezphot.dataobjects import PhotometricSpectrum
    source_catalogs_PS = CatalogDataset()
    source_catalogs_PS.search_catalogs(
        target_name = 'T22956',
        search_key = '100.com.fits.cat'
    )
    source_catalogs_PS.select_sources(ra, dec, radius=60)
#%%
if __name__ == "__main__":
    fit_filter_key = ['m400', 'm425', 'm450', 
                      'm475', 'm525', 'm550', 'm575', 'm600', 'm625', 'm650', 
                      'm675', 'm700', 'm725', 'm750', 'm775', 'm800', 'm825', 'm850', 'm875']

    source_catalogs_PS.select_catalogs(filter=fit_filter_key, obs_start = '2025-03-28', obs_end = '2025-04-20')
    #source_catalogs_PS.exclude_catalogs(telname = '7DT10')
    spec = PhotometricSpectrum(source_catalogs_PS)
    spec.merge_catalogs()
    spec.extract_source_info(ra, dec)
#%%
if __name__ == "__main__":

    spec.OFFSET = 3
    spec.ncol = 3
    #spec.plt_params.ylim = [50,10]

# %%
fig, axs, _ = spec.plot(ra, dec, color_key = 'obsdate', flux_key = 'MAGSKY_AUTO', fluxerr_key = 'MAGERR_AUTO')
#%%
from astropy.time import Time
import numpy as np
data_tbl = spec.data
all_dates = Time(data_tbl['mjd'], format = 'mjd').to_value('iso', subfmt = 'date')
data_tbl['obsdate_str'] = all_dates
tbl_group = data_tbl.group_by('obsdate_str').groups
wl_range = np.arange(4000, 9000, 1)
#%%
import matplotlib.pyplot as plt
plt.figure()
for i, tbl in enumerate(tbl_group):
    obsdate = np.mean(tbl['mjd'][0])
    print(obsdate)
    wl = wl_range
    f_lambda = fitted_model.flux(np.array(obsdate), np.array(wl))
    abmag = -2.5 * np.log10(f_lambda) - 5 * np.log10(wl) - 2.406 + i * spec.OFFSET
    wl_show = wl / 10
    axs[0].plot(wl_show, abmag, label=f'{obsdate:.5f} MJD', c='black', linewidth = 1, alpha=0.5)
#%%
fig[0]
