

#%%
import time
from ezphot.methods import *
from ezphot.imageobjects import *
from ezphot.helper import Helper
from ezphot.utils import DataBrowser
from ezphot.utils import SDTDataQuerier
from tqdm import tqdm
import gc
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
#%%
#matplotlib.use('Agg') 

#%%
def initialize_context():
    context = {}
    context['Preprocessor'] = Preprocess()
    context['PlateSolver'] = Platesolve()
    context['MaskGenerator'] = MaskGenerator()
    context['BkgGenerator'] = BackgroundGenerator()
    context['ErrormapGenerator'] = ErrormapGenerator()
    context['AperturePhotometry'] = AperturePhotometry()
    context['PSFPhotometry'] = PSFPhotometry()
    context['PhotometricCalibration'] = PhotometricCalibration()
    context['Stacking'] = Stack()
    context['DIA'] = Subtract()
    context['sdt_data_querier'] = SDTDataQuerier()
    context['Databrowser'] = DataBrowser('scidata')
    context['helper'] = Helper()
    return context

#%%
def get_config():
    config = {}
    #=============== CONFIOGURATION FOR SINGLE IMAGE PROCESSING ===============#
    config['save'] = True
    config['visualize'] = True
    config['save_fig'] = True
    config['verbose'] = True

    config['do_platesolve'] = False
    config['platesolve_kwargs'] = dict(
        overwrite = True,
        verbose = config['verbose'],
    )

    config['do_generatemask'] = True
    config['mask_kwargs'] = dict(
        sigma = 5,
        mask_radius_factor = 1.5,
        saturation_level = 50000,
        save = config['save'],
        verbose = config['verbose'],
        visualize = config['visualize'],
        save_fig = config['save_fig']
    )

    config['do_circularmask'] = False
    config['circlemask_kwargs'] = dict(
        x_position = 245.8960980,
        y_position = -26.5227669,
        radius_arcsec = 350,
        unit = 'deg',
        save = config['save'],
        verbose = config['verbose'],
        visualize = config['visualize'],
        save_fig = config['save_fig']
    )

    config['do_objectmask'] = False
    config['objectmask_kwargs'] = dict(
        x_position = 245.8960980,
        y_position = -26.5227669,
        radius_arcsec = 600,
        unit = 'deg',
        save = config['save'],
        verbose = config['verbose'],
        visualize = config['visualize'],
        save_fig = config['save_fig']
    )

    config['do_generatebkg'] = True
    config['bkg_kwargs'] = dict(
        box_size = 64,
        filter_size = 3,
        n_iterations = 0,
        mask_sigma = 3,
        mask_radius_factor = 3,
        mask_saturation_level = 60000,
        save = config['save'],
        verbose = config['verbose'],
        visualize = config['visualize'],
        save_fig = config['save_fig']
    )

    config['do_generateerrormap'] = True
    config['errormap_from_propagation'] = True
    config['errormap_kwargs'] = dict(
        save = True,
        verbose = config['verbose'],
        visualize = config['visualize'],
        save_fig = config['save_fig']
    )

    config['do_aperturephotometry'] = True
    config['aperturephotometry_kwargs'] = dict(
        sex_params = None,
        detection_sigma = 5,
        aperture_diameter_arcsec = [5, 7, 10],
        aperture_diameter_seeing = [3.5, 4.5],
        save_transient_figure = True,
        save_candidate_figure = True,
        show_transient_numbers = 100,
        show_candidate_numbers = 100,
        saturation_level = 60000,
        kron_factor = 2.5,
        save = config['save'],
        verbose = config['verbose'],
        visualize = config['visualize'],
        save_fig = config['save_fig'],
    )

    config['do_photometric_calibration'] = True
    config['photometriccalibration_kwargs'] = dict(
        catalog_type = 'GAIAXP',
        max_distance_second = 1.0,
        calculate_color_terms = True,
        calculate_mag_terms = True,
        classstar_lower = 0.7,
        elongation_upper = 3,
        elongation_sigma = 5,
        mag_lower = 13,
        mag_upper = 16,
        dynamic_mag_range = False,
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
        save = config['save'],
        verbose = config['verbose'],
        visualize = config['visualize'],
        save_fig = config['save_fig'],
        save_refcat = True,
    )

    #=============== CONFIGURATION FOR STACKING ===============#
    config['do_stacking'] = True
    config['stacking_kwargs'] = dict(
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
        verbose = config['verbose'],
        save = config['save']
    )

    #=============== CONFIGURATION FOR STACKED IMAGE PROCESSING ===============#
    config['stack_aperturephotometry_kwargs'] = dict(
        sex_params = None,
        detection_sigma = 5,
        aperture_diameter_arcsec = [5,7,10],
        aperture_diameter_seeing = [3.5, 4.5],
        saturation_level = 60000,
        kron_factor = 2.5,
        save = config['save'],
        verbose = config['verbose'],
        visualize = config['visualize'],
        save_fig = config['save_fig'],
    )

    config['stack_photometriccalibration_kwargs'] = dict(
        catalog_type = 'GAIAXP',
        max_distance_second = 1.0,
        calculate_color_terms = True,
        calculate_mag_terms = True,
        classstar_lower = 0.7,
        elongation_upper = 3,
        elongation_sigma = 5,
        mag_lower = 13,
        mag_upper = 16,
        dynamic_mag_range = False,
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
        save = config['save'],
        verbose = config['verbose'],
        visualize = config['visualize'],
        save_fig = config['save_fig'],
        save_starcat = False,
        save_refcat = True,
    )

    # =============== CONFIGURATION FOR DIA =============== #
    config['DIA_kwargs'] = dict(
        detection_sigma = 5,
        aperture_diameter_arcsec = [5, 7, 10],
        aperture_diameter_seeing = [3.5, 4.5],
        target_transient_number = 5,
        reject_variable_sources = True,
        negative_detection = True,
        reverse_subtraction = False,
        save = config['save'],
        verbose = config['verbose'],
        visualize = config['visualize'],
        show_transient_numbers = 100
    )
    return config

#%%
context = initialize_context()
config = get_config()
print(context['Preprocessor'])
print(context['PlateSolver'])
print(context['MaskGenerator'])
print(context['BkgGenerator'])
print(context['ErrormapGenerator'])
print(context['AperturePhotometry'])
print(context['PSFPhotometry'])
print(context['Stacking'])
#%% Preprocessing
import itertools
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_tile_filter(args, context):
    tile_id, filter_name = args
    print(f'Processing tile {tile_id} with filter {filter_name}')
    
    Databrowser = context['Databrowser'].__class__('rawdata')
    Databrowser.obsdate = '2025-07-25_gain2750'
    Preprocessor = context['Preprocessor']
    
    try:
        target_imglist = Databrowser.search(f'*{tile_id}*_{filter_name}_*.fits', 'science')
    except Exception as e:
        print(f"Error searching images for {tile_id}-{filter_name}: {e}")
        return

    for target_img in tqdm(target_imglist):
        try:
            mbias_path = Preprocessor.get_masterframe_from_image(target_img, imagetyp='BIAS', max_days=60)[0]
            mdark_path = Preprocessor.get_masterframe_from_image(target_img, imagetyp='DARK', max_days=60)[0]
            mflat_path = Preprocessor.get_masterframe_from_image(target_img, imagetyp='FLAT', max_days=60)[0]

            mbias = MasterImage(path=mbias_path['file'], telinfo=target_img.telinfo, load=True)
            mdark = MasterImage(path=mdark_path['file'], telinfo=target_img.telinfo, load=True)
            mflat = MasterImage(path=mflat_path['file'], telinfo=target_img.telinfo, load=True)

            _ = Preprocessor.correct_bdf(
                target_img,
                bias_image=mbias,
                dark_image=mdark,
                flat_image=mflat,
                save=True
            )
        except Exception as e:
            print(f"Error processing {target_img}: {e}")

# Main entry
if __name__ == '__main__':
    tile_ids = ["T08031", 'T08033', 'T07803', 'T07579']
    filter_list = ['g', 'r', 'i']

    all_args = list(itertools.product(tile_ids, filter_list))
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_tile_filter, all_args), total=len(all_args)))
        

# #%% Data loadingfrom astropy.table import Table

# tile_ids = ["T07804", "T08032", "T07805", "T08031", 'T08033', 'T07803', 'T07579']
# databrowser = context['Databrowser']
# all_imgcount = dict()

# for tile_id in tile_ids:
#     databrowser.objname = tile_id
#     all_path = databrowser.search('calib*202507*100.fits')
#     all_imgcount[tile_id] = {filt: len(paths) for filt, paths in all_path.items()}

# # Get all unique filters
# all_filters = sorted({filt for d in all_imgcount.values() for filt in d})

# # Build data rows
# rows = []
# for tile_id in sorted(all_imgcount):
#     row = [tile_id] + [all_imgcount[tile_id].get(f, 0) for f in all_filters]
#     rows.append(row)

# # Create table
# colnames = ['tile_id'] + all_filters
# tbl = Table(rows=rows, names=colnames)

# # Add total column
# tbl['Total'] = [sum(row[1:]) for row in tbl]

# # Add total row
# #total_row = ['Total'] + [sum(tbl[f]) for f in all_filters] + [sum(tbl['Total'])]
# #tbl.add_row(total_row)

# tbl.pprint(max_width=-1)
# # #%% Reference Image preparation (RIS)
# # for tile_id in tile_ids:
# #     databrowser.objname = tile_id
# #     target_imglist = databrowser.search('calib*.com.fits', 'science')
# #     if len(target_imglist) == 0:
# #         print(f"[WARNING] No images found for tile {tile_id}. Skipping.")
# #         continue
# #     else:
# #         for target_img in target_imglist:
# #             if target_img.filter in ['g', 'r', 'i']:
# #                 target_img = target_img.to_referenceimage()
# #                 target_img.register()
# #%% Reference Image preparation (HIPS2FITS)
from ezphot.utils import ImageQuerier
from ezphot.utils import Tiles
tile_ids = ['T08030', 'T07580', 'T07577', 'T07802', 'T08034', 'T07356', 'T07355', 'T08264', 'T08495', 'T08494', 'T08260', 'T08496', 'T07354', 'T07581', 'T08029', 'T07357', 'T07807', 'T08493']

tile = Tiles()
helper = Helper()
image_querier = ImageQuerier()
telinfo = helper.get_telinfo('7DT', 'C361K', 'HIGH', 1)
all_coverage = defaultdict(list)
from ezphot.utils.imagequerier import ImageQuerier  # assuming your file

def run_single_query(catalog, width, height, ra, dec, pixelscale, telinfo, objname, rotation_angle):
    image_querier = ImageQuerier(catalog_key=catalog)
    return image_querier.query(
        width=width,
        height=height,
        ra=ra,
        dec=dec,
        pixelscale=pixelscale,
        save_path=None,
        telinfo=telinfo,
        objname=objname,
        rotation_angle=rotation_angle,
        verbose=True
    )
    
import time
for tile_id in tile_ids:
    tile_info = tile.get_tile_info(tile_id)
    ra = tile_info['ra'][0]
    dec = tile_info['dec'][0]
    radius_deg = 1.35/2
    width = 9576
    height = 6388
    pixelscale = 0.505
    rotation_angle = 0

    for catalog in ['SkyMapper/SMSS4/g', 'SkyMapper/SMSS4/r', 'SkyMapper/SMSS4/i', 'DSS/DSS2/r', 'DSS/DSS2/b']:
        stack_image = run_single_query(catalog, width, height, ra, dec, pixelscale, telinfo, tile_id, 0)
        stack_image.register()
        time.sleep(10)

# #%% Coverage check
# import pandas as pd
# # Convert to DataFrame
# df = pd.DataFrame.from_dict(all_coverage, orient='index')
# df.reset_index(inplace=True)
# df.rename(columns={'index': 'tile'}, inplace=True)

# # Fill missing values with False
# df = df.fillna(False)

# # Add 'volume' column: count of True per row
# df['volume'] = df.drop(columns='tile').sum(axis=1)

# # Create total row: sum of True per column
# totals = df.drop(columns=['tile']).sum(numeric_only=True)
# totals['tile'] = 'total'
# df_total = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

# # Show result
# from IPython.display import display
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# display(df_total)

# # Optional: convert to Astropy Table
# from astropy.table import Table
# coverage_table = Table.from_pandas(df_total)
# coverage_table.pprint(max_width=-1)
# coverage_table['Total'] = [sum(row[1:]) for row in coverage_table]
#%% main Process

tile_ids = ["T07804", "T08032", "T07805", "T08031", 'T08033', 'T07803', 'T07579', 'T08262', 'T07578', 'T08263', 'T08261', 'T07806']
tile_ids = ["T07804", "T08032", "T07805", "T08031"]
tile_ids = ['T08033', 'T07803', 'T07579']
tile_ids = ['T08262', 'T07578', 'T08263', 'T08261', 'T07806']
tile_ids = ['T08030', 'T07580', 'T07577', 'T07802', 'T08034', 'T07356', 'T07355', 'T08264', 'T08495', 'T08494', 'T08260', 'T08496', 'T07354', 'T07581', 'T08029', 'T07357', 'T07807', 'T08493']

for tile_id in tile_ids:
    # Load the data
    sdtdata = context['sdt_data_querier']
    sdtdata.sync_scidata(targetname = tile_id)

    # start = time.time()
    
    # # Define the image processing function
    # # 76 images -> 9min 41s
    # def imgprocess(target_img):
    #     mbias_path = context['Preprocessor'].get_masterframe_from_image(target_img, imagetyp = 'BIAS', max_days = 60)[0]
    #     mdark_path = context['Preprocessor'].get_masterframe_from_image(target_img, imagetyp = 'DARK', max_days = 60)[0]
    #     mflat_path = context['Preprocessor'].get_masterframe_from_image(target_img, imagetyp = 'FLAT', max_days = 60)[0]
    #     mbias = MasterImage(path = mbias_path['file'], telinfo = telinfo, load = True)
    #     mdark = MasterImage(path = mdark_path['file'], telinfo = telinfo, load = True)
    #     mflat = MasterImage(path = mflat_path['file'], telinfo = telinfo, load = True)

    #     status = dict()
    #     status['image'] = target_img.path
    #     status['platesolve'] = None
    #     if config['do_platesolve']:
    #         try:
    #             target_img = context['PlateSolver'].solve_astrometry(
    #                 target_img=target_img,
    #                 **config['platesolve_kwargs'],
    #             )
    #             target_img = context['PlateSolver'].solve_scamp(
    #                 target_img=target_img,
    #                 scamp_sexparams=None,
    #                 scamp_params=None,
    #                 **config['platesolve_kwargs'],
    #             )[0]
    #             status['platesolve'] = True
    #         except Exception as e:
    #             status['platesolve'] = e

    #     status['mask'] = None
    #     target_srcmask = None
    #     if config['do_generatemask']:
    #         try:
    #             # Mask the object frames
    #             target_srcmask = context['MaskGenerator'].mask_sources(
    #                 target_img = target_img,
    #                 target_mask = None,
    #                 **config['mask_kwargs'],
    #             )
    #             status['mask'] = True
    #         except:
    #             status['mask'] = e

    #     status['circular_mask'] = None
    #     if config['do_circularmask']:
    #         try:
    #             # Generate the circular mask
    #             target_srcmask = context['MaskGenerator'].mask_circle(
    #                 target_img = target_img,
    #                 target_mask = target_srcmask,
    #                 mask_type = 'source',
    #                 **config['circlemask_kwargs'],
    #             )
    #             status['circular_mask'] = True

    #         except Exception as e:
    #             status['circular_mask'] = e

    #     status['object_mask'] = None
    #     target_objectmask = None
    #     if config['do_objectmask']:
    #         try:
    #             # Generate the circular mask
    #             target_objectmask = context['MaskGenerator'].mask_circle(
    #                 target_img = target_img,
    #                 target_mask = None,
    #                 mask_type = 'invalid',
    #                 **config['objectmask_kwargs'],
    #             )
    #             status['object_mask'] = True

    #         except Exception as e:
    #             status['object_mask'] = e

    #     status['background'] = None
    #     target_bkg = None
    #     if config['do_generatebkg']:
    #         try:
    #             # Generate the background
    #             target_bkg, bkg_instance = context['BkgGenerator'].estimate_with_sep(
    #                 target_img = target_img,
    #                 target_mask = target_srcmask,
    #                 **config['bkg_kwargs'],
    #             )
    #             status['background'] = True
    #         except Exception as e:
    #             status['background'] = e

    #     status['errormap'] = None
    #     target_bkgrms = None
    #     bkg_instance = None
    #     if config['do_generateerrormap']:
    #         try:
    #             if config['errormap_from_propagation']:
    #                 if target_bkg is None:
    #                     target_bkg, bkg_instance = context['BkgGenerator'].estimate_with_sep(
    #                         target_img = target_img,
    #                         target_mask = target_srcmask,
    #                         **config['bkg_kwargs'],
    #                     )
    #                 target_bkgrms = context['ErrormapGenerator'].calculate_bkgrms_from_propagation(
    #                     target_bkg = target_bkg,
    #                     egain = target_img.egain,
    #                     readout_noise = target_img.telinfo['readnoise'],
    #                     mbias_img = mbias,
    #                     mdark_img = mdark,
    #                     mflat_img = mflat,
    #                     **config['errormap_kwargs'],
    #                 )
    #                 status['errormap'] = True
    #         except Exception as e:
    #             status['errormap'] = e

    #     sex_catalog = None
    #     if config['do_aperturephotometry']:
    #         try:
    #             # Perform aperture photometry
    #             sex_catalog = context['AperturePhotometry'].sex_photometry(
    #                 target_img = target_img,
    #                 target_bkg = target_bkg,
    #                 target_bkgrms = target_bkgrms,
    #                 target_mask = target_objectmask, ################################ HERE, Central stars are masked for ZP caclculation
    #                 **config['aperturephotometry_kwargs'],
    #             )
    #             status['aperture_photometry'] = True
    #         except Exception as e:
    #             status['aperture_photometry'] = e

    #     calib_catalog = None
    #     if config['do_photometric_calibration'] and config['do_aperturephotometry']:
    #         try:
    #             target_img, calib_catalog, filtered_catalog = context['PhotometricCalibration'].photometric_calibration(
    #                 target_img = target_img,
    #                 target_catalog = sex_catalog,
    #                 **config['photometriccalibration_kwargs'],
    #             )
    #             status['photometric_calibration'] = True
    #         except Exception as e:
    #             status['photometric_calibration'] = e

    #     # Clean up memory
    #     del target_srcmask, target_objectmask
    #     del mbias, mdark, mflat
    #     del target_img
    #     del target_bkg
    #     del bkg_instance
    #     del target_bkgrms
    #     del calib_catalog
    #     del sex_catalog
    #     del filtered_catalog
    #     gc.collect()
    #     return status
    
    # from concurrent.futures import ProcessPoolExecutor, as_completed
    # import gc
    # from tqdm import tqdm

    # start = time.time()
    # def chunk_list(lst, chunk_size):
    #     """Yield successive chunk_size-sized chunks from lst."""
    #     for i in range(0, len(lst), chunk_size):
    #         yield lst[i:i + chunk_size]

    # def process_batch(batch_images, batch_index, max_workers=8):
    #     print(f"\nStarting batch {batch_index+1} with {len(batch_images)} images...")

    #     results = []
    #     with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #         futures = [executor.submit(imgprocess, img) for img in batch_images]

    #         for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_index+1}"):
    #             try:
    #                 result = future.result()
    #                 if result is not None:
    #                     results.append(result)
    #             except Exception as e:
    #                 print(f"[ERROR in batch {batch_index+1}] {e}")
        
    #     # Clean up memory between batches
    #     gc.collect()
    #     return results

    # # ? Main loop over batches
    # batch_size = 8
    # all_results = []

    # for batch_index, batch in enumerate(chunk_list(target_imglist, batch_size)):
    #     batch_results = process_batch(batch, batch_index, max_workers=8)
    #     all_results.extend(batch_results)
    
    # single_procee_time = (time.time() - start)

    # start = time.time()
    # # Stack the images
    # Databrowser.objname = tile_id
    # target_imginfo = Databrowser.search(
    #     pattern = 'calib*202507*100.fits',
    #     return_type = 'imginfo'
    # )
    # if len(target_imginfo) == 0:
    #     print(f"[WARNING] No images found for tile {tile_id}. Skipping.")
    #     continue
    
    # target_imginfo_groups = target_imginfo.group_by(['filter', 'telescop']).groups
    # print('Number of groups:', len(target_imginfo_groups))
    # stack_imglist = []
    # stack_bkgrmslist = []
    # failed_imglist = []
    # for target_imginfo_group in target_imginfo_groups:
    #     target_imginfo_group = context['helper'].group_table(target_imginfo_group, 'mjd', 0.3)
    #     target_imginfo_subgroups = target_imginfo_group.group_by('group').groups
    #     for target_imginfo_subgroup in target_imginfo_subgroups:
    #         target_imglist = [ScienceImage(path=row['file'], telinfo=telinfo, load=True) for row in target_imginfo_subgroup]
    #         target_imglist = context['Stacking'].select_quality_images(target_imglist, seeing_limit = 6, depth_limit = 15, ellipticity_limit = 0.6, max_numbers = len(target_imglist))
    #         if len(target_imglist) == 0:
    #             print(f"[WARNING] No images found. skipping stacking.")
    #             continue
    #         target_bkglist = [Background(path = target_img.savepath.bkgpath, load=True) for target_img in target_imglist]
    #         target_bkgrmslist = [Errormap(path = target_img.savepath.bkgrmspath, emaptype = 'bkgrms', load=True) for target_img in target_imglist]

    #         try:
    #             stack_img, stack_bkgrms = context['Stacking'].stack_multiprocess(
    #                 target_imglist = target_imglist,
    #                 target_bkglist = target_bkglist,
    #                 target_bkgrmslist = target_bkgrmslist,
    #                 target_outpath = None,
    #                 bkgrms_outpath = None,
    #                 **config['stacking_kwargs'],
    #             )
    #             # Clean up memory
    #             for target_img in target_imglist:
    #                 target_img.data = None
    #             for target_bkg in target_bkglist:
    #                 target_bkg.data = None
    #             for target_bkgrms in target_bkgrmslist:
    #                 target_bkgrms.data = None
    #             stack_img.data = None
    #             stack_bkgrms.data = None
                    
    #             stack_imglist.append(stack_img)
    #             stack_bkgrmslist.append(stack_bkgrms)
    #         except:
    #             print(f"[ERROR] Stacking failed, skipping stacking.")
    #             failed_imglist.extend(target_imglist)
    #             continue
    
    # print(f"Stacking completed in {time.time() - start:.2f} seconds.")
            
    # # Stack the images
    # Databrowser.objname = tile_id
    # Databrowser.filter = 'r'
    # stacked_imglist = Databrowser.search(
    #     pattern = 'calib*100.com.fits',
    #     return_type = 'science'
    # )
    # if len(stacked_imglist) == 0:
    #     print(f"[WARNING] No images found for tile {tile_id}. Skipping.")
    #     continue

    # def stackprocess(stacked_img):
    #     stack_bkgrms = Errormap(path=stacked_img.savepath.bkgrmspath, emaptype='bkgrms', load=True)
    #     if not stack_bkgrms.is_exists:
    #         print(f"[WARNING] Background RMS file not found: {stack_bkgrms.path}")
    #         stack_bkgrms = None
            
    #     status = dict()
    #     try:
    #         # Perform aperture photometry
    #         sex_catalog = context['AperturePhotometry'].sex_photometry(
    #             target_img = stacked_img,
    #             target_bkg = None,
    #             target_bkgrms = stack_bkgrms,
    #             target_mask = None, ################################ HERE, Central stars are masked for ZP caclculation
    #             **config['stack_aperturephotometry_kwargs'],
    #         )
    #         status['aperture_photometry'] = True
    #     except Exception as e:
    #         status['aperture_photometry'] = e

    #     try:
    #         stacked_img, calib_catalog, filtered_catalog = context['PhotometricCalibration'].photometric_calibration(
    #             target_img = stacked_img,
    #             target_catalog = sex_catalog,
    #             **config['stack_photometriccalibration_kwargs']
    #         )
    #         status['photometric_calibration'] = True
    #     except Exception as e:
    #         status['photometric_calibration'] = e
            
    #     # Clean up memory
    #     del stacked_img
    #     del stack_bkgrms
    #     del calib_catalog
    #     del filtered_catalog
    #     gc.collect()
    #     return status

    # with ProcessPoolExecutor(max_workers=10) as executor:
    #     futures = [executor.submit(stackprocess, stacked_img) for stacked_img in stacked_imglist]
    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         try:
    #             result = future.result()
    #         except Exception as e:
    #             print(f"[ERROR] {e}")
                
    # Subtract the images
    Databrowser = context['Databrowser'].__class__('scidata')
    Databrowser.objname = tile_id
    Databrowser.filter = 'r'
    stacked_imgset = Databrowser.search(
        pattern = 'calib*com.fits',
        return_type = 'science'
    )        
    stacked_imglist = stacked_imgset.target_images
    # if len(stacked_imglist) == 0:
    #     print(f"[WARNING] No images found for tile {tile_id}. Skipping.")
    #     continue
    stacked_imglist = [stack_img for stack_img in stacked_imglist if stack_img.filter in ['g', 'r', 'i']]
        
    def subtract_process(stacked_img):
        reference_img = context['DIA'].get_referenceframe_from_image(stacked_img)[0]
        #reference_img = context['DIA'].get_referenceframe_from_image(target_img = stacked_img, telname = 'SkyMapper')[0]
        result = context['DIA'].find_transients(
            target_img = stacked_img, 
            reference_imglist = [reference_img],
            target_bkg = None,
            **config['DIA_kwargs'])
        
        del reference_img
        del stacked_img
        return result

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(subtract_process, stacked_img) for stacked_img in stacked_imglist]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
            except Exception as e:
                print(f"[ERROR] {e}")

# %% Check transaients 
from ezphot.imageobjects import ScienceImage
from ezphot.catalog import Catalog
Databrowser = context['Databrowser'].__class__('scidata')
#tile_id = 'T08261'
Databrowser.objname = tile_id
Databrowser.filter = 'r'
stacked_imgset = Databrowser.search(
    pattern = 'sub*.fits',
    return_type = 'science'
)        
sub_img = stacked_imgset.target_images[3]
ref_img = ScienceImage(sub_img.savedir / sub_img.savepath.savepath.name.replace('sub_', 'ref_'), load = True)
sci_img = ScienceImage(sub_img.savedir / sub_img.savepath.savepath.name.replace('sub_', 'sci_'), load = True)
cat_all = sub_img.catalog
cat_candidate = Catalog(sub_img.savepath.catalogpath.with_suffix('.candidate'), load = True)
cat_transient = Catalog(sub_img.savepath.catalogpath.with_suffix('.transient'), load = True)
# %%

DIA = context['DIA']

# %%
print('# of detected sources:')
print(len(cat_all.data))
print('# of candidate sources:')
print(len(cat_candidate.data))
print('# of transient sources:')
print(len(cat_transient.data))
# %%
print(sci_img.seeing, ref_img.seeing)
print(sci_img.depth, ref_img.depth)
# %%
alls = cat_all.data
candidates = cat_candidate.data
transients = cat_transient.data
x_list = transients['X_WORLD'][:100]
y_list = transients['Y_WORLD'][:100]
DIA.show_transient_positions(
    sci_img,
    ref_img,
    sub_img,
    x_list,
    y_list
)
# %%
ImageSet([sci_img, ref_img, sub_img]).run_ds9()
#%%
cat_transient.select_sources(
    x = 241.623970,
    y = -70.327119
)
# %%
sci_img.depth
# %%
cat_transient.target_data['MAGSKY_AUTO']
# %%
cat_transient
# %%
