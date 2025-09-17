#%%
S250830bp_TILES = ['T00528', 'T00464', 'T00465', 'T00527', 'T00529', 'T00595', 'T00463', 'T00596', 'T00405']
# %%
from ezphot.utils import SDTDataQuerier
import time
sdtdataquerier = SDTDataQuerier()
for tile in S250830bp_TILES:
    sdtdataquerier.sync_scidata(targetname = tile)
    time.sleep(10)
# %%
from ezphot.methods import AperturePhotometry
from ezphot.methods import PhotometricCalibration
aperphot = AperturePhotometry()
photcal = PhotometricCalibration()
for i in [0,1,2,3,4,5,6,7,8]:
    from ezphot.utils import DataBrowser
    databrowser = DataBrowser('scidata')
    databrowser.observatory = '7DT'
    databrowser.objname = S250830bp_TILES[i]
    target_imgset = databrowser.search(pattern = 'calib*100.com.fits', return_type = 'science')
    target_imglist = target_imgset.target_images

    def process_image(target_img):
        # target_srcmask = target_img.calculate_circularmask(
        #     x_position = 325.37, 
        #     y_position = -77.392, 
        #     radius_arcsec = 250,
        #     verbose = False,
        #     visualize = False,
        #     save = False)
        # target_srcmask = None
        # target_srcmask = target_img.calculate_sourcemask(
        #     target_srcmask = target_srcmask,
        #     verbose = False,
        #     visualize=  False,
        #     save = False)
        # target_bkg = target_img.calculate_bkg(
        #     target_srcmask = target_srcmask, 
        #     verbose = False,
        #     visualize = False,
        #     save = True)
        # target_bkgrms = target_img.calculate_bkgrms(
        #     target_srcmask = target_srcmask, 
        #     verbose = False,
        #     visualize = False,
        #     save = True)
        target_catalog = aperphot.sex_photometry(
            target_img = target_img,
            target_bkg = target_img.bkgmap,
            target_bkgrms = target_img.bkgrms,
            verbose = False,
            visualize = False,
            save = True,
            save_fig = False
        )
        photcal.photometric_calibration(
            target_img = target_img,
            target_catalog = target_catalog,
            dynamic_mag_range = False,
            verbose = False,
            visualize = False,
            save = True,
            save_fig = False
        )
        
        target_img.clear(verbose = False)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    # Run with multiprocess with tqdm (decs = 'Processing...')
    with tqdm(total = len(target_imglist), desc = 'Processing...') as pbar:
        with ProcessPoolExecutor(max_workers = 10) as executor:
            futures = [executor.submit(process_image, target_img) for target_img in target_imglist]
            for future in as_completed(futures):
                pbar.update(1)

# %%



#%%
from ezphot.helper import Helper
from ezphot.imageobjects import ScienceImage, ImageSet
from ezphot.methods import Stack
helper = Helper()
stack = Stack()
for tile in S250830bp_TILES:
    databrowser = DataBrowser('scidata')
    databrowser.observatory = '7DT'
    databrowser.objname = tile
    target_imginfo = databrowser.search(pattern = 'calib*100.fits', return_type = 'imginfo')
    imginfo_telescope_groups = target_imginfo.group_by(['telescop', 'filter']).groups
    for imginfo_telescope in imginfo_telescope_groups:
        imginfo_telescope = helper.group_table(imginfo_telescope, 'jd', 0.3)
        imginfo_obsdate_groups = imginfo_telescope.group_by('group').groups
        for imginfo_obsdate in imginfo_obsdate_groups:
            target_imglist = [ScienceImage(imginfo_obsdate['file']) for imginfo_obsdate in imginfo_obsdate]
            target_imglist_new = []
            for target_img in target_imglist:
                if 'ZP_APER_1' in target_img.header.keys():
                    target_imglist_new.append(target_img)
            if len(target_imglist_new) > 0:
                target_imglist = target_imglist_new
                target_imgset = ImageSet(target_imglist)
                target_bkglist = target_imgset.bkgmap
                target_bkgrmslist = target_imgset.bkgrms
                stack.stack_multiprocess(
                    target_imglist,
                    target_bkglist,
                    target_bkgrmslist,
                    save = True
                )
            
                for target_img in target_imglist:
                    target_img.clear()
                for target_bkg in target_bkglist:
                    target_bkg.clear()
                for target_bkgrms in target_bkgrmslist:
                    target_bkgrms.clear()
    
   
# %%
from ezphot.utils import DataBrowser
tile = S250830bp_TILES[1]
databrowser = DataBrowser('scidata')
databrowser.observatory = '7DT'
databrowser.objname = tile
databrowser.filter = 'r'
target_imgset = databrowser.search(pattern = 'calib*100.com.fits', return_type = 'science')
target_imglist = target_imgset.target_images
# %%
for i,target_img in enumerate(target_imglist):
    print('INDEX:', i)
    print('PATH:', target_img.path.name)
    print('OBSDATE:', target_img.obsdate)
    print('DEPTH: ', target_img.depth)
    print('SEEING: ', target_img.seeing)
    print('EXPTIME: ', target_img.exptime)
#%%
#%%

target_imglist[0].to_referenceimage().register()
# %%

from ezphot.utils import ImageQuerier
imgquerier = ImageQuerier(catalog_key = 'SkyMapper/SMSS4/r')
#%%
import numpy as np
ref_img = imgquerier.query(
    width = target_imglist[0].naxis1,
    height = target_imglist[0].naxis2,
    ra = target_imglist[0].center['ra'],
    dec = target_imglist[0].center['dec'],
    pixelscale = np.mean(target_imglist[0].pixelscale),
    telinfo = target_imglist[0].telinfo,
    save_path = None,
    objname = target_imglist[0].objname,
)
#%%
#%%
from ezphot.methods import Subtract
subtract = Subtract()
#%%
tile = S250830bp_TILES[3]
databrowser = DataBrowser('scidata')
databrowser.observatory = '7DT'
databrowser.objname = tile
databrowser.filter = 'r'
target_imgset = databrowser.search(pattern = 'calib*100.com.fits', return_type = 'science')
target_imgset.select_images(obs_start = '2025-08-31')
target_imglist = target_imgset.target_images
ref_img = subtract.get_referenceframe_from_image(target_imglist[0])[0]
# %%
for target_img in target_imglist:
    result = subtract.find_transients(
        target_img = target_img,
        reference_imglist = [ref_img],
        target_bkg = None,
        detection_sigma = 3,
        aperture_diameter_arcsec = [5, 7, 10],
        aperture_diameter_seeing = [3.5, 4.5],
        kron_factor = 2.5,
        catalog_type = 'GAIAXP',
        target_transient_number = 5,
        reject_variable_sources = True,
        negative_detection = True,
        reverse_subtraction = False,
        save = True,gusgh1020
        verbose = False,
        visualize = True,
        save_transient_figure = True,
        save_candidate_figure = True,
        show_transient_numbers = 100,
        show_candidate_numbers = 100,
        iu = 60000,
        il = -10000,
        tu = 60000,
        tl = -10000,
        nrx = 1,
        nry = 1,
        nsx = 10,
        nsy = 10,
        ko = 3,
        bgo = 1,
        r = 10,
)
# %%
target_imglist[1].show()
# %%
