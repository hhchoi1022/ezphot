

#%%
from ezphot.methods import *
from ezphot.imageobjects import *
from ezphot.helper import Helper
from ezphot.utils import DataBrowser
from ezphot.skycatalog import *
import psutil, os
from pympler import asizeof

from tqdm import tqdm
import gc
import time
import numpy as np
#%%
tile_ids = [
    # 'T04071',
    # 'T05354',
    # 'T08344',
    # 'T15265',
    # 'T14988',
    # 'T16088',
    # 'T19432',
    # 'T19150',
    # 'T16205',
    # 'T16484',
    # 'T20116',
    # 'T20675',
    'T22064',
    'T20118',
    'T20397',
    'T22066']
    
tile_ids_too = [
    'T03751',
    'T03573',
    'T02955',
    'T04069',
    'T06782',
    'T08346',
    'T06569'
]
transient_coordinate = dict()
transient_coordinate['T03751'] = [(73.858894, -53.162114)]
transient_coordinate['T03573'] = [(46.494429, -54.151839)]
transient_coordinate['T02955'] = [(68.868971, -57.475672)]
transient_coordinate['T04069'] = [(26.942508, -51.823756)]
transient_coordinate['T06782'] = [(7.382421, -39.514339)]
transient_coordinate['T08346'] = [(4.5396222202987, -33.854283871522)]
transient_coordinate['T06569'] = [(7.5357660984872, -41.078709318686)]

event_start_date = '2025-11-10'
#%%
tile = 'T20397'
ra = None
dec = None
# if tile in transient_coordinate.keys():
#     ra = transient_coordinate[tile][0][0]
#     dec = transient_coordinate[tile][0][1]
    
db = DataBrowser('scidata')
db.observatory = '7DT'
db.objname = tile
# db.filter = 'g'
target_images = db.search(pattern = 'calib*com.fits', return_type = 'path')
print(len(target_images))
target_catalogs = db.search(pattern = 'sub*_r_*transient', return_type = 'catalog')
for catalog in target_catalogs.catalogs:
    catalog.data
if ra is not None and dec is not None:
    target_catalogs.select_sources(ra = ra, dec = dec, radius = 10)
# %%
target_catalogs.catalogs
#%%
merged_catalog, metadata = target_catalogs.merge_catalogs(
    max_distance_arcsec = 3
)
#%%
merged_catalog.sort('n_detections', reverse=True)
# %%
merged_catalog_with_high_detections = merged_catalog[merged_catalog['n_detections'] > 0]
print(len(merged_catalog_with_high_detections))
#%%
from ezphot.methods import Subtract
subtract = Subtract()
#%%
sub_img = target_catalogs.catalogs[0].target_img
sub_img.show()
#%%
sci_img = ScienceImage(str(sub_img.path).replace('sub_', 'sci_'))
ref_img = ScienceImage(str(sub_img.path).replace('sub_', 'ref_'))
#%%
subtract.show_transient_positions(
    sci_img,
    ref_img,
    sub_img,
    merged_catalog_with_high_detections['ra'],
    merged_catalog_with_high_detections['dec'],
    np.arange(len(merged_catalog_with_high_detections)),
    save = False,
    ncols = 3,
)
#%%

# %%
from astropy.coordinates import SkyCoord
import astropy.units as u
import re
coord_str = '094540.93_+043752.4'
is_negative = '-' in coord_str
numbers = re.findall(r'\d+', coord_str)
ra_hour, ra_minute, ra_second, ra_millisecond = numbers[0][:2], numbers[0][2:4], numbers[0][4:], numbers[1]
dec_degree, dec_minute, dec_second, dec_millisecond = numbers[2][:2], numbers[2][2:4], numbers[2][4:], numbers[3]
ra_str = f'{ra_hour}:{ra_minute}:{ra_second}.{ra_millisecond}'
dec_str = f'{dec_degree}:{dec_minute}:{dec_second}.{dec_millisecond}'
if is_negative:
    dec_str = '-' + dec_str
coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
ra_deg = coord.ra.deg
dec_deg = coord.dec.deg
from ezphot.utils import CatalogQuerier
catalog_querier = CatalogQuerier('SKYBOT')
#%%
from astropy.time import Time
obs_time = Time(sci_img.obsdate)
result = catalog_querier.query(coord = SkyCoord(ra = ra_deg, dec = dec_deg, unit = (u.deg, u.deg)), epoch = obs_time, radius_arcsec = 100)
# %%
target_catalogs.select_sources(ra = ra_deg, dec = dec_deg, radius = 5)
#%%
target_catalogs.catalogs
# %%
target = target_catalogs.catalogs[0].target_data
print(target['X_WORLD'], target['Y_WORLD'])
# %%
SkyCoord(ra = target['X_WORLD'], dec = target['Y_WORLD'], unit = (u.deg, u.deg)).to_string('hmsdms')
# %%
sci_img.show_position(ra_deg, dec_deg, coord_type = 'coord', zoom_radius_pixel = 100)
ref_img.show_position(ra_deg, dec_deg, coord_type = 'coord', zoom_radius_pixel = 100)
sub_img.show_position(ra_deg, dec_deg, coord_type = 'coord', zoom_radius_pixel = 100)
# %%

#%%
tile = 'T19432'
ra_deg = 343.33782698 # For T19432
dec_deg = 0.87583172503 # For T19432
# ra_deg = 343.38083298
# dec_deg = 0.8070928774
# tile = 'T16088'
# ra_deg = 345.4193976
# dec_deg = -9.453043098
# ra = None
# dec = None
# tile = 'T14988'
# ra_deg = 346.60113677 # For T14988
# dec_deg = -12.301824877 # For T14988
# tile = 'T08344'
# ra_deg = 1.778651774
# dec_deg = -33.89865595

tile = 'T19150'
ra_deg = 341.08216894
dec_deg = -0.02798500009
# tile = 'T03751'
# ra_deg = 73.858894
# dec_deg = -53.162114
# tile = 'T03573'
# ra_deg = 46.494429
# dec_deg = -54.151839
# tile = 'T02955'
# ra_deg = 68.868971
# dec_deg = -57.475672
# tile = 'T16484'
# ra_deg = 139.7411944
# dec_deg = -7.564863905
tile = 'T20116'
ra_deg = 143.95655
dec_deg = 3.20963157
# ra_deg = 145.0401539 # QSO
# dec_deg = 3.51171597 # QSO
# tile = 'T08346'
# ra_deg = 4.5396222202987
# dec_deg = -33.854283871522
# tile = 'T06569'
# ra_deg = 7.5357660984872
# dec_deg = -41.078709318686
#%%
db = DataBrowser('scidata')
db.observatory = '7DT'
db.objname = tile
target_images = db.search(pattern = 'calib*com.fits', return_type = 'path')
print(len(target_images))
target_catalogs = db.search(pattern = 'calib*fits.cat', return_type = 'catalog')
# target_catalogs = db.search(pattern = 'sub*cat', return_type = 'catalog')

target_catalogs.select_catalogs(obs_start = '2025-11-10')
for catalog in target_catalogs.catalogs:
    catalog.data
if ra_deg is not None and dec_deg is not None:
    target_catalogs.select_sources(x = ra_deg, y = dec_deg, unit = 'coord', matching_radius = 20)
#%%
from ezphot.dataobjects import PhotometricSpectrum
photspectrum = PhotometricSpectrum(target_catalogs)
photspectrum.plt_params.figure_figsize = (8, 6)
#photspectrum.plt_params.ylim = [23, 18]
#photspectrum.plt_params.ylim = [19, 14.5]
photspectrum.plt_params.ylim = [22, 16]
photspectrum.plt_params.line_style = 'none'
photspectrum.plot(ra_deg, dec_deg, flux_key = 'MAGSKY_APER_1', fluxerr_key = 'MAGERR_APER_1')

# %%
from ezphot.utils import CatalogQuerier
catalog_querier = CatalogQuerier('SKYBOT')
#%%
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
result = catalog_querier.query(coord = SkyCoord(ra = ra_deg, dec = dec_deg, unit = (u.deg, u.deg)), epoch = Time(catalog.target_img.obsdate), radius_arcsec = 100)
# %%