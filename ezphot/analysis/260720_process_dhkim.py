

#%%
from bridge.connector import GWPortalConnector
#%%
gwconnector = GWPortalConnector('processed')
tbl_all = gwconnector.query(since_days=10)
# %%
from astropy.table import Table
tbl_target_path = '/home/hhchoi1022/RQ_tile_filter_availability_by_ID.csv'
tbl_target = Table.read(tbl_target_path)
# %%

tbl_all.colnames
# %%
tbl_row = tbl_target[0]
#%%
ra = tbl_row['RA']
dec = tbl_row['DEC']
radius = tbl_row['Radius']
#%%
from shapely.geometry import Polygon
from shapely.geometry import Point

for tbl_row in tbl_target:
    cols = ['g', 'r', 'i', 'z', 'm400', 'm425', 'm450', 'm475', 'm500', 'm525', 'm550', 'm575', 'm600', 'm625', 'm650', 'm675', 'm700', 'm725', 'm750', 'm775', 'm800', 'm825', 'm850', 'm875']
    tot_images = sum([tbl_row[col] for col in cols])
    ra = tbl_row['RA']
    dec = tbl_row['DEC']
    tbl_result = gwconnector.query(ra=ra, dec=dec, radius=0.6)
    num_total = len(tbl_result)
    num_images = 0
    for row in tbl_result:
        poly = Polygon(row['vertices']['coordinates'][0])
        if poly.contains(Point(ra, dec)):
            num_images += 1
        filepath = row['filepath']
        target_img = ScienceImage(filepath)
    print(f"{ra} {dec} {num_images} / {num_total} [{tot_images}]")
    id_ = tbl_row['id']
    path = tbl_row['path']
    
# %%

from ezphot.imageobjects import ScienceImage
# %%
