#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#   Library
import numpy as np
from astropy.table import Table
import requests
import time
import os
import sys
import glob
import pandas as pd
import itertools

from io import StringIO
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS

import matplotlib.pyplot as plt
plt.ioff()
# InteractiveShell.ast_node_interactivity = "last_expr"
import matplotlib as mpl
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')
# In[ ]:
#   Function
# downloading images from ps1
# tra, tdec:center coordinate of the image(in degrees)
# size: image pixel size, prefered less than 6000 pix
# filters: needed filters (grizy)
def getimages(tra, tdec, size=240, filters="grizy", format="fits", imagetypes="stack"):
    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    if not isinstance(imagetypes, str):
        imagetypes = ",".join(imagetypes)
    cbuf = StringIO()
    cbuf.write('\n'.join(["{} {}".format(ra, dec) for (ra, dec) in zip(tra, tdec)]))
    cbuf.seek(0)
    r = requests.post(ps1filename, data=dict(filters=filters, type=imagetypes),
                      files=dict(file=cbuf))
    r.raise_for_status()
    tab = Table.read(r.text, format="ascii")

    urlbase = "{}?size={}&format={}".format(fitscut, size, format)
    tab["url"] = ["{}&ra={}&dec={}&red={}".format(urlbase, ra, dec, filename)
                  for (filename, ra, dec) in zip(tab["filename"], tab["ra"], tab["dec"])]
    return tab

# When downloading images, coordinates in degrees is needed, and when using Swarp, coordinates in HMS DMS is needed
# If both format is prepared, no need of this code
# Just converting coordinate format
def degrees_to_hms_dms(ra_in_degrees, dec_in_degrees):
    """
    Convert RA in degrees to HMS (hours, minutes, seconds) and 
    Dec in degrees to DMS (degrees, minutes, seconds).
    
    Parameters:
    ra_in_degrees (float): RA in degrees.
    dec_in_degrees (float): Dec in degrees.

    Returns:
    tuple: The RA in HMS format (HH:MM:SS) and Dec in DMS format (±DD:MM:SS.ssss).
    """
    coord = SkyCoord(ra=ra_in_degrees * u.deg, dec=dec_in_degrees * u.deg, frame='icrs')
    
    ra_hms = coord.ra.to_string(unit=u.hourangle, sep=':', pad=True, precision=0)  # HH:MM:SS
    dec_dms = coord.dec.to_string(unit=u.degree, sep=':', alwayssign=True, precision=4)  # ±DD:MM:SS.ssss
    
    return ra_hms, dec_dms



def plot_rectangle_on_sky(center_ra, center_dec, x_size, y_size, pixel_scale):
    """
    주어진 중심 좌표와 이미지 크기 및 픽셀 스케일로 사각형을 그리는 함수
    
    Parameters:
    center_ra (float): 중심 RA (deg)
    center_dec (float): 중심 Dec (deg)
    x_size (float): 이미지의 x 방향 크기 (pixels)
    y_size (float): 이미지의 y 방향 크기 (pixels)
    pixel_scale (float): 픽셀 스케일 (arcsec/pixel)
    """
    
    # 중심 좌표 SkyCoord 객체 생성
    center = SkyCoord(ra=center_ra*u.deg, dec=center_dec*u.deg, frame='icrs')

    # 이미지의 전체 크기를 각도로 변환 (arcsec -> degrees로 변환)
    x_deg = (x_size * pixel_scale) / 3600.0  # degrees
    y_deg = (y_size * pixel_scale) / 3600.0  # degrees

    # 사각형의 네 모서리 좌표 계산 (동서남북 방향으로)
    corners_ra = [center.ra.deg - x_deg / 2, center.ra.deg + x_deg / 2,
                  center.ra.deg + x_deg / 2, center.ra.deg - x_deg / 2]
    corners_dec = [center.dec.deg - y_deg / 2, center.dec.deg - y_deg / 2,
                   center.dec.deg + y_deg / 2, center.dec.deg + y_deg / 2]

    # 시각화 (사각형)
    # plt.figure(figsize=(6, 6))
    plt.plot(corners_ra + [corners_ra[0]], corners_dec + [corners_dec[0]], ls='-', color='tomato', alpha=0.5)  # 폐곡선을 위해 첫 점을 마지막에 추가
    plt.scatter(center_ra, center_dec, color='r', marker='*', s=50)
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.title(f'Center ({center_ra:.3f}, {center_dec:.3f})')
    plt.gca().invert_xaxis()  # 천문학 좌표에서는 RA가 오른쪽에서 왼쪽으로 증가함
    # plt.legend(loc='upper center')
    plt.tight_layout()
    # plt.grid(True)
    # plt.show()

# In[ ]:
#   Path
# ps1 image download page
ps1filename = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
fitscut = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
# path_save = '.'

# In[ ]:
#   Hard Coded Input
intbl = Table.read(f"/large_data/factory/skygrid/skygrid_tiles_7DS.csv")

# Needed tiles lists
tiles = [
    "T11623", "T11624", "T11625", "T11626", "T11627", "T11874", 
    "T11877", "T11878", "T11879", "T11880", "T11881", "T11882", "T11883", "T11884", "T11885", 
    "T11886", "T12137", "T12138", "T12139", "T12140", "T12141", "T12142", "T12143", "T12144", 
    "T12145", "T12146", "T12399", "T12400", "T12401", "T12402", "T12403", "T12404", "T12405", 
    "T12406", "T12407", "T12408", "T12662", "T12663", "T12664", "T12665", "T12666", "T12667", 
    "T12668", "T12669", "T12670", "T12671", "T12927", "T12928", "T12929", "T12930", "T12931", 
    "T12932", "T12933", "T12934", "T12935", "T12936"
]

filte = 'r'

n_retry = 10

for oo, obj in enumerate(tiles):    
    print(f"OBJECT: {obj} ({oo+1}/{len(tiles)})")
    filename = f"ref_PS1_{obj}_00000000_000000_{filte}_0.fits"
    cra, cdec = intbl['ra'][intbl['tile'] == obj].item(), intbl['dec'][intbl['tile'] == obj].item()

    #   Input parameters
    # obj = 'T12400'
    # filte = 'r'
    # cra, cdec = 116.34980988593155,-20.377358490566035 # Getting center coordinate

    # In[ ]:
    #   Image Setting
    xsize, ysize = 10200,6800 # x, y size of the image
    pixscale = 0.51 # pixel scale [arcsec / pixel]

    #   Margin
    margin_frac = 0.1
    xmargin, ymargin = xsize*(1+margin_frac), ysize*(1+margin_frac)
    r_ra, r_dec = xmargin*pixscale/3600./2., ymargin*pixscale/3600./2.


    if filte not in ['g', 'r', 'i', 'z', 'y']:
        print(f"Input filte must be in grizy")
        sys.exit()

    # n, m = 8, 8
    n, m = 8, 4
    narr, marr = np.arange(-n, n), np.arange(-m, m)

    ps1size = 6000
    ps1pixscale = 0.25
    step = ps1size*ps1pixscale/3600.


    rastep = np.linspace(cra - r_ra, cra + r_ra, n)
    decstep = np.linspace(cdec - r_dec, cdec + r_dec, n)

    ra_dec_pairs = list(itertools.product(rastep, decstep))

    #
    path_swarp_conf = '~/gppy/config/7dt.swarp' # Swarp configuration file needed
    path_slice = './ps1_slice'
    path_out = f'./{filte}/{filename}'
    if not os.path.exists(os.path.dirname(path_out)):
        os.makedirs(os.path.dirname(path_out))



    # In[ ]:


    # Define the path to the file
    # Needed images list
    # file_path = '/data3/share/7DT_7DS/skygrid_tiles_7DS.csv'

    # Read the file with the correct delimiter
    # df = pd.read_csv(file_path, delimiter=',')




    # In[ ]:
    # nn = 0
    # tile = tiles[nn]
    # obj = df[df['tile'] == tile]  # Selecting the tile objectf
    # cra = [obj['ra'].values[0]]   # Getting center coordinate
    # cdec = [obj['dec'].values[0]]


    # Loop over each tile
    t0 = time.time()

    path_slice = f"./{obj}{filte}/"  # Create a directory for each tile's images, change location or names if needed
    os.makedirs(path_slice, exist_ok=True)

    # Grid and download images
    tdec = np.array([])
    tra = np.array([])

    # Downloading images of grid format overlaping for 0.2' degrees, in this example. 8x8=64 images are created
    # for i in narr:  # 8 steps to cover the field in Dec
    #     for j in marr:  # 8 steps to cover the field in RA
    #         tdec = np.append(tdec, cdec + step + i * step*2)  # Smaller step for more overlap
    #         tra = np.append(tra, cra + step + j * step*2)

    tra = np.array([val[0] for val in ra_dec_pairs])
    tdec = np.array([val[1] for val in ra_dec_pairs])

    table = getimages(tra, tdec, filters=filte, size=6000)  # Keep the size no bigger than 6000 pixels

    print(f"{time.time() - t0:.1f} s: got list of {len(table)} images for {len(tra)} positions")

    table.sort(['projcell', 'subcell', 'filter'])

    # %%

    plt.plot(tra, tdec, marker='.', ls='none', color='grey')
    # plt.plot(cra, cdec, 'rx', ms=10)
    plot_rectangle_on_sky(center_ra=cra, center_dec=cdec, x_size=xsize, y_size=ysize, pixel_scale=pixscale)
    # plt.show()
    plt.savefig(f"{os.path.dirname(path_out)}/ps1_query_pointing_{obj}_{filte}.png")
    plt.close('all')
    # %%
    image_counter = 1
    for row in table:
        ra = row['ra']
        dec = row['dec']
        projcell = row['projcell']
        subcell = row['subcell']
        filter = row['filter']

        fname = "t{:08.4f}{:+07.4f}.{}.fits".format(ra, dec, filter)
        file_path = os.path.join(path_slice, fname)

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"{time.time() - t0:.1f} s: Image already exists and is complete: {fname}")
        else:
            if os.path.exists(file_path):
                print(f"{time.time() - t0:.1f} s: Incomplete image detected, re-downloading: {fname}")
            else:
                print(f"{time.time() - t0:.1f} s: Downloading missing image: {fname}")

            url = row["url"]

            n_try = 0
            query_success = False
            while (n_try < n_retry) and (query_success == False):
                try:
                    response = requests.get(url, timeout=(5, 10))
                    with open(file_path, "wb") as file:
                        file.write(response.content)
                    query_success = True
                    n_try = 0
                except Exception as e:
                    print(f"Error Occured:")
                    print(e)
                    n_try += 1


        print(f"{image_counter}. Image processed: {fname}")
        image_counter += 1
        # break

    # Convert degrees to HMS/DMS for the current tile
    center_ra, center_dec = degrees_to_hms_dms(cra, cdec)

    # SWarp stacking process
    start_time = time.time()

    image_files = sorted(glob.glob(f'{path_slice}/*.fits'))
    list_file_path = f'{path_slice}/images_to_combine_{obj}_{filte}.txt'
    with open(list_file_path, 'w') as f:
        for img in image_files:
            f.write(f'{img}\n')

    print(f"{len(image_files)} will be stacked")

    swarp_command = (
        f'swarp @{list_file_path} -c {path_swarp_conf} '
        f'-IMAGEOUT_NAME {path_out} '
        f'-CENTER "{center_ra},{center_dec}" '
        f'-IMAGE_SIZE 10200,6800 '  # 7DT image size
        f'-NTHREADS 8 '
    )
    print(swarp_command)
    print(f"="*60)
    os.system(swarp_command)

    os.system(f'rm {list_file_path}')
    os.system('rm swarp.xml')
    os.system('rm coadd.weight.fits') # Need?

    end_time = time.time()
    print(f"Total runtime for {obj}: {end_time - start_time} seconds")

