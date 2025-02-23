import requests
from astropy.table import Table
from io import StringIO
import os
import time

# ps1 image download page
ps1filename = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
fitscut = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"

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

def download_images_for_pointings(tab, path_slice, n_retry=3):
    """
    Download images for a list of RA/Dec pointings.
    
    Parameters:
    - ra_dec_pairs (list of tuples): List of (RA, Dec) pairs for image downloading.
    - filters (str): Filter to use for image query (e.g., 'g', 'r', 'i', 'z', 'y').
    - path_slice (str): Path to save downloaded images.
    - n_retry (int): Number of retries for failed downloads.
    
    Returns:
    - image_files (list): List of paths to downloaded image files.
    """
    image_files = []
    image_counter = 1
    for rr, row in enumerate(tab):
        ra = row['ra']
        dec = row['dec']
        projcell = row['projcell']
        subcell = row['subcell']
        filte = row['filter']
        t0 = time.time()
        fname = "t{:08.4f}{:+07.4f}.{}.fits".format(ra, dec, filte)
        file_path = os.path.join(path_slice, fname)

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"[{rr:0>2}] {time.time() - t0:.1f} s: Image already exists and is complete: {fname}", end='-->')
        else:
            if os.path.exists(file_path):
                print(f"[{rr:0>2}] {time.time() - t0:.1f} s: Incomplete image detected, re-downloading: {fname}", end='-->')
            else:
                print(f"[{rr:0>2}] {time.time() - t0:.1f} s: Downloading missing image: {fname}", end='-->')

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
                    print(f"Retry...")
                    n_try += 1

        image_files.append(file_path)
        print(f"Done!")
        image_counter += 1
    return image_files
