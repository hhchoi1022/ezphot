#%%
from pathlib import Path
from tippy.catalog import ReferenceCatalog
from typing import List
from astropy.io import ascii
from astropy.table import Table
from shapely.geometry import Polygon

def select_reference_sources(catalog: ReferenceCatalog,
                             mag_lower : float = 12, 
                             mag_upper : float = 18, 
                             **kwargs):
    if not catalog.data:
        raise RuntimeError(f'No catalog data found for {catalog.objname}')
    
    # For APASS Cut
    cutline_apass = dict(e_ra = [0, 0.5], e_dec = [0, 0.5], e_V_mag = [0.01, 0.05], V_mag = [mag_lower, mag_upper])
    # For GAIA cut 
    cutline_gaia = dict(V_flag = [0,1], V_mag = [mag_lower, mag_upper])
    # For GAIAXP cut pmra, pmdec for astrometric reference stars, bp-rp for color
    cutline_gaiaxp = {"pmra" : [-20,20], "pmdec" : [-20,20], "bp-rp" : [0.0, 1.5], "g_mean" : [mag_lower, mag_upper]}
    # For PS1 cut
    cutline_ps1 = {"gFlags": [0,10], "g_mag": [mag_lower, mag_upper]}
    # For SMSS cut
    cutline_smss = {"ngood": [20,999], "class_star": [0.8, 1.0], "g_mag": [mag_lower, mag_upper]}
    
    if catalog.catalog_type == 'APASS':
        cutline = cutline_apass
    elif catalog.catalog_type == 'GAIA':
        cutline = cutline_gaia
    elif catalog.catalog_type == 'GAIAXP':
        cutline = cutline_gaiaxp
    elif catalog.catalog_type == 'PS1':
        cutline = cutline_ps1
    elif catalog.catalog_type == 'SMSS':
        cutline = cutline_smss
    else:
        raise ValueError('Invalid catalog type: %s' % catalog.catalog_type)
    cutline = {**cutline, **kwargs}
    
    ref_sources = catalog.data
    applied_kwargs = []
    for key, value in cutline.items():
        if key in ref_sources.colnames:
            applied_kwargs.append({key : [value]})
            ref_sources = ref_sources[(ref_sources[key] >= value[0]) & (ref_sources[key] <= value[1])]
    return ref_sources, applied_kwargs


def get_catalogs_coord(ra: float,
                       dec: float,
                       fov_ra: float = 1.0,
                       fov_dec: float = 1.0,
                       catalog_type: str = 'GAIAXP',
                       catalog_archive_path: str = None,
                       verbose : bool = False) -> List[ReferenceCatalog]:
    """
    Return a list of ReferenceCatalog instances whose tiles intersect the given (ra, dec, fov) region.

    Parameters
    ----------
    ra : float
        RA center in degrees.
    dec : float
        Dec center in degrees.
    fov_ra : float
        FOV width in RA in degrees.
    fov_dec : float
        FOV height in Dec in degrees.
    catalog_type : str
        Catalog type to filter (GAIAXP, GAIA, APASS, PS1, SMSS, SDSS).
    catalog_archive_path : str
        Path to the catalog summary ASCII file.

    Returns
    -------
    List[ReferenceCatalog]
        List of ReferenceCatalog instances matching the region.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import numpy as np
    
    def make_sky_rectangle_polygon(ra_center, dec_center, fov_ra, fov_dec):
        ra_offset = (fov_ra / 2) / np.cos(np.radians(dec_center))
        dec_offset = fov_dec / 2

        return Polygon([
            (ra_center - ra_offset, dec_center - dec_offset),
            (ra_center + ra_offset, dec_center - dec_offset),
            (ra_center + ra_offset, dec_center + dec_offset),
            (ra_center - ra_offset, dec_center + dec_offset)
        ])

    if catalog_archive_path is None:
        catalog_archive_path = Path(__file__).parent / 'catalog_archive' / 'catalog_summary.ascii_fixed_width'    
    
    try:
        summary = ascii.read(catalog_archive_path, format='fixed_width')
    except FileNotFoundError:
        raise RuntimeError(f"ReferenceCatalog summary not found at {catalog_archive_path}")

    # Filter by catalog type
    summary = summary[summary['cat_type'] == catalog_type]

    # Filter the summary catalog near the target coordinates
    target_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    catalog_coord = SkyCoord(ra=summary['ra'] * u.deg, dec=summary['dec'] * u.deg, frame='icrs')
    separation = target_coord.separation(catalog_coord).value
    summary_nearby = summary[separation < min(5, 10* (fov_ra**2 + fov_dec**2)**0.5)]  # 3 degrees is a reasonable threshold

    # Define target polygon
    target_poly = make_sky_rectangle_polygon(ra, dec, fov_ra, fov_dec)

    catalogs = []
    for row in summary_nearby:
        tile_poly = make_sky_rectangle_polygon(row['ra'], row['dec'], row['fov_ra'], row['fov_dec'])

        if tile_poly.intersects(target_poly):
            try:
                catalogs.append(
                    ReferenceCatalog(objname=row['objname'], catalog_type=row['cat_type'],
                                     fov_ra=row['fov_ra'], fov_dec=row['fov_dec'])
                )
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to load catalog for {row['objname']}: {e}")
    
    if catalogs:
        if verbose:
            print(f"Found {len(catalogs)} catalogs matching the region.")
        return catalogs
    else:
        if verbose:
            print("No catalogs found matching the region.")
        return []
