#%%
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy.io import ascii
import astropy
from shapely.geometry import box

from ezphot.skycatalog import SkyCatalog
from ezphot.helper import Helper

#%%

class SkyCatalogUtility:
    """
    Utility class for loading, filtering, and selecting sources from sky catalogs.    
    """
    
    def __init__(self,
                 catalog_archive_path: Optional[str] = None):
        """
        Parameters
        ----------
        catalog_archive_path : Optional[str], optional
            Path to the catalog archive summary file. If not provided, the path will be set to the catalog_dir in the config.
        """
        self.helper = Helper()
        self.catalog_archive_path = Path(catalog_archive_path) if catalog_archive_path else Path(self.helper.config['CATALOG_DIR']) / 'catalog_summary.ascii_fixed_width'

    def get_catalogs(self,
                     ra: float,
                     dec: float,
                     objname: str = None,
                     fov_ra: float = 1.0,
                     fov_dec: float = 1.0,
                     catalog_type: str = 'GAIAXP',
                     fraction_criteria: float = 0.05,
                     query_when_not_archived: bool = True,
                     verbose: bool = False) -> List[SkyCatalog]:
        """
        Load SkyCatalog instances that overlap with a given coordinate and FOV.
        
        Parameters
        ----------
        ra : float
            The right ascension of the target coordinate.
        dec : float
            The declination of the target coordinate.
        objname : str, optional
            The name of the target object.
        fov_ra : float, optional
            The field of view in right ascension.
        fov_dec : float, optional
            The field of view in declination.
        catalog_type : str, optional
            The type of catalog to load.
        fraction_criteria : float, optional
            The fraction of the catalog that must overlap with the target coordinate and FOV.
        query_when_not_archived : bool, optional
            Whether to query the catalog when it is not archived.
        verbose : bool, optional
            Whether to print verbose output.

        Returns
        -------
        catalogs : List[SkyCatalog]
            The list of SkyCatalog instances that overlap with the target coordinate and FOV.
        """
        def make_sky_rectangle(ra_center, dec_center, fov_ra, fov_dec):
            ra_offset = (fov_ra / 2) / np.cos(np.radians(dec_center))
            dec_offset = fov_dec / 2
            return box(ra_center - ra_offset, dec_center - dec_offset,
                       ra_center + ra_offset, dec_center + dec_offset)

        # Load catalog summary
        catalog_path = self.catalog_archive_path
        if catalog_path is None:
            catalog_path = Path(__file__).parent / 'catalog_archive' / 'catalog_summary.ascii_fixed_width'
        summary = ascii.read(catalog_path, format='fixed_width')
        summary = summary[summary['cat_type'] == catalog_type]

        # Fast angular pre-filter
        ra_arr = np.asarray(summary['ra'])
        dec_arr = np.asarray(summary['dec'])
        cos_dec0, sin_dec0 = np.cos(np.radians(dec)), np.sin(np.radians(dec))
        cos_dec, sin_dec = np.cos(np.radians(dec_arr)), np.sin(np.radians(dec_arr))
        delta_ra = np.radians(ra_arr - ra)
        angular_sep = np.degrees(np.arccos(np.clip(sin_dec0 * sin_dec + cos_dec0 * cos_dec * np.cos(delta_ra), -1, 1)))

        search_radius = min(5, 10 * np.sqrt(fov_ra ** 2 + fov_dec ** 2))
        summary_nearby = summary[angular_sep < search_radius]

        # Target polygon
        target_poly = make_sky_rectangle(ra, dec, fov_ra, fov_dec)

        # Intersection filter
        filtered_rows = []
        for row in summary_nearby:
            tile_poly = make_sky_rectangle(row['ra'], row['dec'], row['fov_ra'], row['fov_dec'])
            if tile_poly.intersects(target_poly):
                intersection_fraction = tile_poly.intersection(target_poly).area / target_poly.area
                self.helper.print(f"Catalog {row['objname']} intersection fraction: {intersection_fraction:.2f}", verbose)
                if intersection_fraction >= fraction_criteria:
                    filtered_rows.append(row)

        # Parallel load
        # with ThreadPoolExecutor(max_workers=6) as executor:
        #     catalogs = list(executor.map(self._load_catalog_worker, filtered_rows))
        catalogs = [self._load_catalog_worker(row) for row in filtered_rows]

        if catalogs:
            self.helper.print(f"Found {len(catalogs)} catalogs matching the region (fraction > {fraction_criteria}).", verbose)
        else:
            self.helper.print("No catalogs found.", verbose)
        
        if query_when_not_archived:
            if len(catalogs) == 0:
                skycatalog = SkyCatalog(objname = objname, ra = ra, dec = dec, fov_ra = fov_ra, fov_dec = fov_dec, catalog_type = catalog_type, overlapped_fraction = fraction_criteria, verbose = verbose)
                catalogs = [skycatalog]

        return catalogs

    def select_reference_sources(self,
                                 catalog: SkyCatalog,
                                 mag_lower: Optional[float] = None,
                                 mag_upper: Optional[float] = None,
                                 **kwargs) -> Tuple['astropy.table.Table', list]:
        """
        Select reference sources from a given SkyCatalog based on predefined cuts.
        
        Parameters
        ----------
        catalog : SkyCatalog
            The SkyCatalog instance to select reference sources from.
        mag_lower : Optional[float], optional
            The lower magnitude bound for filtering reference sources.
        mag_upper : Optional[float], optional
            The upper magnitude bound for filtering reference sources.
        **kwargs : dict
            Additional keyword arguments to filter reference sources.

        Returns
        -------
        ref_sources : astropy.table.Table
            The selected reference sources.
        applied_kwargs : list
            The applied keyword arguments.
        """
        if not catalog.data:
            raise RuntimeError(f'No catalog data found for {catalog.objname}')

        # Define default cutlines
        cutlines = {
            'APASS': dict(e_V_mag=[0.001, 0.2], V_mag=[mag_lower, mag_upper]),
            'GAIA': dict(V_flag=[0, 1], V_mag=[mag_lower, mag_upper]),
            'GAIAXP': {"pmra": [-20, 20], "pmdec": [-20, 20], "bp-rp": [0.0, 1.5], "g_mean": [mag_lower, mag_upper]},
            'PS1': {"gFlags": [0, 10], "g_mag": [mag_lower, mag_upper]},
            'SMSS': {"ngood": [5, 999], "class_star": [0.3, 1.0], "g_mag": [mag_lower, mag_upper]}
        }

        if catalog.catalog_type not in cutlines:
            raise ValueError(f"Invalid catalog type: {catalog.catalog_type}")

        cutline = {**cutlines[catalog.catalog_type], **kwargs}

        ref_sources = catalog.data
        applied_kwargs = []
        for key, value in cutline.items():
            if key in ref_sources.colnames:
                applied_kwargs.append({key: [value]})
                ref_sources = ref_sources[(ref_sources[key] >= value[0]) & (ref_sources[key] <= value[1])]

        return ref_sources, applied_kwargs

    
    @staticmethod
    def _load_catalog_worker(row):
        """Worker to load a SkyCatalog from a summary row."""
        return SkyCatalog(objname=row['objname'],
                          catalog_type=row['cat_type'],
                          fov_ra=row['fov_ra'],
                          fov_dec=row['fov_dec'],
                          verbose = False)
# %%
if __name__ == "__main__":
    # Example usage
    from ezphot.utils import DataBrowser
    from ezphot.imageobjects import *
    from ezphot.methods import *
    import glob
    databrowser = DataBrowser('scidata')
    databrowser.observatory = 'RASA36'
    databrowser.telkey = 'RASA36_KL4040_HIGH_1x1'
    databrowser.objname = 'NGC1566'
    stacked_imgset = databrowser.search(pattern='Calib*60.com.fits', return_type='science')
    stacked_imglist = stacked_imgset.target_images
#%%
    # db = DataBrowser('scidata')
    # db.objname = 'NGC1566'
    # db.observatory = 'RASA36'
    
    # target_imgset = db.search('*.com.fits', 'science')
    # target_imglist = target_imgset.target_images
    target_img = stacked_imglist[0]
    ra = target_img.ra
    dec = target_img.dec
    fov_ra = target_img.fovx
    fov_dec = target_img.fovy
    catalog_type = 'APASS'
    utility = SkyCatalogUtility()
    
    catalogs = utility.get_catalogs(ra, dec, fov_ra, fov_dec, catalog_type=catalog_type, verbose=True)

# %%
