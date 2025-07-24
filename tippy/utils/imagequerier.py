#%%
from astroquery.hips2fits import hips2fits
import astropy.units as u
import numpy as np
from astroquery.mocserver import MOCServer
from astropy.coordinates import Longitude, Latitude, Angle
from matplotlib import cm 
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from pathlib import Path
from astropy.table import Table
import os
from multiprocessing import Pool
from astropy.time import Time

from tippy.helper import Helper
from tippy.image import ScienceImage, ReferenceImage

#%%
hips2fits.timeout = 300
class HIPS2FITS:
    """
    A class to handle image queries.
    """

    def __init__(self, catalog_key: str = None):
        """
        Initializes the ImageQuerier with the path to the image.

        :param image_path: Path to the image file.
        """
        if catalog_key is not None:
            if catalog_key not in self.catalog_ids.keys():
                raise ValueError(f"Catalog Key '{catalog_key}' is not recognized. Available keys: {list(self.catalog_ids.keys())}")
        self.current_catalog_key = catalog_key

    def change_catalog(self, catalog_key):
        """Change the current catalog to query."""
        if catalog_key in self.catalog_ids.keys():
            self.current_catalog_key = catalog_key
            print(self.__repr__())
        else:
            raise ValueError(f"Catalog Key '{catalog_key}' is not recognized.")

    @property
    def config(self):
        class Configuration:
            """Handles configuration for Vizier queries."""

            @property
            def projection(self):
                return 'TAN'
            
            @projection.setter
            def projection(self, value):
                self.projection = value
            
            @property
            def coordsys(self):
                return 'icrs'
        
            @coordsys.setter
            def coordsys(self, value):
                self.coordsys = value
            
            @property
            def format(self):
                return 'fits'
            
            @format.setter
            def format(self, value):
                self.format = value
                
            @property
            def stretch(self):
                return 'linear'
            
            @stretch.setter
            def stretch(self, value):
                self.stretch = value
                
            @property
            def cmap(self):
                return 'Greys_r'

            @cmap.setter
            def cmap(self, value):
                self.cmap = value
                
            @property
            def min_cut(self):
                return 0.5
            
            @min_cut.setter
            def min_cut(self, value):
                self.min_cut = value
                
            @property
            def max_cut(self):
                return 99.5
            
            @max_cut.setter
            def max_cut(self, value):
                self.max_cut = value
            
            def __repr__(self):
                return (f"========HIPS2FITS Configuration========\n"
                        f"  projection      = {self.projection}\n"
                        f"  coordsys        = {self.coordsys}\n"
                        f"  format          = {self.format}\n"
                        f"  stretch         = {self.stretch}\n"
                        f"  cmap            = {self.cmap}\n"
                        f"  min_cut         = {self.min_cut}\n"
                        f"  max_cut         = {self.max_cut}\n"
                        f"====================================")
        
        return Configuration()
    
    @property
    def catalog_ids(self):
        catalog_ids = dict()
        # SkyMapper DR4 
        catalog_ids['SkyMapper/SMSS4/g'] = 'CDS/P/Skymapper/DR4/g'
        catalog_ids['SkyMapper/SMSS4/r'] = 'CDS/P/Skymapper/DR4/r'
        catalog_ids['SkyMapper/SMSS4/i'] = 'CDS/P/Skymapper/DR4/i'
        
        # Skymapper DR1
        catalog_ids['SkyMapper/SMSS1/u'] = 'CDS/P/Skymapper-U'
        catalog_ids['SkyMapper/SMSS1/g'] = 'CDS/P/Skymapper-G'
        catalog_ids['SkyMapper/SMSS1/v'] = 'CDS/P/Skymapper-V'
        catalog_ids['SkyMapper/SMSS1/r'] = 'CDS/P/Skymapper-R'
        catalog_ids['SkyMapper/SMSS1/i'] = 'CDS/P/Skymapper-I'
        catalog_ids['SkyMapper/SMSS1/z'] = 'CDS/P/Skymapper-Z'

        # Pan-STARRS DR1 
        catalog_ids['PanSTARRS/PS1/g'] = "CDS/P/PanSTARRS/DR1/g"
        catalog_ids['PanSTARRS/PS1/r'] = "CDS/P/PanSTARRS/DR1/r"
        catalog_ids['PanSTARRS/PS1/i'] = "CDS/P/PanSTARRS/DR1/i"
        catalog_ids['PanSTARRS/PS1/z'] = "CDS/P/PanSTARRS/DR1/z"
        catalog_ids['PanSTARRS/PS1/y'] = "CDS/P/PanSTARRS/DR1/y"
        
        # SDSS DR9 
        catalog_ids['SDSS/SDSS9/u'] = "CDS/P/SDSS9/u"
        catalog_ids['SDSS/SDSS9/g'] = "CDS/P/SDSS9/g"
        catalog_ids['SDSS/SDSS9/r'] = "CDS/P/SDSS9/r"
        catalog_ids['SDSS/SDSS9/i'] = "CDS/P/SDSS9/i"
        catalog_ids['SDSS/SDSS9/z'] = "CDS/P/SDSS9/z"
        
        # DESI Legacy Imaging Survey
        catalog_ids['DESI/DESI/g'] = "CDS/P/DESI-Legacy-Surveys/DR10/g"
        catalog_ids['DESI/DESI/r'] = "CDS/P/DESI-Legacy-Surveys/DR10/r"
        catalog_ids['DESI/DESI/i'] = "CDS/P/DESI-Legacy-Surveys/DR10/i"
        catalog_ids['DESI/DESI/z'] = "CDS/P/DESI-Legacy-Surveys/DR10/z"
        
        # DSS 
        catalog_ids['DSS/DSS2/b'] = "CDS/P/DSS2/blue"
        catalog_ids['DSS/DSS2/r'] = "CDS/P/DSS2/red"
        # catalog_ids['DSSDSS2/nir'] = "CDS/P/DSS2/NIR"
        
        # ZTF
        catalog_ids['ZTF/ZTF7/g'] = "CDS/P/ZTF/DR7/g"
        catalog_ids['ZTF/ZTF7/r'] = "CDS/P/ZTF/DR7/r"
        catalog_ids['ZTF/ZTF7/i'] = "CDS/P/ZTF/DR7/i"
        
        # DECAM
        catalog_ids['DECALS/DEC5/g'] = "CDS/P/DECaLS/DR5/g"
        catalog_ids['DECALS/DEC5/r'] = "CDS/P/DECaLS/DR5/r"
        
        # DES
        catalog_ids['DES/DES2/g'] = "CDS/P/DES-DR2/g"
        catalog_ids['DES/DES2/r'] = "CDS/P/DES-DR2/r"
        catalog_ids['DES/DES2/i'] = "CDS/P/DES-DR2/i"
        catalog_ids['DES/DES2/z'] = "CDS/P/DES-DR2/z"
        catalog_ids['DES/DES2/Y'] = "CDS/P/DES-DR2/Y"
        
        
        return catalog_ids
    
    def show_available_catalogs(self):
        """Display available catalogs."""
        print("Current catalog: ", self.current_catalog_key)
        print("Available catalogs\n==================")
        for catalog_name, catalog_id in self.catalog_ids.items():
            print(f"{catalog_name}: {catalog_id}")
    
    def _query(self,
               wcs: WCS = None, # If wcs inputted, overrides ra, dec, fov, rotation_angle
               width: int = 2000,
               height: int = 2000,
               ra: float = 0.0,
               dec: float = 0.0,
               fov: float = 5.0,
               rotation_angle: float = 0.0,
               save_path: str = None,
               verbose: bool = False,
               ):
        
        if save_path is None:
            save_path = os.path.join(os.getcwd(), f"hips2fits_{self.current_catalog_key}_{ra}_{dec}.fits")
            if verbose:
                print(f"Default save path: {save_path}")
        
        if wcs is not None:
            # If WCS is provided, use it to query the image
            result = hips2fits.query_with_wcs(
                hips=self.catalog_ids[self.current_catalog_key],
                wcs=wcs,
                format=self.config.format,
                min_cut=self.config.min_cut,
                max_cut=self.config.max_cut,
                stretch=self.config.stretch,
                cmap=cm.get_cmap(self.config.cmap),
                verbose=verbose,
            )
        else:
            result = hips2fits.query(
                hips=self.catalog_ids[self.current_catalog_key],
                width=width,
                height=height,
                projection = self.config.projection,
                fov=Angle(fov * u.deg),
                ra=Longitude(ra * u.deg),
                dec=Latitude(dec * u.deg),
                coordsys=self.config.coordsys,
                rotation_angle=Angle(rotation_angle * u.deg),
                format=self.config.format,
                min_cut=self.config.min_cut,
                max_cut=self.config.max_cut,
                stretch=self.config.stretch,
                cmap=cm.get_cmap(self.config.cmap),
                verbose=verbose,
            )
            
        if verbose:
            print(f"Saved: {save_path}")
        result[0].writeto(save_path, overwrite=True)
        return save_path

    
    def check_coverage(self, ra: float, dec: float, radius_deg: float = 1.0, verbose=True):
        """
        Check if the current HiPS catalog has coverage at the given RA/Dec.

        Parameters
        ----------
        ra : float
            Right Ascension in degrees
        dec : float
            Declination in degrees
        radius_deg : float
            Search radius in degrees
        verbose : bool
            If True, prints whether coverage exists

        Returns
        -------
        bool
            True if coverage exists, False otherwise
        """
        if self.current_catalog_key is None:
            raise ValueError("No HiPS catalog selected.")
        
        hips_id = self.catalog_ids[self.current_catalog_key]
        center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        region = CircleSkyRegion(center=center, radius=radius_deg * u.deg)

        try:
            # name 'MOC' is not defined error in MOCSServer.query_region. Modify the import statement
            # to import MOC from mocpy
            results = MOCServer.query_region(
                region=region,
                criteria=f"obs_id={hips_id}",
                intersect="overlaps",
                max_rec=1
            )
        except Exception as e:
            if verbose:
                print(f"[Coverage Check Error] {e}")
            return False

        has_coverage = len(results) > 0
        if verbose:
            print(f"[Coverage Check] {hips_id} {'has' if has_coverage else 'does NOT have'} coverage at RA={ra}, Dec={dec}")
        return has_coverage
            

class ImageQuerier(HIPS2FITS):
    """
    A class to handle image queries using HiPS2FITS.
    Inherits from HIPS2FITS to utilize its methods and properties.
    """

    def __init__(self, catalog_key: str = None):
        """
        Initializes the ImageQuerier with the specified catalog key.

        :param catalog_key: Key for the catalog to query.
        """
        super().__init__(catalog_key=catalog_key)
        self.helper = Helper()

    def __repr__(self):
        return f"ImageQuerier(catalog={self.current_catalog_key})\n{self.config}"
    
    def _split_query_regions(self,
                             width: int,
                             height: int,
                             ra: float,
                             dec: float,
                             fov: float,
                             rotation_angle: float = 0.0,
                             max_pixels: int = 45000000,
                             margin_fraction: float = 0.1,
                             verbose: bool = True):
        """
        Split a large query into tiles with unique RA/Dec centers that cover the full region.

        Returns
        -------
        list of dict : Each dict contains query parameters for a tile
        """
        total_pixels = width * height

        if total_pixels <= max_pixels:
            width_with_margin = int(np.ceil(width * (1 + margin_fraction)))
            height_with_margin = int(np.ceil(height * (1 + margin_fraction)))
            fov_with_margin = fov * (1 + margin_fraction)
            return [{
                'width': width_with_margin,
                'height': height_with_margin,
                'ra': ra,
                'dec': dec,
                'fov': fov_with_margin,
                'rotation_angle': rotation_angle,
                'tile_id': '0_0'
            }]

        # Determine number of splits needed per axis
        n_splits = int(np.ceil(np.sqrt(total_pixels / max_pixels)))
        tile_width = width // n_splits
        tile_height = height // n_splits
        base_tile_fov = fov / n_splits  # Original FoV per tile

        # Apply margin to width/height and FoV
        tile_width_with_margin = int(np.ceil(tile_width * (1 + margin_fraction)))
        tile_height_with_margin = int(np.ceil(tile_height * (1 + margin_fraction)))
        tile_fov_with_margin = base_tile_fov * (1 + margin_fraction)

        # Pixel scale (deg/pixel) stays fixed
        pixscale_deg = fov / width
        dec_rad = np.deg2rad(dec)

        if verbose:
            print(f"[SPLIT] {width}x{height} >>> {n_splits}x{n_splits} tiles "
                f"({tile_width}x{tile_height} px), margin = {margin_fraction:.1%}")

        tile_params = []

        for i in range(n_splits):
            for j in range(n_splits):
                # Offset in sky from the center (no margin added here)
                delta_ra_deg = ((j + 0.5) - n_splits / 2) * tile_width * pixscale_deg / np.cos(dec_rad)
                delta_dec_deg = ((n_splits / 2) - (i + 0.5)) * tile_height * pixscale_deg

                center_ra = ra + delta_ra_deg
                center_dec = dec + delta_dec_deg

                tile_params.append({
                    'width': tile_width_with_margin,
                    'height': tile_height_with_margin,
                    'ra': center_ra,
                    'dec': center_dec,
                    'fov': tile_fov_with_margin,
                    'rotation_angle': rotation_angle,
                    'tile_id': f"{i}_{j}"
                })

        return tile_params

    def _query_tile_worker(self, kwargs):
        """
        Standalone worker function for multiprocessing.
        This function must be top-level (not class method) for pickling.
        """
        tile_param, save_path, verbose = kwargs
        save_path = Path(save_path)

        save_path = save_path.with_suffix(f".{tile_param['tile_id']}.fits")

        return self._query(
            wcs=None,
            width=tile_param['width'],
            height=tile_param['height'],
            ra=tile_param['ra'],
            dec=tile_param['dec'],
            fov=tile_param['fov'],
            rotation_angle=tile_param['rotation_angle'],
            verbose=verbose,
            save_path=save_path
        )

    def query(self,
              width: int,
              height: int,
              ra: float,
              dec: float,
              pixelscale: float,
              telinfo: Table,
              save_path: str = None,
              objname: str = None,
              rotation_angle: float = 0.0,
              verbose: bool = True,
              resigster: bool = False,
              n_processes: int = 4):
        """
        Run HiPS2FITS queries with split tiles using multiprocessing.

        Returns
        -------
        list of astropy.io.fits.HDUList
        """
        observatory, telname, filter_ = self.current_catalog_key.split('/')

        fov = max(width, height) * pixelscale / 3600  # Convert pixel scale to degrees
        tile_params = self._split_query_regions(
            width=width,
            height=height,
            ra=ra,
            dec=dec,
            fov=fov,
            rotation_angle=rotation_angle,
            verbose=verbose
        )
        if save_path is None:
            output_path = os.path.join(Path.home(), f"hips2fits_{observatory}_{telname}_{ra}_{dec}.fits")
        else:
            output_path = str(save_path)
        if verbose:
            print(f"[QUERY] Dispatching {len(tile_params)} tiles with {n_processes} processes")

        with Pool(processes=n_processes) as pool:
            tasks = [(param, output_path, verbose) for param in tile_params]
            results = pool.map(self._query_tile_worker, tasks)
        
        from tippy.photometry import TIPStacking
        from tippy.photometry import TIPPlateSolve
        from tippy.helper import Helper
        self.helper = Helper()
        self.stacking = TIPStacking()
        self.platesolving = TIPPlateSolve()
        target_imglist = [ScienceImage(result, telinfo = telinfo, load = True) for result in results]
        # # # Stacking 
        stack_instance, stack_weight_instance = self.stacking.stack_swarp(
            target_imglist = target_imglist,
            target_bkglist= None,
            target_errormaplist = None,
            target_outpath = output_path,
            errormap_outpath = None,
            combine_type = 'average',
            resample = True,
            resample_type = 'LANCZOS3',
            center_ra = ra,
            center_dec = dec,
            pixel_scale = pixelscale,
            x_size = width,
            y_size = height,
            scale = False,
            scale_type = 'min',
            zp_key = 'ZP_APER_1',
            convolve = False,
            seeing_key = 'SEEING',
            kernel = 'gaussian',
            save = True,
            verbose = True
            )
        
        stack_instance.load()
        stack_instance.remove()
        stack_instance = stack_instance.to_referenceimage()
        # Update stack_instance with metadata
        update_header_kwargs = dict(
            BINNING = 1,
            TELNAME = telname,
            FILTER = filter_,
            TELESCOP = observatory,
            OBJNAME = objname if objname else 'Unknown',
            IMAGETYP = 'LIGHT',
            OBSDATE = Time('2001-01-01T00:00:00').isot,  # Placeholder date
            SEEING = 2.0,
            UL5_APER_2 = 21.0
        )
        stack_instance.header.update(**update_header_kwargs)
        if np.max(stack_instance.data) < 1e3:
            stack_instance.data *= 1e3
        if save_path is not None:
            stack_instance.savedir = Path(save_path).parent
            if verbose:
                print(f'Save path: {stack_instance.savepath.savepath}')
        else:
            stack_instance.savedir = None
            if verbose:
                print(f'Default save path: {stack_instance.savepath.savepath}')
        stack_instance.write()
        
        for target_img in target_imglist:
            target_img.remove()
        
        return stack_instance


#%%
    
# Example usage:
if __name__ == "__main__":
    import glob
    self = ImageQuerier(catalog_key='DECALS/DEC5/g')

    
    from astropy.io import fits
    from tippy.image import ScienceImage
    from tippy.helper import Helper, TIPDataBrowser
    from tippy.utils import SDTData
    tile_id = 'T22956'
    databrowser = TIPDataBrowser('scidata')
    databrowser.observatory = '7DT'
    databrowser.objname = 'T22956'
    target_img = databrowser.search(f'calib*_g_*100.fits', 'science')[0]
    telinfo = target_img.telinfo
    width = target_img.naxis1
    height = target_img.naxis2
    fov = max(width,height) * np.mean(target_img.pixelscale) /3600
    ra = target_img.ra
    dec = target_img.dec
    pixelscale= np.mean(target_img.pixelscale) 
    rotation_angle = 0
    verbose=True
    objname = target_img.objname
    # Run the query
    stack_image = self.query(
        width=width,
        height=height,
        ra=ra,
        dec=dec,
        pixelscale=pixelscale,
        save_path=None,
        telinfo=telinfo,
        objname=objname,
        rotation_angle=rotation_angle,
        verbose=verbose
    )
    

# %%
