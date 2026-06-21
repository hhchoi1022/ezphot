#%%
import astropy.units as u
import numpy as np
import os
from shapely.geometry import Polygon
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.time import Time
import time
from astropy.table import Table, vstack
from astroquery.vizier import Vizier
from pathlib import Path
from ezphot.helper import Helper
#%%
class SkyCatalog:
    """
    SkyCatalog class is used to query the sky catalog and get the catalog data.
    
    This class acts as an instance of reference catalog for a given object.
    
    """
    def __init__(self,
                 objname : str = None,
                 ra = None, # in deg
                 dec = None, # in deg
                 fov_ra : float = 1.3,
                 fov_dec : float = 0.9,
                 catalog_type : str = 'GAIAXP', #GAIAXP, GAIA, APASS, PS1, SDSS, SMSS, GAIAXP_CORR_LAMOST
                 catalog_version: str = 'v1',
                 overlapped_fraction : float = 0.8,
                 verbose : bool = True
                 ):
        """
        Initialize the SkyCatalog instance
        
        Parameters
        ----------
        objname : str, optional
            Object name to query. If not provided, the object name will be inferred from the ra and dec.
        ra : float, optional
            Right ascension in degrees. If not provided, the object name will be inferred from SIMBAD object name.
        dec : float, optional
            Declination in degrees. If not provided, the object name will be inferred from SIMBAD object name.
        fov_ra : float, optional
            Field of view in right ascension in degrees
        fov_dec : float, optional
            Field of view in declination in degrees
        catalog_type : str, optional
            Catalog type to query
        overlapped_fraction : float, optional
            Overlapped fraction of the field of view
        verbose : bool, optional
            Whether to print verbose output
        """
        if catalog_type not in ['GAIAXP', 'GAIA', 'APASS', 'PS1', 'SDSS', 'SMSS', 'GAIAXP_CORR_LAMOST']:
            raise ValueError('Invalid catalog type: %s' % catalog_type)
        self.helper = Helper()
        self.objname = objname
        self.ra = ra
        self.dec = dec
        self.fov_ra = fov_ra
        self.fov_dec = fov_dec
        self.catalog_type = catalog_type
        self.catalog_version = catalog_version
        self.overlapped_fraction = overlapped_fraction
        
        self.filename = None
        self.filepath = None
        self.summary_path = os.path.join(self.helper.config['CATALOG_DIR'], 'summary.ascii_fixed_width')

        self._register_objinfo(objname = objname, ra = ra, dec = dec, fov_ra = fov_ra, fov_dec = fov_dec, catalog_type = catalog_type, catalog_version = catalog_version, verbose = verbose)
        self._get_catalog(catalog_type = catalog_type, verbose = verbose)

    def __repr__(self):
        txt = f"SkyCatalog[objname = {self.objname}, type = {self.catalog_type}, version = {self.catalog_version}]"
        return txt
    
    def _query(self, catalog_name: str = 'APASS', verbose: bool = False):
        # APASS DR9
        def apass_query(ra_deg, dec_deg, rad_deg, maxmag = 20, minmag = 10, maxsources=100000):
            """
            Query APASS @ VizieR using astroquery.vizier 
            :param ra_deg: RA in degrees
            :param dec_deg: Declination in degrees
            :param rad_deg: field radius in degrees
            :param maxmag: upper limit G magnitude (optional)
            :param maxsources: maximum number of sources
            :return: astropy.table object
            """
            vquery = Vizier(columns=['*'],
                            column_filters={"Bmag":
                                            ("<%f" % maxmag),
                                            "Vmag":
                                            ("<%f" % maxmag),
                                            "g'mag":
                                            ("<%f" % maxmag),
                                            "r'mag":
                                            ("<%f" % maxmag),
                                            "i'mag":
                                            ("<%f" % maxmag),
                                            "Bmag":
                                            (">%f" % minmag),
                                            "Vmag":
                                            (">%f" % minmag),
                                            "g'mag":
                                            (">%f" % minmag),
                                            "r'mag":
                                            (">%f" % minmag),
                                            "i'mag":
                                            (">%f" % minmag),

                                        },
                            
                            row_limit=maxsources)

            field = SkyCoord(ra=ra_deg, dec=dec_deg,
                             unit=(u.deg, u.deg),
                             frame='icrs')
            query_data = vquery.query_region(field,
                                             width=("%fd" % rad_deg),
                                             catalog="II/336/apass9")
            if len(query_data) > 0:
                return query_data[0]
            else:
                return None

        # SDSS DR12
        def sdss_query(ra_deg, dec_deg, rad_deg, maxmag = 20, minmag = 10,maxsources=100000):
            """
            Query SDSS @ VizieR using astroquery.vizier
            :param ra_deg: RA in degrees
            :param dec_deg: Declination in degrees
            :param rad_deg: field radius in degrees
            :param maxmag: upper limit G magnitude (optional)
            :param maxsources: maximum number of sources
            :return: astropy.table object
            """
            vquery = Vizier(columns=['*'],
                            column_filters={"gmag":
                                            ("<%f" % maxmag),
                                            "rmag":
                                            ("<%f" % maxmag),
                                            "imag":
                                            ("<%f" % maxmag),
                                            "gmag":
                                            (">%f" % minmag),
                                            "rmag":
                                            (">%f" % minmag),
                                            "imag":
                                            (">%f" % minmag)
                                            },
                            row_limit=maxsources)

            field = SkyCoord(ra=ra_deg, dec=dec_deg,
                             unit=(u.deg, u.deg),
                             frame='icrs')
            query_data = vquery.query_region(field,
                                             width=("%fd" % rad_deg),
                                             catalog="V/147/sdss12")
            if len(query_data) > 0:
                return query_data[0]
            else:
                return None

        # PanSTARRS DR1
        def ps1_query(ra_deg, dec_deg, rad_deg, maxmag = 20, minmag = 10, maxsources= 500000):
            """
            Query PanSTARRS @ VizieR using astroquery.vizier
            :param ra_deg: RA in degrees
            :param dec_deg: Declination in degrees
            :param rad_deg: field radius in degrees
            :param maxmag: upper limit G magnitude (optional)
            :param maxsources: maximum number of sources
            :return: astropy.table object
            """
            vquery = Vizier(columns=['*'],
                            column_filters={"gmag":
                                            ("<%f" % maxmag),
                                            "rmag":
                                            ("<%f" % maxmag),
                                            "imag":
                                            ("<%f" % maxmag),
                                            "gmag":
                                            (">%f" % minmag),
                                            "rmag":
                                            (">%f" % minmag),
                                            "imag":
                                            (">%f" % minmag),

                                        },
                            
                            row_limit=maxsources)

            field = SkyCoord(ra=ra_deg, dec=dec_deg,
                             unit=(u.deg, u.deg),
                             frame='icrs')
            query_data = vquery.query_region(field,
                                             width=("%fd" % rad_deg),
                                             catalog="II/349/ps1")
            if len(query_data) > 0:
                return query_data[0]
            else:
                return None
            
        # SkyMapper DR4
        def smss_query(ra_deg, dec_deg, rad_deg, maxmag = 20, minmag = 10, maxsources=1000000):
            """
            Query PanSTARRS @ VizieR using astroquery.vizier
            :param ra_deg: RA in degrees
            :param dec_deg: Declination in degrees
            :param rad_deg: field radius in degrees
            :param maxmag: upper limit G magnitude (optional)
            :param maxsources: maximum number of sources
            :return: astropy.table object
            """
            vquery = Vizier(columns=['ObjectId','RAICRS','DEICRS','Niflags','flags','Ngood','Ngoodu','Ngoodv','Ngoodg','Ngoodr','Ngoodi','Ngoodz','ClassStar','uPSF','e_uPSF','vPSF','e_vPSF','gPSF','e_gPSF','rPSF','e_rPSF','iPSF','e_iPSF','zPSF','e_zPSF'],
                            column_filters={"gPSF":
                                            ("<%f" % maxmag),
                                            "rPSF":
                                            ("<%f" % maxmag),
                                            "iPSF":
                                            ("<%f" % maxmag),
                                            "gmag":
                                            (">%f" % minmag),
                                            "rmag":
                                            (">%f" % minmag),
                                            "imag":
                                            (">%f" % minmag),

                                        },
                            
                            row_limit=maxsources)

            field = SkyCoord(ra=ra_deg, dec=dec_deg,
                                unit=(u.deg, u.deg),
                                frame='icrs')
            query_data = vquery.query_region(field,
                                             width=("%fd" % rad_deg),
                                             catalog="II/379/smssdr4")
            if len(query_data) > 0:
                return query_data[0]
            else:
                return None

        # SkyMapper DR4
        def gaia_query(ra_deg, dec_deg, rad_deg, maxmag = 20, minmag = 10, maxsources=1000000):
            """
            Query PanSTARRS @ VizieR using astroquery.vizier
            :param ra_deg: RA in degrees
            :param dec_deg: Declination in degrees
            :param rad_deg: field radius in degrees
            :param maxmag: upper limit G magnitude (optional)
            :param maxsources: maximum number of sources
            :return: astropy.table object
            """
            vquery = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'E_BP_RP_corr', 'Bmag', 'BFlag', 'Vmag', 'VFlag', 'Rmag', 'RFlag', 'gmag', 'gFlag', 'rmag', 'rFlag', 'imag', 'iFlag'],                            
                            row_limit=maxsources)

            field = SkyCoord(ra=ra_deg, dec=dec_deg,
                                unit=(u.deg, u.deg),
                                frame='icrs')
            query_data = vquery.query_region(field,
                                             width=("%fd" % rad_deg),
                                             catalog="I/360/syntphot")
            query_data[0]['e_Bmag'] = 0.02
            query_data[0]['e_Vmag'] = 0.02
            query_data[0]['e_Rmag'] = 0.02
            query_data[0]['e_gmag'] = 0.02
            query_data[0]['e_rmag'] = 0.02
            query_data[0]['e_imag'] = 0.02
            if len(query_data) > 0:
                return query_data[0]
            else:
                return None
        
        if catalog_name == 'APASS':
            self.catalog_type = 'APASS'
            self.helper.print('Start APASS query...', verbose)
            data = apass_query(ra_deg = float(self.ra), dec_deg = float(self.dec), rad_deg =  np.max([self.fov_ra, self.fov_dec]))
        elif catalog_name == 'PS1':
            self.catalog_type = 'PS1'
            self.helper.print('Start PS1 query...', verbose) 
            data = ps1_query(ra_deg = float(self.ra), dec_deg = float(self.dec), rad_deg =  np.max([self.fov_ra, self.fov_dec]))
        elif catalog_name == 'SDSS':
            self.catalog_type = 'SDSS'
            self.helper.print('Start SDSS query...', verbose)
            data = sdss_query(ra_deg = float(self.ra), dec_deg = float(self.dec), rad_deg =  np.max([self.fov_ra, self.fov_dec]))
        elif catalog_name == 'SMSS':
            self.catalog_type = 'SMSS'
            self.helper.print('Start SMSS query...', verbose)
            data = smss_query(ra_deg = float(self.ra), dec_deg = float(self.dec), rad_deg =  np.max([self.fov_ra, self.fov_dec]))
        elif catalog_name == 'GAIA':
            self.catalog_type = 'GAIA'
            self.helper.print('Start GAIA query...', verbose)
            data = gaia_query(ra_deg = float(self.ra), dec_deg = float(self.dec), rad_deg =  np.max([self.fov_ra, self.fov_dec]))
        else:
            raise ValueError(f'{catalog_name} is not supported (supported catalogs: APASS, PS1, SDSS, SMSS, GAIA)') 
        
        if not data:
            raise ValueError(f'{catalog_name} is not found in the catalog')              
        
        return data

    def get_reference_sources(self, mag_lower : float = 10, mag_upper : float = 20, **kwargs):
        if not self.data:
            raise RuntimeError(f'No catalog data found for {self.objname}')
        
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
        # For GAIAXP_CORR_LAMOST cut
        cutline_gaiaxp_corr_lamost = {"pmra": [-20, 20], "pmdec": [-20, 20], "bp-rp": [0.0, 1.5], "g_mean": [mag_lower, mag_upper]}
        if self.catalog_type == 'APASS':
            cutline = cutline_apass
        elif self.catalog_type == 'GAIA':
            cutline = cutline_gaia
        elif self.catalog_type == 'GAIAXP':
            cutline = cutline_gaiaxp
        elif self.catalog_type == 'PS1':
            cutline = cutline_ps1
        elif self.catalog_type == 'SMSS':
            cutline = cutline_smss
        elif self.catalog_type == 'GAIAXP_CORR_LAMOST':
            cutline = cutline_gaiaxp_corr_lamost
        else:
            raise ValueError('Invalid catalog type: %s' % self.catalog_type)
        cutline = {**cutline, **kwargs}
        
        ref_sources = self.data
        applied_kwargs = []
        for key, value in cutline.items():
            if key in ref_sources.colnames:
                applied_kwargs.append({key : [value]})
                ref_sources = ref_sources[(ref_sources[key] > value[0]) & (ref_sources[key] < value[1])]
        return ref_sources, applied_kwargs    
    
    

    def _get_catalog(self, catalog_type : str, verbose : bool = True):

        if catalog_type == 'GAIAXP':
            self._get_GAIAXP(verbose = verbose)
        elif catalog_type == 'GAIA':
            self._get_GAIA(verbose = verbose)
        elif catalog_type == 'APASS':
            self._get_APASS(verbose = verbose)
        elif catalog_type == 'PS1':
            self._get_PS1(verbose = verbose)
        elif catalog_type == 'SDSS':
            self._get_SDSS(verbose = verbose)
        elif catalog_type == 'SMSS':
            self._get_SMSS(verbose = verbose)
        elif catalog_type == 'GAIAXP_CORR_LAMOST':
            self._get_GAIAXP_CORR_LAMOST(verbose = verbose)
        else:
            raise ValueError('Invalid catalog type: %s' % catalog_type)
    
    def _get_GAIAXP(self, verbose = False):

        def GAIAXP_format(GAIAXP_catalog) -> Table:
            original = ('source_id', 'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp', 'mag_u', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'mag_m375w', 'mag_m400', 'mag_m412', 'mag_m425', 'mag_m425w', 'mag_m437', 'mag_m450', 'mag_m462', 'mag_m475', 'mag_m487', 'mag_m500', 'mag_m512', 'mag_m525', 'mag_m537', 'mag_m550', 'mag_m562', 'mag_m575', 'mag_m587', 'mag_m600', 'mag_m612', 'mag_m625', 'mag_m637', 'mag_m650', 'mag_m662', 'mag_m675', 'mag_m687', 'mag_m700', 'mag_m712', 'mag_m725', 'mag_m737', 'mag_m750', 'mag_m762', 'mag_m775', 'mag_m787', 'mag_m800', 'mag_m812', 'mag_m825', 'mag_m837', 'mag_m850', 'mag_m862', 'mag_m875', 'mag_m887')
            format_ = ('id', 'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'g_mean', 'bp-rp', 'u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'm375w_mag', 'm400_mag', 'm412_mag', 'm425_mag', 'm425w_mag', 'm437_mag', 'm450_mag', 'm462_mag', 'm475_mag', 'm487_mag', 'm500_mag', 'm512_mag', 'm525_mag', 'm537_mag', 'm550_mag', 'm562_mag', 'm575_mag', 'm587_mag', 'm600_mag', 'm612_mag', 'm625_mag', 'm637_mag', 'm650_mag', 'm662_mag', 'm675_mag', 'm687_mag', 'm700_mag', 'm712_mag', 'm725_mag', 'm737_mag', 'm750_mag', 'm762_mag', 'm775_mag', 'm787_mag', 'm800_mag', 'm812_mag', 'm825_mag', 'm837_mag', 'm850_mag', 'm862_mag', 'm875_mag', 'm887_mag')
            GAIAXP_catalog.rename_columns(original, format_)
            formatted_catalog = self._match_digit_tbl(GAIAXP_catalog)
            return formatted_catalog

        if self.filename:
            self.helper.print(f'Catalog file found in archive: {self.filename}', verbose)
            data = self._get_catalog_from_archive(filename = self.filename)
        else:
            raise ValueError(f'{self.objname} does not exist in GAIAXP catalog')
                    
        self.data = None
        if data:
            self.data = GAIAXP_format(data)
            
    def _get_GAIAXP_CORR_LAMOST(self, verbose = False):

        def GAIAXP_CORR_LAMOST_format(GAIAXP_catalog) -> Table:
            # Already formatted
            # original = ()
            # format_ = ()
            # GAIAXP_catalog.rename_columns(original, format_)
            formatted_catalog = self._match_digit_tbl(GAIAXP_catalog)
            return formatted_catalog

        if self.filename:
            self.helper.print(f'Catalog file found in archive: {self.filename}', verbose)
            data = self._get_catalog_from_archive(filename = self.filename)
        else:
            raise ValueError(f'{self.objname} does not exist in GAIAXP catalog')
                    
        self.data = None
        if data:
            self.data = GAIAXP_CORR_LAMOST_format(data)

    def _get_GAIA(self, verbose = True):
        def GAIA_format(GAIA_catalog) -> Table:
            original = ('RA_ICRS', 'DE_ICRS', 'Bmag', 'e_Bmag', 'BFlag', 'Vmag', 'e_Vmag', 'VFlag', 'Rmag', 'e_Rmag', 'RFlag', 'gmag', 'e_gmag', 'gFlag', 'rmag', 'e_rmag', 'rFlag', 'imag', 'e_imag', 'iFlag')
            format_ = ('ra', 'dec', 'B_mag', 'e_B_mag', 'B_flag', 'V_mag', 'e_Vmag', 'V_flag', 'R_mag', 'e_Rmag', 'R_flag', 'g_mag', 'e_gmag', 'g_flag', 'r_mag', 'e_rmag', 'r_flag', 'i_mag', 'e_imag', 'i_flag')
            GAIA_catalog.rename_columns(original, format_)
            if 'E_BP_RP_corr' in GAIA_catalog.colnames:
                GAIA_catalog.rename_columns(['E_BP_RP_corr'], ['c_star'])
            else:
                GAIA_catalog['c_star'] = 0
            formatted_catalog = self._match_digit_tbl(GAIA_catalog)
            return formatted_catalog

        # If filename is defined by _register_objinfo function (meaning located in catalog_archive), Query from archive
        if self.filename:
            self.helper.print(f'Catalog file found in archive: {self.filename}', verbose)
            data = self._get_catalog_from_archive(filename = self.filename)
        # Else, query object in the catalog and save to catalog_archive
        else:
            try:
                # Try query and save to catalog_archive
                data = self._query(catalog_name = 'GAIA', verbose = verbose)
                filename = f'{self.objname}_GAIA.csv'
                catalog_file = os.path.join(self.helper.config['CATALOG_DIR'], self.catalog_type, filename)
                os.makedirs(os.path.join(self.helper.config['CATALOG_DIR'], self.catalog_type), exist_ok = True)
                data.write(catalog_file, format ='csv', overwrite = True)
                summary_tbl = ascii.read(self.summary_path, format = 'fixed_width')
                catalog_file = Path(catalog_file)
                file_info = catalog_file.stat()
                rel_path = catalog_file.relative_to(self.helper.config['CATALOG_DIR'])
                summary_tbl.add_row([str(rel_path), self.objname, self.catalog_type, self.ra, self.dec, self.fov_ra, self.fov_dec, self.fov_ra * self.fov_dec, file_info.st_size,  time.strftime("%Y-%m-%d %H:%M:%S.%f"[:-3], time.localtime(file_info.st_mtime)), self.catalog_version])
                summary_tbl.write(self.summary_path, format = 'ascii.fixed_width', overwrite = True)
                self.helper.print(f'GAIA catalog saved and registered to {catalog_file}', verbose)
            except:
                # Elase, return Error
                raise ValueError(f'{self.objname} does not exist in GAIA catalog')
        
        # After getting the data, format the data
        self.data = None
        if data:
            self.data = GAIA_format(data)
       
    def _get_APASS(self, verbose = True):
        def APASS_format(APASS_catalog) -> Table:
            original = ('RAJ2000','DEJ2000','e_RAJ2000','e_DEJ2000','Bmag','e_Bmag','Vmag','e_Vmag',"g'mag","e_g'mag","r'mag","e_r'mag","i'mag","e_i'mag")
            format_ = ('ra','dec','e_ra','e_dec','B_mag','e_B_mag','V_mag','e_V_mag','g_mag','e_g_mag','r_mag','e_r_mag','i_mag','e_i_mag')
            APASS_catalog.rename_columns(original, format_)
            formatted_catalog = self._match_digit_tbl(APASS_catalog)
            return formatted_catalog

        # If filename is defined by _register_objinfo function (meaning located in catalog_archive), Query from archive
        if self.filename:
            self.helper.print(f'Catalog file found in archive: {self.filename}', verbose)
            data = self._get_catalog_from_archive(filename = self.filename)
        # Else, query object in the catalog and save to catalog_archive
        else:
            try:
                # Try query and save to catalog_archive
                data = self._query(catalog_name = 'APASS', verbose = verbose)
                filename = f'{self.objname}_APASS.csv'
                catalog_file = os.path.join(self.helper.config['CATALOG_DIR'], self.catalog_type, filename)
                os.makedirs(os.path.join(self.helper.config['CATALOG_DIR'], self.catalog_type), exist_ok = True)
                data.write(catalog_file, format ='csv', overwrite = True)
                summary_tbl = ascii.read(self.summary_path, format = 'fixed_width')
                catalog_file = Path(catalog_file)
                file_info = catalog_file.stat()
                rel_path = catalog_file.relative_to(self.helper.config['CATALOG_DIR'])
                summary_tbl.add_row([str(rel_path), self.objname, self.catalog_type, self.ra, self.dec, self.fov_ra, self.fov_dec, self.fov_ra * self.fov_dec, file_info.st_size,  time.strftime("%Y-%m-%d %H:%M:%S.%f"[:-3], time.localtime(file_info.st_mtime)), self.catalog_version])
                summary_tbl.write(self.summary_path, format = 'ascii.fixed_width', overwrite = True)
                self.helper.print(f'APASS catalog saved and registered to {catalog_file}', verbose)
            except:
                # Elase, return Error
                raise ValueError(f'{self.objname} does not exist in APASS catalog')
        
        # After getting the data, format the data
        self.data = None
        if data:
            self.data = APASS_format(data)
    
    def _get_PS1(self, verbose = True):
        def PS1_format(PS1_catalog) -> Table:
            original = ('objID','RAJ2000','DEJ2000','e_RAJ2000','e_DEJ2000','gmag','e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag','ymag','e_ymag','gKmag','e_gKmag','rKmag','e_rKmag','iKmag','e_iKmag','zKmag','e_zKmag','yKmag','e_yKmag')
            format_ = ('ID','ra','dec','e_ra','e_dec','g_mag','e_g_mag','r_mag','e_r_mag','i_mag','e_i_mag','z_mag','e_z_mag','y_mag','e_y_mag','g_Kmag','e_g_Kmag','r_Kmag','e_r_Kmag','i_Kmag','e_i_Kmag','z_Kmag','e_z_Kmag','y_Kmag','e_y_Kmag')
            PS1_catalog.rename_columns(original, format_)
            formatted_catalog = self._match_digit_tbl(PS1_catalog)
            return formatted_catalog
        
        # If filename is defined by _register_objinfo function (meaning located in catalog_archive), Query from archive
        if self.filename:
            self.helper.print(f'Catalog file found in archive: {self.filename}', verbose)
            data = self._get_catalog_from_archive(filename = self.filename)
        # Else, query object in the catalog and save to catalog_archive
        else:
            try:
                # Try query and save to catalog_archive
                data = self._query(catalog_name = 'PS1', verbose = verbose)
                filename = f'{self.objname}_PS1.csv'
                catalog_file = os.path.join(self.helper.config['CATALOG_DIR'], self.catalog_type, filename)
                os.makedirs(os.path.join(self.helper.config['CATALOG_DIR'], self.catalog_type), exist_ok = True)
                data.write(catalog_file, format ='csv', overwrite = True)
                summary_tbl = ascii.read(self.summary_path, format = 'fixed_width')
                catalog_file = Path(catalog_file)
                file_info = catalog_file.stat()
                rel_path = catalog_file.relative_to(self.helper.config['CATALOG_DIR'])
                summary_tbl.add_row([str(rel_path), self.objname, self.catalog_type, self.ra, self.dec, self.fov_ra, self.fov_dec, self.fov_ra * self.fov_dec, file_info.st_size,  time.strftime("%Y-%m-%d %H:%M:%S.%f"[:-3], time.localtime(file_info.st_mtime)), self.catalog_version])
                summary_tbl.write(self.summary_path, format = 'ascii.fixed_width', overwrite = True)
                self.helper.print(f'PS1 catalog saved and registered to {catalog_file}', verbose)
            except:
                # Elase, return Error
                raise ValueError(f'{self.objname} does not exist in PS1 catalog')
        
        # After getting the data, format the data
        self.data = None
        if data:
            self.data = PS1_format(data)

    def _get_SMSS(self, verbose = True):
        def SMSS_format(SMSS_catalog) -> Table:
            original = ('ObjectId','RAICRS','DEICRS','Niflags','flags','Ngood','Ngoodu','Ngoodv','Ngoodg','Ngoodr','Ngoodi','Ngoodz','ClassStar','uPSF','e_uPSF','vPSF','e_vPSF','gPSF','e_gPSF','rPSF','e_rPSF','iPSF','e_iPSF','zPSF','e_zPSF')
            format_ = ('ID','ra','dec','nimflag','flag','ngood','ngoodu','ngoodv','ngoodg','ngoodr','ngoodi','ngoodz','class_star','u_mag','e_u_mag','v_mag','e_v_mag','g_mag','e_g_mag','r_mag','e_r_mag','i_mag','e_i_mag','z_mag','e_z_mag')
            SMSS_catalog.rename_columns(original, format_)
            formatted_catalog = self._match_digit_tbl(SMSS_catalog)
            return formatted_catalog

        # If filename is defined by _register_objinfo function (meaning located in catalog_archive), Query from archive
        if self.filename:
            self.helper.print(f'Catalog file found in archive: {self.filename}', verbose)
            data = self._get_catalog_from_archive(filename = self.filename)
        # Else, query object in the catalog and save to catalog_archive
        else:
            try:
                # Try query and save to catalog_archive
                data = self._query(catalog_name = 'SMSS', verbose = verbose)
                filename = f'{self.objname}_SMSS.csv'
                catalog_file = os.path.join(self.helper.config['CATALOG_DIR'], self.catalog_type, filename)
                os.makedirs(os.path.join(self.helper.config['CATALOG_DIR'], self.catalog_type), exist_ok = True)
                data.write(catalog_file, format ='csv', overwrite = True)
                summary_tbl = ascii.read(self.summary_path, format = 'fixed_width')
                catalog_file = Path(catalog_file)
                file_info = catalog_file.stat()
                rel_path = catalog_file.relative_to(self.helper.config['CATALOG_DIR'])
                summary_tbl.add_row([str(rel_path), self.objname, self.catalog_type, self.ra, self.dec, self.fov_ra, self.fov_dec, self.fov_ra * self.fov_dec, file_info.st_size,  time.strftime("%Y-%m-%d %H:%M:%S.%f"[:-3], time.localtime(file_info.st_mtime)), self.catalog_version])
                summary_tbl.write(self.summary_path, format = 'ascii.fixed_width', overwrite = True)
                self.helper.print(f'SMSS catalog saved and registered to {catalog_file}', verbose)
            except:
                # Elase, return Error
                raise ValueError(f'{self.objname} does not exist in SMSS catalog')
        
        # After getting the data, format the data
        self.data = None
        if data:
            self.data = SMSS_format(data)

    def _get_SDSS(self, verbose = True):
        def SDSS_format(SDSS_catalog) -> Table:
            original = ('RA_ICRS','DE_ICRS','umag','e_umag','gmag','e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag')
            format_ = ('ra','dec','umag','e_umag','gmag','e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag')
            SDSS_catalog.rename_columns(original, format_)
            formatted_catalog = self._match_digit_tbl(SDSS_catalog)
            return formatted_catalog

        # If filename is defined by _register_objinfo function (meaning located in catalog_archive), Query from archive
        if self.filename:
            self.helper.print(f'Catalog file found in archive: {self.filename}', verbose)
            data = self._get_catalog_from_archive(filename = self.filename)
        # Else, query object in the catalog and save to catalog_archive
        else:
            try:
                # Try query and save to catalog_archive
                data = self._query(catalog_name = 'SDSS', verbose = verbose)
                filename = f'{self.objname}_SDSS.csv'
                catalog_file = os.path.join(self.helper.config['CATALOG_DIR'], self.catalog_type, filename)
                os.makedirs(os.path.join(self.helper.config['CATALOG_DIR'], self.catalog_type), exist_ok = True)
                data.write(catalog_file, format ='csv', overwrite = True)
                summary_tbl = ascii.read(self.summary_path, format = 'fixed_width')
                catalog_file = Path(catalog_file)
                file_info = catalog_file.stat()
                rel_path = catalog_file.relative_to(self.helper.config['CATALOG_DIR'])
                summary_tbl.add_row([str(rel_path), self.objname, self.catalog_type, self.ra, self.dec, self.fov_ra, self.fov_dec, self.fov_ra * self.fov_dec, file_info.st_size,  time.strftime("%Y-%m-%d %H:%M:%S.%f"[:-3], time.localtime(file_info.st_mtime)), self.catalog_version])
                summary_tbl.write(self.summary_path, format = 'ascii.fixed_width', overwrite = True)
                self.helper.print(f'SDSS catalog saved and registered to {catalog_file}', verbose)
            except:
                # Elase, return Error
                raise ValueError(f'{self.objname} does not exist in SDSS catalog')
        
        # After getting the data, format the data
        self.data = None
        if data:
            self.data = SDSS_format(data)

    def _match_digit_tbl(self, tbl):
        for column in tbl.columns:
            if tbl[column].dtype == 'float64':
                tbl[column].format = '{:.5f}'
        return tbl

    def _get_catalog_from_archive(self, filename : str):
        catalog_file = os.path.join(self.helper.config['CATALOG_DIR'], filename)
        is_exist = os.path.exists(catalog_file)
        
        if is_exist:
            data = ascii.read(catalog_file, format = 'csv')
            return data
        else:
            return None

    def _get_cataloginfo_by_coord(self, 
                                 coord: SkyCoord, 
                                 fov_ra: float = 1.5, 
                                 fov_dec: float = 1.5, 
                                 overlapped_fraction: float = 0.9,
                                 verbose: bool = False) -> Table:
        """
        Retrieves catalog information based on coordinates.
        Requires fov_ra and fov_dec to calculate overlap.
        Returns catalogs with sufficient overlap based on the given fraction.
        """

        try:
            catalog_summary_tbl = ascii.read(self.summary_path, format='fixed_width')
            catalog_coords = SkyCoord(ra=catalog_summary_tbl['ra'], 
                                      dec=catalog_summary_tbl['dec'], 
                                      unit=(u.deg, u.deg))

            # Cut tiles into 10deg x 10deg regions
            ra_min, ra_max = coord.ra.deg - 5, coord.ra.deg + 5
            dec_min, dec_max = coord.dec.deg - 5, coord.dec.deg + 5
            cut_tiles_mask = (
                (catalog_summary_tbl['ra'] >= ra_min) & (catalog_summary_tbl['ra'] <= ra_max) &
                (catalog_summary_tbl['dec'] >= dec_min) & (catalog_summary_tbl['dec'] <= dec_max)
            )
            catalog_summary_tbl = catalog_summary_tbl[cut_tiles_mask]
            catalog_coords = catalog_coords[cut_tiles_mask]

            overlap_catalogs = []
            overlap_fractions = []
            for idx, (cat_ra, cat_dec, cat_fov_ra, cat_fov_dec) in enumerate(zip(
                    catalog_summary_tbl['ra'], 
                    catalog_summary_tbl['dec'], 
                    catalog_summary_tbl['fov_ra'], 
                    catalog_summary_tbl['fov_dec'])):
                target_polygon = Polygon([  
                    (coord.ra.deg - fov_ra / 2, coord.dec.deg - fov_dec / 2),
                    (coord.ra.deg + fov_ra / 2, coord.dec.deg - fov_dec / 2),
                    (coord.ra.deg + fov_ra / 2, coord.dec.deg + fov_dec / 2),
                    (coord.ra.deg - fov_ra / 2, coord.dec.deg + fov_dec / 2)
                ])
                tile_polygon = Polygon([  
                    (cat_ra - cat_fov_ra / 2, cat_dec - cat_fov_dec / 2),
                    (cat_ra + cat_fov_ra / 2, cat_dec - cat_fov_dec / 2),
                    (cat_ra + cat_fov_ra / 2, cat_dec + cat_fov_dec / 2),
                    (cat_ra - cat_fov_ra / 2, cat_dec + cat_fov_dec / 2)
                ])
                if target_polygon.intersects(tile_polygon):
                    intersection = target_polygon.intersection(tile_polygon)
                    target_area = fov_ra * fov_dec
                    fraction_overlap = intersection.area / target_area
                    if fraction_overlap >= overlapped_fraction:
                        overlap_catalogs.append(catalog_summary_tbl[idx])
                        overlap_fractions.append(fraction_overlap)

            if len(overlap_catalogs) > 0:
                return vstack(overlap_catalogs)
            else:
                raise ValueError("No catalog found with sufficient overlap.")
        except Exception as e:
            raise RuntimeError(f'Failed to access catalog summary: {e}')

    def _get_cataloginfo_by_objname(self, objname, catalog_type, catalog_version):
        catalog_summary_file = os.path.join(self.helper.config['CATALOG_DIR'], 'summary.ascii_fixed_width')
        catalog_summary_tbl = ascii.read(catalog_summary_file, format = 'fixed_width')
        
        idx = (catalog_summary_tbl['objname'] == objname) & (catalog_summary_tbl['catalog_type'] == catalog_type) & (catalog_summary_tbl['catalog_version'] == catalog_version)
        if np.sum(idx) > 0:
            matched_info = catalog_summary_tbl[idx]
            return matched_info
        else:
            raise ValueError(f"{objname} not found in catalog_summary")

    def _register_objinfo(self, objname, ra, dec, fov_ra, fov_dec, catalog_type, catalog_version, verbose):
        self.objname = objname
        self.ra = ra
        self.dec = dec
        self.fov_ra = fov_ra
        self.fov_dec = fov_dec
        self.catalog_type = catalog_type
        self.catalog_version = catalog_version
        
        # 1. Check if neither objname nor (ra, dec) are provided
        if (objname is None) and (ra is None) and (dec is None):
            raise ValueError('objname or (ra, dec) must be provided')

        # 2. If objname is given but ra and dec are not, retrieve coordinates
        if objname is not None:
            try:
                catinfo = self._get_cataloginfo_by_objname(objname = objname, catalog_type = catalog_type, catalog_version = catalog_version)
                self.ra = catinfo['ra'][0]
                self.dec = catinfo['dec'][0]
                self.fov_ra = catinfo['fov_ra'][0]
                self.fov_dec = catinfo['fov_dec'][0]
                self.filename = catinfo['file'][0]
                self.filepath = os.path.join(self.helper.config['CATALOG_DIR'], self.filename)
            except:
                try:
                    coord = self._query_coord_from_objname(objname = objname, verbose = verbose)
                    self.ra = coord.ra.deg
                    self.dec = coord.dec.deg
                except:
                    raise ValueError(f"Failed to query coordinates for {objname}")

        # 3. If objname is not provided, generate it using the coordinate format
        else:
            if ra is not None and dec is not None:
                coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

                try:
                    catinfo = self._get_cataloginfo_by_coord(coord = coord, fov_ra = fov_ra, fov_dec = fov_dec, overlapped_fraction = self.overlapped_fraction)
                    self.objname = catinfo['objname'][0]
                    self.ra = catinfo['ra'][0]
                    self.dec = catinfo['dec'][0]
                    self.catalog_version = catinfo['catalog_version'][0]
                    self.fov_ra = catinfo['fov_ra'][0]
                    self.fov_dec = catinfo['fov_dec'][0]
                    self.filename = catinfo['file'][0]
                    self.filepath = os.path.join(self.helper.config['CATALOG_DIR'], self.filename)
                    self.is_registered = True
                except:
                    ra_hms = coord.ra.hms
                    dec_dms = coord.dec.dms
                    self.objname = f'J{int(ra_hms.h):02}{int(ra_hms.m):02}{ra_hms.s:05.2f}' \
                                f'{int(dec_dms.d):+03}{int(abs(dec_dms.m)):02}{abs(dec_dms.s):04.1f}' 
        
        if (self.objname is None) or (self.ra is None) or (self.dec is None):
            raise ValueError('objname, ra, and dec must be provided')
        
    def _query_coord_from_objname(self, objname, verbose) -> SkyCoord:
        from astroquery.simbad import Simbad

        # Create a custom Simbad instance with the necessary fields
        self.helper.print(f'Querying coordinates for {objname} from SIMBAD', verbose)
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('ra', 'dec')

        # Query an object (e.g., "Vega")
        result_table = custom_simbad.query_object(objname)

        # Extract coordinates
        if result_table is not None:
            ra = result_table['ra'][0]  # Right Ascension
            dec = result_table['dec'][0]  # Declination
            coord = SkyCoord(ra, dec, unit = (u.deg, u.deg))
            self.helper.print(f'Coordinates for {objname} from SIMBAD: {coord}', verbose)
            return coord
        else:
            raise ValueError("Object not found in SIMBAD.")

    

# %%
if __name__ == '__main__':
    from astropy.io import ascii
    self = SkyCatalog(objname = 'NGC1566', catalog_type = 'APASS', catalog_version = 'v1')
# %%
