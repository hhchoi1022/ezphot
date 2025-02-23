#%%
import numpy as np
import pandas as pd
import os
import inspect
import glob
from astropy.io import ascii
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
#%%

class Catalog():
    
    def __init__(self,
                 target_name : str = None,
                 ra = None,
                 dec = None,
                 radius_deg : float = 2,
                 maxmag = 20,
                 minmag = 10,
                 maxsources = 1000000,
                 ):
        
        self.fieldinfo = dict()
        if (ra == None) | (dec == None):
            coord = self.get_target_coord(target_name)
            ra = coord.ra.deg
            dec = coord.dec.deg
        if target_name == None:
            target_name = f'[RA,Dec = {round(ra, 2)},{round(dec,2)}]'
        self.fieldinfo['target'] = target_name
        self.fieldinfo['ra'] = ra
        self.fieldinfo['dec'] = dec
        self.fieldinfo['radius'] = radius_deg
        self.fieldinfo['maxmag'] = maxmag
        self.fieldinfo['minmag'] = minmag
        self.fieldinfo['maxsources'] = maxsources
        self.APASS = self.get_APASS()
        self.PS1 = self.get_PS1()
        self.SDSS = self.get_SDSS()
        self.SMSS = self.get_SMSS()
        self.GAIA = self.get_GAIA()
        self.dict = dict(APASS = self.APASS, PS1 = self.PS1, SDSS = self.SDSS, SMSS = self.SMSS, GAIA = self.GAIA)
        	
    @property
    def catpath(self):
        # Get the file where the class is defined
        file_path = inspect.getfile(Catalog)
        
        # Convert the file path to an absolute path using os.path.abspath
        absolute_path = os.path.abspath(file_path)

        path_dir = os.path.dirname(absolute_path)
        
        path_cat = os.path.join(path_dir,'catalog_archive')
        
        return path_cat

    def __repr__(self):
        txt = f"{self.fieldinfo['target']} Catalog[APASS = {self.APASS.is_exist}, PS1 = {self.PS1.is_exist}, SMSS = {self.SMSS.is_exist}, SDSS = {self.SDSS.is_exist}]"
        return txt
    
    def get_GAIA(self):
        def GAIA_format(GAIA_catalog) -> Table:
            original = ('RA_ICRS', 'DE_ICRS', 'Bmag', 'e_Bmag', 'BFlag', 'Vmag', 'e_Vmag', 'VFlag', 'Rmag', 'e_Rmag', 'RFlag', 'gmag', 'e_gmag', 'gFlag', 'rmag', 'e_rmag', 'rFlag', 'imag', 'e_imag', 'iFlag')
            format_ = ('ra', 'dec', 'B_mag', 'e_B_mag', 'B_flag', 'V_mag', 'e_Vmag', 'V_flag', 'R_mag', 'e_Rmag', 'R_flag', 'g_mag', 'e_gmag', 'g_flag', 'r_mag', 'e_rmag', 'r_flag', 'i_mag', 'e_imag', 'i_flag')
            GAIA_catalog.rename_columns(original, format_)
            if 'E_BP_RP_corr' in GAIA_catalog.colnames:
                GAIA_catalog.rename_columns(['E_BP_RP_corr'], ['c_star'])
            else:
                GAIA_catalog['c_star'] = 0
            formatted_catalog = self.match_digit_tbl(GAIA_catalog)
            return formatted_catalog

        class GAIA:
            def __repr__(self):
                txt = '[Attributes]\n'+''.join(f"GAIA.{key}\n" for key in self.__dict__.keys())
                return txt
            
        data = self.get_catalog(catalog_name='GAIA')
        if not data:
            is_exist = False
        else:
            is_exist = True

        cat = GAIA()
        cat.is_exist = is_exist
        if cat.is_exist:
            cat.data = GAIA_format(data)
            cat.conversion = dict()
        return cat
       
    def get_APASS(self):
        def APASS_format(APASS_catalog) -> Table:
            original = ('RAJ2000','DEJ2000','e_RAJ2000','e_DEJ2000','Bmag','e_Bmag','Vmag','e_Vmag','g_mag','e_g_mag','r_mag','e_r_mag','i_mag','e_i_mag')
            format_ = ('ra','dec','e_ra','e_dec','B_mag','e_B_mag','V_mag','e_V_mag','g_mag','e_g_mag','r_mag','e_r_mag','i_mag','e_i_mag')
            APASS_catalog.rename_columns(original, format_)
            formatted_catalog = self.match_digit_tbl(APASS_catalog)
            return formatted_catalog

        def APASS_to_PANSTARRS1(APASS_catalog):
            '''
            parameters
            ----------
            {APASS catalog filepath}
            
            returns 
            -------
            Converted APASS catalog in PS1 magnitude  
            
            notes 
            -----
            Conversion equation : https://arxiv.org/pdf/1809.09157.pdf (Torny(2018))
            -----
            '''
                        
            ra = APASS_catalog['ra']
            dec = APASS_catalog['dec']
            g = APASS_catalog['g_mag']
            r = APASS_catalog['r_mag']
            i = APASS_catalog['i_mag']
            
            e_g = APASS_catalog['e_g_mag']
            e_r = APASS_catalog['e_r_mag']
            e_i = APASS_catalog['e_i_mag']

            gr = g-r
            ri = r-i
            
            g_c = g - 0.009 - 0.061*gr
            r_c = r + 0.065 - 0.026*gr
            i_c = i - 0.015 - 0.068*ri
            
            e_gr = np.sqrt((e_g)**2+(e_r)**2)
            
            e_g_c = np.sqrt((e_g)**2 + 0.026**2 + (0.061*e_gr)**2)
            e_r_c = np.sqrt((e_r)**2 + 0.027**2 + (0.026*e_gr)**2)
            e_i_c = np.sqrt((e_i)**2 + 0.045**2 + (0.068*e_gr)**2)

            source = {'ra':ra,
                    'dec':dec,
                    'g_mag':g_c,
                    'e_g_mag':e_g_c,
                    'r_mag':r_c,
                    'e_r_mag':e_r_c,
                    'i_mag':i_c,
                    'e_i_mag':e_i_c
                    }
            
            atable = pd.DataFrame(source)
            ctable = Table.from_pandas(atable)
            result = self.match_digit_tbl(ctable)
            return result

        def APASS_to_JH(APASS_catalog):
            '''
            parameters
            ----------
            {APASS catalog filepath}
            
            returns 
            -------
            Converted APASS catalog in Johnson-Cousins magnitude  
            
            notes 
            -----
            Conversion equation : https://arxiv.org/pdf/astro-ph/0609121v1.pdf (Jordi(2006))
            More information about SDSS conversion : https://www.sdss.org/dr12/algorithms/sdssubvritransform/
            -----
            '''
            
            ra = APASS_catalog['ra']
            dec = APASS_catalog['dec']
            B = APASS_catalog['B_mag']
            V = APASS_catalog['V_mag']
            g = APASS_catalog['g_mag']
            r = APASS_catalog['r_mag']
            i = APASS_catalog['i_mag']

            e_B = APASS_catalog['e_B_mag']
            e_V = APASS_catalog['e_V_mag']
            e_g = APASS_catalog['e_g_mag']
            e_r = APASS_catalog['e_r_mag']
            e_i = APASS_catalog['e_i_mag']

            ri = r-i

            e_ri = np.sqrt(e_r**2+e_i**2)
            
            R = r - 0.153*ri - 0.117
            I = R -0.930*ri - 0.259

            e_R = np.sqrt(e_r**2+ 0.003**2 + (0.153*e_ri)**2)
            e_I = np.sqrt(e_r**2+ 0.002**2 + (0.930*e_ri)**2)
            
            source = {'ra':ra,
                    'dec':dec,
                    'B_mag':B,
                    'e_B_mag':e_B,
                    'V_mag':V,
                    'e_V_mag':e_V,
                    'R_mag':R,
                    'e_R_mag':e_R,
                    'I_mag':I,
                    'e_I_mag':e_I
                    }
            
            ptable = pd.DataFrame(source)
            ctable = Table.from_pandas(ptable)
            result = self.match_digit_tbl(ctable)
            return result
        
        def PANSTARRS1_to_SDSS(PANSTARR_catalog):
            '''
            parameters
            ----------
            {PanSTARRS DR1 catalog filepath}
            
            returns 
            -------
            Converted PanSTARRS catalog in SDSS magnitude  
            
            notes 
            -----
            Conversion equation : https://iopscience.iop.org/article/10.1088/0004-637X/750/2/99/pdf (Torny(2012))
            -----
            '''
                        
            ra = PANSTARR_catalog['ra']
            dec = PANSTARR_catalog['dec']
            g = PANSTARR_catalog['g_mag']
            r = PANSTARR_catalog['r_mag']
            i = PANSTARR_catalog['i_mag']
            
            e_g = PANSTARR_catalog['e_g_mag']
            e_r = PANSTARR_catalog['e_r_mag']
            e_i = PANSTARR_catalog['e_i_mag']
            
            gr = g-r
            ri = r-i
            
            g_c = g + 0.014 + 0.162*gr
            r_c = r - 0.001 + 0.011*gr
            i_c = i - 0.004 + 0.020*gr
        
            e_gr = np.sqrt((e_g)**2+(e_r)**2)
            e_ri = np.sqrt((e_r)**2+(e_i)**2)
            
            e_g_c = np.sqrt((e_g)**2 + 0.009**2 + (0.162*e_gr)**2)
            e_r_c = np.sqrt((e_r)**2 + 0.004**2 + (0.011*e_gr)**2)
            e_i_c = np.sqrt((e_i)**2 + 0.005**2 + (0.020*e_gr)**2)
            
            source = {'ra':ra,
                    'dec':dec,
                    'g_mag':g_c,
                    'e_g_mag':e_g_c,
                    'r_mag':r_c,
                    'e_r_mag':e_r_c,
                    'i_mag':i_c,
                    'e_i_mag':e_i_c,
                    }
            ptable = pd.DataFrame(source)
            ctable = Table.from_pandas(ptable)
            result = self.match_digit_tbl(ctable)
            return result

        class APASS:
            def __repr__(self):
                txt = '[Attributes]\n'+''.join(f"APASS.{key}\n" for key in self.__dict__.keys())
                return txt
            
        data = self.get_catalog(catalog_name='APASS')
        if not data:
            is_exist = False
        else:
            is_exist = True

        cat = APASS()
        cat.is_exist = is_exist
        if cat.is_exist:
            cat.data = APASS_format(data)
            cat.conversion = dict()
            cat.conversion['PS1'] = APASS_to_PANSTARRS1(cat.data)
            cat.conversion['JH'] = APASS_to_JH(cat.data)
            cat.conversion['SDSS'] = PANSTARRS1_to_SDSS(cat.conversion['PS1'])
        return cat
    
    def get_PS1(self):
        def PS1_format(PS1_catalog) -> Table:
            original = ('objID','RAJ2000','DEJ2000','e_RAJ2000','e_DEJ2000','gmag','e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag','ymag','e_ymag','gKmag','e_gKmag','rKmag','e_rKmag','iKmag','e_iKmag','zKmag','e_zKmag','yKmag','e_yKmag')
            format_ = ('ID','ra','dec','e_ra','e_dec','g_mag','e_g_mag','r_mag','e_r_mag','i_mag','e_i_mag','z_mag','e_z_mag','y_mag','e_y_mag','g_Kmag','e_g_Kmag','r_Kmag','e_r_Kmag','i_Kmag','e_i_Kmag','z_Kmag','e_z_Kmag','y_Kmag','e_y_Kmag')
            PS1_catalog.rename_columns(original, format_)
            formatted_catalog = self.match_digit_tbl(PS1_catalog)
            return formatted_catalog
        
        def PANSTARRS1_to_SDSS(PANSTARR_catalog):
            '''
            parameters
            ----------
            {PanSTARRS DR1 catalog filepath}
            
            returns 
            -------
            Converted PanSTARRS catalog in SDSS magnitude  
            
            notes 
            -----
            Conversion equation : https://iopscience.iop.org/article/10.1088/0004-637X/750/2/99/pdf (Torny(2012))
            -----
            '''
                        
            ra = PANSTARR_catalog['ra']
            dec = PANSTARR_catalog['dec']
            g = PANSTARR_catalog['g_mag']
            r = PANSTARR_catalog['r_mag']
            i = PANSTARR_catalog['i_mag']
            z = PANSTARR_catalog['z_mag']

            gk = PANSTARR_catalog['g_Kmag']
            rk = PANSTARR_catalog['r_Kmag']
            ik = PANSTARR_catalog['i_Kmag']
            zk = PANSTARR_catalog['z_Kmag']
            
            e_g = PANSTARR_catalog['e_g_mag']
            e_r = PANSTARR_catalog['e_r_mag']
            e_i = PANSTARR_catalog['e_i_mag']
            e_z = PANSTARR_catalog['e_z_mag']
            
            e_gk = PANSTARR_catalog['e_g_Kmag']
            e_rk = PANSTARR_catalog['e_r_Kmag']
            e_ik = PANSTARR_catalog['e_i_Kmag']
            e_zk = PANSTARR_catalog['e_z_Kmag']
            
            gr = g-r
            grk = gk-rk
            ri = r-i
            rik = rk-ik
            
            g_c = g + 0.014 + 0.162*gr
            r_c = r - 0.001 + 0.011*gr
            i_c = i - 0.004 + 0.020*gr
            z_c = z + 0.013 - 0.050*gr
            
            gk_c = gk + 0.014 + 0.162*grk
            rk_c = rk - 0.001 + 0.011*grk
            ik_c = ik - 0.004 + 0.020*grk
            zk_c = zk + 0.013 - 0.050*grk

            e_gr = np.sqrt((e_g)**2+(e_r)**2)
            e_grk = np.sqrt((e_gk)**2+(e_rk)**2)
            e_ri = np.sqrt((e_r)**2+(e_i)**2)
            e_rik = np.sqrt((e_rk)**2+(e_ik)**2)
            
            e_g_c = np.sqrt((e_g)**2 + 0.009**2 + (0.162*e_gr)**2)
            e_r_c = np.sqrt((e_r)**2 + 0.004**2 + (0.011*e_gr)**2)
            e_i_c = np.sqrt((e_i)**2 + 0.005**2 + (0.020*e_gr)**2)
            e_z_c = np.sqrt((e_z)**2 + 0.010**2 + (0.050*e_gr)**2)
            
            e_gk_c = np.sqrt((e_gk)**2 + 0.009**2 + (0.162*e_grk)**2)
            e_rk_c = np.sqrt((e_rk)**2 + 0.004**2 + (0.011*e_grk)**2)
            e_ik_c = np.sqrt((e_ik)**2 + 0.005**2 + (0.020*e_grk)**2)
            e_zk_c = np.sqrt((e_zk)**2 + 0.010**2 + (0.050*e_grk)**2)
            
            source = {'ra':ra,
                    'dec':dec,
                    'g_mag':g_c,
                    'e_g_mag':e_g_c,
                    'r_mag':r_c,
                    'e_r_mag':e_r_c,
                    'i_mag':i_c,
                    'e_i_mag':e_i_c,
                    'z_mag':z_c,
                    'e_z_mag':e_z_c,
                    'g_Kmag':gk_c,
                    'e_g_Kmag':e_gk_c,
                    'r_Kmag':rk_c,
                    'e_r_Kmag':e_rk_c,
                    'i_Kmag':ik_c,
                    'e_i_Kmag':e_ik_c,
                    'z_Kmag':zk_c,
                    'e_z_Kmag':e_zk_c}
            ptable = pd.DataFrame(source)
            ctable = Table.from_pandas(ptable)
            result = self.match_digit_tbl(ctable)
            return result

        def PANSTARRS1_to_APASS(PANSTARR_catalog):
            '''
            parameters
            ----------
            {PanSTARRS DR1 catalog filepath}
            
            returns 
            -------
            Converted PanSTARRS catalog in APASS magnitude  
            
            notes 
            -----
            Conversion equation : https://arxiv.org/pdf/1809.09157.pdf (Torny(2018))
            -----
            '''
                        
            ra = PANSTARR_catalog['ra']
            dec = PANSTARR_catalog['dec']
            g = PANSTARR_catalog['g_mag']
            r = PANSTARR_catalog['r_mag']
            i = PANSTARR_catalog['i_mag']

            gk = PANSTARR_catalog['g_Kmag']
            rk = PANSTARR_catalog['r_Kmag']
            ik = PANSTARR_catalog['i_Kmag']
            
            e_g = PANSTARR_catalog['e_g_mag']
            e_r = PANSTARR_catalog['e_r_mag']
            e_i = PANSTARR_catalog['e_i_mag']
            
            e_gk = PANSTARR_catalog['e_g_Kmag']
            e_rk = PANSTARR_catalog['e_r_Kmag']
            e_ik = PANSTARR_catalog['e_i_Kmag']
            
            gr = g-r
            grk = gk-rk
            
            g_c = g + 0.023 + 0.054*gr
            r_c = r - 0.058 + 0.023*gr
            i_c = i + 0.003 + 0.057*gr

            gk_c = gk + 0.023 + 0.054*grk
            rk_c = rk - 0.058 + 0.023*grk
            ik_c = ik + 0.003 + 0.057*grk
            
            e_gr = np.sqrt((e_g)**2+(e_r)**2)
            e_grk = np.sqrt((e_gk)**2+(e_rk)**2)
            
            e_g_c = np.sqrt((e_g)**2 + 0.032**2 + (0.054*e_gr)**2)
            e_r_c = np.sqrt((e_r)**2 + 0.039**2 + (0.023*e_gr)**2)
            e_i_c = np.sqrt((e_i)**2 + 0.050**2 + (0.057*e_gr)**2)
            
            e_gk_c = np.sqrt((e_gk)**2 + 0.032**2 + (0.054*e_grk)**2)
            e_rk_c = np.sqrt((e_rk)**2 + 0.039**2 + (0.023*e_grk)**2)
            e_ik_c = np.sqrt((e_ik)**2 + 0.050**2 + (0.057*e_grk)**2)
            
            source = {'ra':ra,
                    'dec':dec,
                    'g_mag':g_c,
                    'e_g_mag':e_g_c,
                    'r_mag':r_c,
                    'e_r_mag':e_r_c,
                    'i_mag':i_c,
                    'e_i_mag':e_i_c,
                    'g_Kmag':gk_c,
                    'e_g_Kmag':e_gk_c,
                    'r_Kmag':rk_c,
                    'e_r_Kmag':e_rk_c,
                    'i_Kmag':ik_c,
                    'e_i_Kmag':e_ik_c}
            ptable = pd.DataFrame(source)
            ctable = Table.from_pandas(ptable)
            result = self.match_digit_tbl(ctable)
            return result

        def PANSTARRS1_to_SMSS(PANSTARR_catalog):
            '''
            parameters
            ----------
            {PanSTARRS DR1 catalog filepath}
            
            returns 
            -------
            Converted PanSTARRS catalog in Skymapper DR1 magnitude  
            
            notes 
            -----
            Conversion equation : https://arxiv.org/pdf/1809.09157.pdf (Torny(2018))
            -----
            '''
                        
            ra = PANSTARR_catalog['ra']
            dec = PANSTARR_catalog['dec']
            g = PANSTARR_catalog['g_mag']
            r = PANSTARR_catalog['r_mag']
            i = PANSTARR_catalog['i_mag']
            z = PANSTARR_catalog['z_mag']

            gk = PANSTARR_catalog['g_Kmag']
            rk = PANSTARR_catalog['r_Kmag']
            ik = PANSTARR_catalog['i_Kmag']
            zk = PANSTARR_catalog['z_Kmag']
            
            e_g = PANSTARR_catalog['e_g_mag']
            e_r = PANSTARR_catalog['e_r_mag']
            e_i = PANSTARR_catalog['e_i_mag']
            e_z = PANSTARR_catalog['e_z_mag']
            
            e_gk = PANSTARR_catalog['e_g_Kmag']
            e_rk = PANSTARR_catalog['e_r_Kmag']
            e_ik = PANSTARR_catalog['e_i_Kmag']
            e_zk = PANSTARR_catalog['e_z_Kmag']
            
            gr = g-r
            grk = gk-rk
            ri = r-i
            rik = rk-ik
            
            g_c = g + 0.010 - 0.228*gr
            r_c = r + 0.004 + 0.039*gr
            i_c = i + 0.008 - 0.110*ri
            z_c = z - 0.004 - 0.097*ri

            gk_c = gk + 0.010 - 0.228*grk
            rk_c = rk + 0.004 + 0.039*grk
            ik_c = ik + 0.008 - 0.110*rik
            zk_c = zk - 0.004 - 0.097*rik
            
            e_gr = np.sqrt((e_g)**2+(e_r)**2)
            e_grk = np.sqrt((e_gk)**2+(e_rk)**2)
            e_ri = np.sqrt((e_r)**2+(e_i)**2)
            e_rik = np.sqrt((e_rk)**2+(e_ik)**2)
            
            e_g_c = np.sqrt((e_g)**2 + 0.032**2 + (0.228*e_gr)**2)
            e_r_c = np.sqrt((e_r)**2 + 0.016**2 + (0.039*e_gr)**2)
            e_i_c = np.sqrt((e_i)**2 + 0.022**2 + (0.110*e_gr)**2)
            e_z_c = np.sqrt((e_z)**2 + 0.020**2 + (0.097*e_gr)**2)
            
            e_gk_c = np.sqrt((e_gk)**2 + 0.032**2 + (0.228*e_grk)**2)
            e_rk_c = np.sqrt((e_rk)**2 + 0.016**2 + (0.039*e_grk)**2)
            e_ik_c = np.sqrt((e_ik)**2 + 0.022**2 + (0.110*e_grk)**2)
            e_zk_c = np.sqrt((e_zk)**2 + 0.020**2 + (0.097*e_grk)**2)
            
            source = {'ra':ra,
                    'dec':dec,
                    'g_mag':g_c,
                    'e_g_mag':e_g_c,
                    'r_mag':r_c,
                    'e_r_mag':e_r_c,
                    'i_mag':i_c,
                    'e_i_mag':e_i_c,
                    'z_mag':z_c,
                    'e_z_mag':e_z_c,
                    'g_Kmag':gk_c,
                    'e_g_Kmag':e_gk_c,
                    'r_Kmag':rk_c,
                    'e_r_Kmag':e_rk_c,
                    'i_Kmag':ik_c,
                    'e_i_Kmag':e_ik_c,
                    'z_Kmag':zk_c,
                    'e_z_Kmag':e_zk_c}
            ptable = pd.DataFrame(source)
            ctable = Table.from_pandas(ptable)
            result = self.match_digit_tbl(ctable)
            return result

        def PANSTARRS1_to_JH(PANSTARR_catalog):
            '''
            parameters
            ----------
            {PanSTARRS DR1 catalog filepath}
            
            returns 
            -------
            Converted PanSTARRS catalog in APASS magnitude  
            
            notes 
            -----
            Conversion equation : https://iopscience.iop.org/article/10.1088/0004-637X/750/2/99/pdf (Torny(2012))
            -----
            '''
                        
            ra = PANSTARR_catalog['ra']
            dec = PANSTARR_catalog['dec']
            g = PANSTARR_catalog['g_mag']
            r = PANSTARR_catalog['r_mag']
            i = PANSTARR_catalog['i_mag']

            gk = PANSTARR_catalog['g_Kmag']
            rk = PANSTARR_catalog['r_Kmag']
            ik = PANSTARR_catalog['i_Kmag']
            
            e_g = PANSTARR_catalog['e_g_mag']
            e_r = PANSTARR_catalog['e_r_mag']
            e_i = PANSTARR_catalog['e_i_mag']
            
            e_gk = PANSTARR_catalog['e_g_Kmag']
            e_rk = PANSTARR_catalog['e_r_Kmag']
            e_ik = PANSTARR_catalog['e_i_Kmag']

            gr = g-r
            grk = gk-rk

            B = g + 0.213 + 0.587*gr
            V = r + 0.006 + 0.474*gr
            R = r - 0.138 - 0.131*gr
            I = i - 0.367 - 0.149*gr
            
            Bk = gk + 0.213 + 0.587*grk
            Vk = rk + 0.006 + 0.474*grk
            Rk = rk - 0.138 - 0.131*grk
            Ik = ik - 0.367 - 0.149*grk   

            e_gr = np.sqrt((e_g)**2+(e_r)**2)
            e_grk = np.sqrt((e_g)**2)
            
            e_B_c = np.sqrt((e_g)**2 + 0.034**2 + (0.587*e_gr)**2)
            e_V_c = np.sqrt((e_r)**2 + 0.012**2 + (0.006*e_gr)**2)
            e_R_c = np.sqrt((e_r)**2 + 0.015**2 + (0.131*e_gr)**2)
            e_I_c = np.sqrt((e_i)**2 + 0.016**2 + (0.149*e_gr)**2)
            
            e_Bk_c = np.sqrt((e_gk)**2 + 0.034**2 + (0.587*e_grk)**2)
            e_Vk_c = np.sqrt((e_rk)**2 + 0.012**2 + (0.006*e_grk)**2)
            e_Rk_c = np.sqrt((e_rk)**2 + 0.015**2 + (0.131*e_grk)**2)
            e_Ik_c = np.sqrt((e_ik)**2 + 0.016**2 + (0.149*e_grk)**2)

            source = {'ra':ra,
                    'dec':dec,
                    'B_mag':B,
                    'e_B_mag':e_B_c,
                    'V_mag':V,
                    'e_V_mag':e_V_c,
                    'R_mag':R,
                    'e_R_mag':e_R_c,
                    'I_mag':I,
                    'e_I_mag':e_I_c,
                    'B_Kmag':Bk,
                    'e_B_Kmag':e_Bk_c,
                    'V_Kmag':Vk,
                    'e_V_Kmag':e_Vk_c,
                    'R_Kmag':Rk,
                    'e_R_Kmag':e_Rk_c,
                    'I_Kmag':Ik,
                    'e_I_Kmag':e_Ik_c}
            
            ptable = pd.DataFrame(source)
            ctable = Table.from_pandas(ptable)
            result = self.match_digit_tbl(ctable)
            return result
            
        class PS1:
            def __repr__(self):
                txt = '[Attributes]\n'+''.join(f"PS1.{key}\n" for key in self.__dict__.keys())
                return txt
            
        data = self.get_catalog(catalog_name='PS1')
        if not data:
            is_exist = False
        else:
            is_exist = True

        cat = PS1()
        cat.is_exist = is_exist
        if cat.is_exist:
            cat.data = PS1_format(data)
            cat.conversion = dict()
            cat.conversion['SDSS'] = PANSTARRS1_to_SDSS(cat.data)
            cat.conversion['APASS'] = PANSTARRS1_to_APASS(cat.data)
            cat.conversion['SMSS'] = PANSTARRS1_to_SMSS(cat.data)
            cat.conversion['JH'] = PANSTARRS1_to_JH(cat.data)
        return cat

    def get_SMSS(self):
        def SMSS_format(SMSS_catalog) -> Table:
            original = ('ObjectId','RAICRS','DEICRS','Niflags','flags','Ngood','Ngoodu','Ngoodv','Ngoodg','Ngoodr','Ngoodi','Ngoodz','ClassStar','uPSF','e_uPSF','vPSF','e_vPSF','gPSF','e_gPSF','rPSF','e_rPSF','iPSF','e_iPSF','zPSF','e_zPSF')
            format_ = ('ID','ra','dec','nimflag','flag','ngood','ngoodu','ngoodv','ngoodg','ngoodr','ngoodi','ngoodz','class_star','u_mag','e_u_mag','v_mag','e_v_mag','g_mag','e_g_mag','r_mag','e_r_mag','i_mag','e_i_mag','z_mag','e_z_mag')
            SMSS_catalog.rename_columns(original, format_)
            formatted_catalog = self.match_digit_tbl(SMSS_catalog)
            return formatted_catalog

        def SMSS_to_PanSTARRS1(SMSS_catalog):
            '''
            parameters
            ----------
            {SMSS catalog filepath}
            
            returns 
            -------
            Converted SMSS catalog in PS1 magnitude  
            
            notes 
            -----
            Conversion equation : https://arxiv.org/pdf/1809.09157.pdf (Torny(2018))
            -----
            '''
                        
            ra = SMSS_catalog['ra']
            dec = SMSS_catalog['dec']
            g = SMSS_catalog['g_mag']
            r = SMSS_catalog['r_mag']
            i = SMSS_catalog['i_mag']
            z = SMSS_catalog['z_mag']
            flag = SMSS_catalog['flag']
            ngood = SMSS_catalog['ngood']
            class_star = SMSS_catalog['class_star']
            
            e_g = SMSS_catalog['e_g_mag']
            e_r = SMSS_catalog['e_r_mag']
            e_i = SMSS_catalog['e_i_mag']
            e_z = SMSS_catalog['e_z_mag']

            gr = g-r
            ri = r-i
            
            g_c = g + 0.004 + 0.272*gr
            r_c = r - 0.016 - 0.035*gr
            i_c = i - 0.011 + 0.100*ri
            z_c = z + 0.009 + 0.082*ri

            e_gr = np.sqrt((e_g)**2+(e_r)**2)
            e_ri = np.sqrt((e_r)**2+(e_i)**2)
            
            e_g_c = np.sqrt((e_g)**2 + 0.029**2 + (0.272*e_gr)**2)
            e_r_c = np.sqrt((e_r)**2 + 0.021**2 + (0.035*e_gr)**2)
            e_i_c = np.sqrt((e_i)**2 + 0.016**2 + (0.100*e_ri)**2)
            e_z_c = np.sqrt((e_i)**2 + 0.020**2 + (0.082*e_ri)**2)
            
            source = {'ra':ra,
                    'dec':dec,
                    'g_mag':g_c,
                    'e_g_mag':e_g_c,
                    'r_mag':r_c,
                    'e_r_mag':e_r_c,
                    'i_mag':i_c,
                    'e_i_mag':e_i_c,
                    'z_mag':z_c,
                    'e_z_mag':e_z_c,
                    'flag':flag,
                    'ngood':ngood,
                    'class_star':class_star
                    }
            
            atable = pd.DataFrame(source)
            ctable = Table.from_pandas(atable)
            result = self.match_digit_tbl(ctable)
            return result

        def SMSS_to_SDSS(SMSS_catalog):
            '''
            parameters
            ----------
            {SMSS catalog filepath}
            
            returns 
            -------
            Converted SMSS catalog in SDSS magnitude  
            
            notes 
            -----
            This conversion is performed by two conversion equation (SMSS > PS1 > SDSS)
            Conversion equation(SMSS1>PS1) : https://arxiv.org/pdf/1809.09157.pdf (Torny(2018))
            Conversion equation(PS1>SDSS) : https://iopscience.iop.org/article/10.1088/0004-637X/750/2/99/pdf (Torny(2012))
            -----
            '''
            
            pcatalog = SMSS_to_PanSTARRS1(SMSS_catalog)
            
            flag = pcatalog['flag']
            ngood = pcatalog['ngood']
            class_star = pcatalog['class_star']
            
            ra = pcatalog['ra']
            dec = pcatalog['dec']
            g = pcatalog['g_mag']
            r = pcatalog['r_mag']
            i = pcatalog['i_mag']
            z = pcatalog['z_mag']


            e_g = pcatalog['e_g_mag']
            e_r = pcatalog['e_r_mag']
            e_i = pcatalog['e_i_mag']
            e_z = pcatalog['e_z_mag']
            
            gr = g-r
            ri = r-i
            
            g_c = g + 0.014 + 0.162*gr
            r_c = r - 0.001 + 0.011*gr
            i_c = i - 0.004 + 0.020*gr
            z_c = z + 0.013 - 0.050*gr

            e_gr = np.sqrt((e_g)**2+(e_r)**2)
            e_ri = np.sqrt((e_r)**2+(e_i)**2)
            
            e_g_c = np.sqrt((e_g)**2 + 0.009**2 + (0.162*e_gr)**2)
            e_r_c = np.sqrt((e_r)**2 + 0.004**2 + (0.011*e_gr)**2)
            e_i_c = np.sqrt((e_i)**2 + 0.005**2 + (0.020*e_gr)**2)
            e_z_c = np.sqrt((e_z)**2 + 0.010**2 + (0.050*e_gr)**2)
            
            source = {'ra':ra,
                    'dec':dec,
                    'g_mag':g_c,
                    'e_g_mag':e_g_c,
                    'r_mag':r_c,
                    'e_r_mag':e_r_c,
                    'i_mag':i_c,
                    'e_i_mag':e_i_c,
                    'z_mag':z_c,
                    'e_z_mag':e_z_c,
                    'flag':flag,
                    'ngood':ngood,
                    'class_star':class_star
                    }
            
            atable = pd.DataFrame(source)
            ctable = Table.from_pandas(atable)
            result = self.match_digit_tbl(ctable)
            return result

        def SMSS_to_JH(SMSS_catalog):
            '''
            parameters
            ----------
            {SMSS catalog filepath}
            
            returns 
            -------
            Converted SMSS catalog in Johnson-Cousins magnitude  
            
            notes 
            -----
            This conversion is performed by two conversion equation (SMSS > PS1 > JH)
            Conversion equation(SMSS1>PS1) : https://arxiv.org/pdf/1809.09157.pdf (Torny(2018))
            Conversion equation(PS1>JH) : https://iopscience.iop.org/article/10.1088/0004-637X/750/2/99/pdf (Torny(2012))
            -----
            '''
            
            pcatalog = SMSS_to_PanSTARRS1(SMSS_catalog)
            
            flag = pcatalog['flag']
            ngood = pcatalog['ngood']
            class_star = pcatalog['class_star']
            
            ra = pcatalog['ra']
            dec = pcatalog['dec']
            g = pcatalog['g_mag']
            r = pcatalog['r_mag']
            i = pcatalog['i_mag']
            
            e_g = pcatalog['e_g_mag']
            e_r = pcatalog['e_r_mag']
            e_i = pcatalog['e_i_mag']

            gr = g-r

            B = g + 0.213 + 0.587*gr
            V = r + 0.006 + 0.474*gr
            R = r - 0.138 - 0.131*gr
            I = i - 0.367 - 0.149*gr

            e_gr = np.sqrt((e_g)**2+(e_r)**2)
            
            e_B_c = np.sqrt((e_g)**2 + 0.034**2 + (0.587*e_gr)**2)
            e_V_c = np.sqrt((e_r)**2 + 0.012**2 + (0.006*e_gr)**2)
            e_R_c = np.sqrt((e_r)**2 + 0.015**2 + (0.131*e_gr)**2)
            e_I_c = np.sqrt((e_i)**2 + 0.016**2 + (0.149*e_gr)**2)

            source = {'ra':ra,
                    'dec':dec,
                    'B_mag':B,
                    'e_B_mag':e_B_c,
                    'V_mag':V,
                    'e_V_mag':e_V_c,
                    'R_mag':R,
                    'e_R_mag':e_R_c,
                    'I_mag':I,
                    'e_I_mag':e_I_c,
                    'flag':flag,
                    'ngood':ngood,
                    'class_star':class_star
                    }
            
            ptable = pd.DataFrame(source)
            ctable = Table.from_pandas(ptable)
            result = self.match_digit_tbl(ctable)
            return result

        class SMSS:
            def __repr__(self):
                txt = '[Attributes]\n'+''.join(f"SMSS.{key}\n" for key in self.__dict__.keys())
                return txt
            
        data = self.get_catalog(catalog_name='SMSS')
        if not data:
            is_exist = False
        else:
            is_exist = True

        cat = SMSS()
        cat.is_exist = is_exist
        if cat.is_exist:
            cat.data = SMSS_format(data)
            cat.conversion = dict()
            cat.conversion['PS1'] = SMSS_to_PanSTARRS1(cat.data)
            cat.conversion['SDSS'] = SMSS_to_SDSS(cat.data)
            cat.conversion['JH'] = SMSS_to_JH(cat.data)
        return cat

    def get_SDSS(self):
        def SDSS_format(SDSS_catalog) -> Table:
            original = ('RA_ICRS','DE_ICRS','umag','e_umag','gmag','e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag')
            format_ = ('ra','dec','umag','e_umag','gmag','e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag')
            SDSS_catalog.rename_columns(original, format_)
            formatted_catalog = self.match_digit_tbl(SDSS_catalog)
            return formatted_catalog

        class SDSS:
            def __repr__(self):
                txt = '[Attributes]\n'+''.join(f"SDSS.{key}\n" for key in self.__dict__.keys())
                return txt
            
        data = self.get_catalog(catalog_name='SDSS')
        if not data:
            is_exist = False
        else:
            is_exist = True

        cat = SDSS()
        cat.is_exist = is_exist
        if cat.is_exist:
            cat.data = SDSS_format(data)
        return cat
    
    def match_digit_tbl(self, tbl):
        for column in tbl.columns:
            if tbl[column].dtype == 'float64':
                tbl[column].format = '{:.5f}'
        return tbl

    def get_catalog(self, catalog_name: str = 'APASS'):
        catalog_file = os.path.join(self.catpath, catalog_name, f"{self.fieldinfo['target']}.csv")
        is_exist = os.path.exists(catalog_file)
        
        if is_exist:
            data = ascii.read(catalog_file)
        else:
            data = self.query_catalog(catalog_name=catalog_name, save=True)
        return data

    def get_target_coord(self, target_name) -> SkyCoord:
        from astroquery.simbad import Simbad

        # Create a custom Simbad instance with the necessary fields
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('ra', 'dec')

        # Query an object (e.g., "Vega")
        result_table = custom_simbad.query_object(target_name)

        # Extract coordinates
        if result_table is not None:
            ra = result_table['RA'][0]  # Right Ascension
            dec = result_table['DEC'][0]  # Declination
            coord = SkyCoord(ra, dec, unit = (u.hourangle, u.deg))
            return coord
        else:
            print(f"{target_name} not found in")
            #raise ValueError("Object not found.")

    def query_catalog(self, catalog_name: str = 'APASS', save: bool = True):

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
        def ps1_query(ra_deg, dec_deg, rad_deg, maxmag = 20, minmag = 10, maxsources=1000000):
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
        
        ra_deg = self.fieldinfo['ra']
        dec_deg = self.fieldinfo['dec']
        if catalog_name == 'APASS':
            data = apass_query(ra_deg = ra_deg, dec_deg = dec_deg, rad_deg = self.fieldinfo['radius'], maxmag = self.fieldinfo['maxmag'], minmag = self.fieldinfo['minmag'], maxsources = self.fieldinfo['maxsources'])
        elif catalog_name == 'PS1':
            data = ps1_query(ra_deg = ra_deg, dec_deg = dec_deg, rad_deg = self.fieldinfo['radius'], maxmag = self.fieldinfo['maxmag'], minmag = self.fieldinfo['minmag'], maxsources = self.fieldinfo['maxsources'])
        elif catalog_name == 'SDSS':
            data = sdss_query(ra_deg = ra_deg, dec_deg = dec_deg, rad_deg = self.fieldinfo['radius'], maxmag = self.fieldinfo['maxmag'], minmag = self.fieldinfo['minmag'], maxsources = self.fieldinfo['maxsources'])
        elif catalog_name == 'SMSS':
            data = smss_query(ra_deg = ra_deg, dec_deg = dec_deg, rad_deg = self.fieldinfo['radius'], maxmag = self.fieldinfo['maxmag'], minmag = self.fieldinfo['minmag'], maxsources = self.fieldinfo['maxsources'])
        elif catalog_name == 'GAIA':
            data = gaia_query(ra_deg = ra_deg, dec_deg = dec_deg, rad_deg = self.fieldinfo['radius'], maxmag = self.fieldinfo['maxmag'], minmag = self.fieldinfo['minmag'], maxsources = self.fieldinfo['maxsources'])
        else:
            raise ValueError(f'{catalog_name} is not registered')                       
        
        if not data:
            pass
            #print(f"{catalog_name} cannot query {self.fieldinfo['target']}.")
        else:
            if save:
                os.makedirs(os.path.join(self.catpath, catalog_name), exist_ok=True)
                path = os.path.join(self.catpath, catalog_name, self.fieldinfo['target'])+'.csv'
                ascii.write(data, path, format = 'csv', overwrite = True)
                print(f"{catalog_name}/{self.fieldinfo['target']} saved: {path}")
            return data


