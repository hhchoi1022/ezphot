import numpy as np
import pandas as pd

from astropy.io import ascii
from astropy.table import Table
#%%
def PANSTARRS1_format(panstar_catalog):
    tbl = ascii.read(panstar_catalog)
    original = ('objID','RAJ2000','DEJ2000','e_RAJ2000','e_DEJ2000','gmag','e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag','ymag','e_ymag','gKmag','e_gKmag','rKmag','e_rKmag','iKmag','e_iKmag','zKmag','e_zKmag','yKmag','e_yKmag')
    format_ = ('ID','ra','dec','e_ra','e_dec','g_mag','e_g_mag','r_mag','e_r_mag','i_mag','e_i_mag','z_mag','e_z_mag','y_mag','e_y_mag','g_Kmag','e_g_Kmag','r_Kmag','e_r_Kmag','i_Kmag','e_i_Kmag','z_Kmag','e_z_Kmag','y_Kmag','e_y_Kmag')
    tbl.rename_columns(original, format_)
    return tbl
    
#%%
def SMSS3_format(skymapper_catalog):
    tbl = ascii.read(skymapper_catalog)
    original = ('object_id','raj2000','dej2000','glon','glat','flags','nimaflags','ngood','nch_max','u_ngood','v_ngood','g_ngood','r_ngood','i_ngood','z_ngood','class_star','u_psf','e_u_psf','v_psf','e_v_psf','g_psf','e_g_psf','r_psf','e_r_psf','i_psf','e_i_psf','z_psf','e_z_psf','ps1_dr1_id','ps1_dr1_dist','gaia_edr3_id1','gaia_edr3_dist1')
    format_ = ('ID','ra','dec','lon','lat','flag','nimflag','ngood','nch_max','u_ngood','v_ngood','g_ngood','r_ngood','i_ngood','z_ngood','class_star','u_mag','e_u_mag','v_mag','e_v_mag','g_mag','e_g_mag','r_mag','e_r_mag','i_mag','e_i_mag','z_mag','e_z_mag','ps1_dr1_id','ps1_dr1_dist','gaia_edr3_id1','gaia_edr3_dist1')
    tbl.rename_columns(original, format_)
    return tbl


# %%
def APASS_format(APASS_catalog):
    tbl = ascii.read(APASS_catalog)
    original = ('RAJ2000','DEJ2000','e_RAJ2000','e_DEJ2000','Bmag','e_Bmag','Vmag','e_Vmag','g_mag','e_g_mag','r_mag','e_r_mag','i_mag','e_i_mag')
    format_ = ('ra','dec','e_ra','e_dec','B_mag','e_B_mag','V_mag','e_V_mag','g_mag','e_g_mag','r_mag','e_r_mag','i_mag','e_i_mag')
    tbl.rename_columns(original, format_)
    for column in tbl.columns:
        if tbl[column].dtype == 'float64':
            tbl[column].format = '{:.5f}'
    return tbl

# %%
def format_digit_tbl(astropy_tbl):
    for column in astropy_tbl.columns:
        if astropy_tbl[column].dtype == 'float64':
            astropy_tbl[column].format = '{:.5f}'
    return astropy_tbl

#%% ################################### PS1 conversion ###################################
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
    
    pcatalog = PANSTARRS1_format(PANSTARR_catalog)
    
    ra = pcatalog['ra']
    dec = pcatalog['dec']
    g = pcatalog['g_mag']
    r = pcatalog['r_mag']
    i = pcatalog['i_mag']
    z = pcatalog['z_mag']

    gk = pcatalog['g_Kmag']
    rk = pcatalog['r_Kmag']
    ik = pcatalog['i_Kmag']
    zk = pcatalog['z_Kmag']
    
    e_g = pcatalog['e_g_mag']
    e_r = pcatalog['e_r_mag']
    e_i = pcatalog['e_i_mag']
    e_z = pcatalog['e_z_mag']
    
    e_gk = pcatalog['e_g_Kmag']
    e_rk = pcatalog['e_r_Kmag']
    e_ik = pcatalog['e_i_Kmag']
    e_zk = pcatalog['e_z_Kmag']
    
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
    result = format_digit_tbl(ctable)
    return result

#%%
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
    
    pcatalog = PANSTARRS1_format(PANSTARR_catalog)
    
    ra = pcatalog['ra']
    dec = pcatalog['dec']
    g = pcatalog['g_mag']
    r = pcatalog['r_mag']
    i = pcatalog['i_mag']

    gk = pcatalog['g_Kmag']
    rk = pcatalog['r_Kmag']
    ik = pcatalog['i_Kmag']
    
    e_g = pcatalog['e_g_mag']
    e_r = pcatalog['e_r_mag']
    e_i = pcatalog['e_i_mag']
    
    e_gk = pcatalog['e_g_Kmag']
    e_rk = pcatalog['e_r_Kmag']
    e_ik = pcatalog['e_i_Kmag']
    
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
    result = format_digit_tbl(ctable)
    return result

# %%
def PANSTARRS1_to_SMSS1(PANSTARR_catalog):
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
    
    pcatalog = PANSTARRS1_format(PANSTARR_catalog)
    
    ra = pcatalog['ra']
    dec = pcatalog['dec']
    g = pcatalog['g_mag']
    r = pcatalog['r_mag']
    i = pcatalog['i_mag']
    z = pcatalog['z_mag']

    gk = pcatalog['g_Kmag']
    rk = pcatalog['r_Kmag']
    ik = pcatalog['i_Kmag']
    zk = pcatalog['z_Kmag']
    
    e_g = pcatalog['e_g_mag']
    e_r = pcatalog['e_r_mag']
    e_i = pcatalog['e_i_mag']
    e_z = pcatalog['e_z_mag']
    
    e_gk = pcatalog['e_g_Kmag']
    e_rk = pcatalog['e_r_Kmag']
    e_ik = pcatalog['e_i_Kmag']
    e_zk = pcatalog['e_z_Kmag']
    
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
    result = format_digit_tbl(ctable)
    return result

# %%
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
    
    pcatalog = PANSTARRS1_format(PANSTARR_catalog)
    
    ra = pcatalog['ra']
    dec = pcatalog['dec']
    g = pcatalog['g_mag']
    r = pcatalog['r_mag']
    i = pcatalog['i_mag']

    gk = pcatalog['g_Kmag']
    rk = pcatalog['r_Kmag']
    ik = pcatalog['i_Kmag']
    
    e_g = pcatalog['e_g_mag']
    e_r = pcatalog['e_r_mag']
    e_i = pcatalog['e_i_mag']
    
    e_gk = pcatalog['e_g_Kmag']
    e_rk = pcatalog['e_r_Kmag']
    e_ik = pcatalog['e_i_Kmag']

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
    result = format_digit_tbl(ctable)
    return result
    
# %% ################################### APASS conversion ###################################

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
    
    acatalog = APASS_format(APASS_catalog)
    
    ra = acatalog['ra']
    dec = acatalog['dec']
    g = acatalog['g_mag']
    r = acatalog['r_mag']
    i = acatalog['i_mag']
    
    e_g = acatalog['e_g_mag']
    e_r = acatalog['e_r_mag']
    e_i = acatalog['e_i_mag']

    gr = g-r
    
    g_c = g - 0.009 - 0.061*gr
    r_c = r + 0.065 - 0.026*gr
    i_c = i - 0.015 - 0.068*gr

    
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
    result = format_digit_tbl(ctable)
    return result

# %%

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
    from astropy.table import Table
    
    acatalog = APASS_format(APASS_catalog)

    ra = acatalog['ra']
    dec = acatalog['dec']
    B = acatalog['B_mag']
    V = acatalog['V_mag']
    g = acatalog['g_mag']
    r = acatalog['r_mag']
    i = acatalog['i_mag']

    e_B = acatalog['e_B_mag']
    e_V = acatalog['e_V_mag']
    e_g = acatalog['e_g_mag']
    e_r = acatalog['e_r_mag']
    e_i = acatalog['e_i_mag']

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
    result = format_digit_tbl(ctable)
    return result

# %% ################################### Skymapper conversion ###################################

def SMSS1_to_PanSTARRS1(SMSS_catalog):
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
    
    scatalog = SMSS3_format(SMSS_catalog)
    
    ra = scatalog['ra']
    dec = scatalog['dec']
    g = scatalog['g_mag']
    r = scatalog['r_mag']
    i = scatalog['i_mag']
    z = scatalog['z_mag']
    flag = scatalog['flag']
    ngood = scatalog['ngood']
    class_star = scatalog['class_star']
    
    e_g = scatalog['e_g_mag']
    e_r = scatalog['e_r_mag']
    e_i = scatalog['e_i_mag']
    e_z = scatalog['e_z_mag']

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
    result = format_digit_tbl(ctable)
    return result

#%%
def SMSS1_to_SDSS(SMSS_catalog):
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
    
    pcatalog = SMSS1_to_PanSTARRS1(SMSS_catalog)
    
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
    result = format_digit_tbl(ctable)
    return result

# %%
def SMSS1_to_JH(SMSS_catalog):
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
    
    pcatalog = SMSS1_to_PanSTARRS1(SMSS_catalog)
    
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
    result = format_digit_tbl(ctable)
    return result

