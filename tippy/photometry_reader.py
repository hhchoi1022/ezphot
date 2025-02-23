

#%%
import re
import pandas as pd
from astropy.io import ascii
from datetime import datetime
from astropy.time import Time
import glob
from astropy.table import Table
from astropy.table import column

def read_photometry_file(file):
    try:
        tbl = ascii.read(file)
        df = tbl.to_pandas()
        return df
    
    except Exception as e:
        print(f"Error reading {file}: {e}")
        obsdate = extract_datetime(file)
        colnames = ["NUMBER", "FLUX_APER", "FLUX_APER_1", "FLUX_APER_2", "FLUX_APER_3", "FLUX_APER_4", "FLUXERR_APER", "FLUXERR_APER_1", "FLUXERR_APER_2", "FLUXERR_APER_3", "FLUXERR_APER_4", "MAG_APER", "MAG_APER_1", "MAG_APER_2", "MAG_APER_3", "MAG_APER_4", "MAGERR_APER", "MAGERR_APER_1", "MAGERR_APER_2", "MAGERR_APER_3", "MAGERR_APER_4", "FWHM_IMAGE", "FWHM_WORLD", "X_IMAGE", "Y_IMAGE", "X_WORLD", "Y_WORLD", "ALPHA_J2000", "DELTA_J2000", "FLAGS", "CLASS_STAR", "THRESHOLD", "ZP_APER", "ZPERR_APER", "DEPTH3_APER", "DEPTH5_APER", "APER_SIZE", "DATE-OBS", "JD", "ZP_APER_1", "ZPERR_APER_1", "DEPTH3_APER_1", "DEPTH5_APER_1", "APER_SIZE_1", "ZP_APER_2", "ZPERR_APER_2", "DEPTH3_APER_2", "DEPTH5_APER_2", "APER_SIZE_2"]
        empty_data = {col: [None] for col in colnames}
        empty_data['obsdate'] = [obsdate]
        return pd.DataFrame(empty_data)
    
def write_photometry_file(key : str = '/data1/supernova_rawdata/SN2021aefx/photometry/KCT*/g/sub*.phot',
                          savepath :str = '/data1/supernova_rawdata/SN2021aefx/photometry/KCT_g.dat',
                          mag_key : str = 'MAG_APER_SKY_1',
                          magerr_key : str = 'MAGERR_APER_SKY_1',
                          depth_key : str = 'DEPTH5_APER_1',
                          zp_key  : str = 'ZP_APER_1',
                          filter_ : str = 'g',
                          observatory : str = 'KCT'):
    files = sorted(glob.glob(key))    
    all_dataframes = [read_photometry_file(file) for file in files]
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    tbl = Table().from_pandas(combined_df)
    save_tbl = Table()
    save_tbl['obsdate'] = Time(tbl['JD'], format = 'jd').mjd
    save_tbl['mag'] = tbl[mag_key]
    save_tbl['e_mag'] = tbl[magerr_key]
    save_tbl['magsys'] = 'AB'
    save_tbl['filter'] = filter_
    save_tbl['depth_5sig'] = tbl[depth_key]
    save_tbl['zp'] = tbl[depth_key]
    save_tbl['observatory'] = observatory
    if isinstance(tbl['MAG_APER_SKY_1'], column.MaskedColumn):
        save_tbl['detected'] = (~tbl['MAG_APER_SKY_1'].mask)
    else:
        save_tbl['detected'] = True
    save_tbl.sort('obsdate')
    save_tbl.write(savepath, format = 'ascii.fixed_width', overwrite = True)
    print(f'Saved: {savepath}')
    return savepath
# %%
output_filelist = []
# KCT in g band
output_file = write_photometry_file(
                                    key='/data1/supernova_rawdata/SN2021aefx/photometry/KCT*/g/*.phot',
                                    savepath='/data1/supernova_rawdata/SN2021aefx/photometry/IMSNG_KCT_g.dat',
                                    filter_='g',
                                    observatory='KCT'
                                    )   
output_filelist.append(output_file)
# KCT in r band
output_file = write_photometry_file(
                                    key='/data1/supernova_rawdata/SN2021aefx/photometry/KCT*/r/sub*.phot',
                                    savepath='/data1/supernova_rawdata/SN2021aefx/photometry/IMSNG_KCT_r.dat',
                                    filter_='r',
                                    observatory='KCT'
                                    )
output_filelist.append(output_file)

# KCT in i band
output_file = write_photometry_file(
                                    key='/data1/supernova_rawdata/SN2021aefx/photometry/KCT*/i/sub*.phot',
                                    savepath='/data1/supernova_rawdata/SN2021aefx/photometry/IMSNG_KCT_i.dat',
                                    filter_='i',
                                    observatory='KCT'
                                    )
output_filelist.append(output_file)
# LSGT in g band
output_file = write_photometry_file(
                                    key='/data1/supernova_rawdata/SN2021aefx/photometry/LSGT*/g/*.phot',
                                    savepath='/data1/supernova_rawdata/SN2021aefx/photometry/IMSNG_LSGT_g.dat',
                                    filter_='g',
                                    observatory='LSGT'
                                    )
output_filelist.append(output_file)
# LSGT in r band
output_file = write_photometry_file(
                                    key='/data1/supernova_rawdata/SN2021aefx/photometry/LSGT*/r/*.phot',
                                    savepath='/data1/supernova_rawdata/SN2021aefx/photometry/IMSNG_LSGT_r.dat',
                                    filter_='r',
                                    observatory='LSGT'
                                    )
output_filelist.append(output_file)
# LSGT in i band
output_file = write_photometry_file(
                                    key='/data1/supernova_rawdata/SN2021aefx/photometry/LSGT*/i/*.phot',
                                    savepath='/data1/supernova_rawdata/SN2021aefx/photometry/IMSNG_LSGT_i.dat',
                                    filter_='i',
                                    observatory='LSGT'
                                    )
output_filelist.append(output_file)

# RASA36 in r band HIGH mode

output_file = write_photometry_file(
                                    key='/data1/supernova_rawdata/SN2021aefx/photometry/RASA36*/r/HIGH/com*.phot',
                                    savepath='/data1/supernova_rawdata/SN2021aefx/photometry/IMSNG_RASA36_r_HIGH.dat',
                                    filter_='r',
                                    observatory='RASA36'
                                    )
output_filelist.append(output_file)
# RASA36 in r band MERGE mode
output_file = write_photometry_file(
                                    key='/data1/supernova_rawdata/SN2021aefx/photometry/RASA36*/r/MERGE/sub*.phot',
                                    savepath='/data1/supernova_rawdata/SN2021aefx/photometry/IMSNG_RASA36_r_MERGE.dat',
                                    filter_='r',
                                    observatory='RASA36'
                                    )
output_filelist.append(output_file)

# %%
output_filelist = glob.glob('/data1/supernova_rawdata/SN2021aefx/photometry/IMSNG*.dat')
from astropy.table import vstack
tbl_IMSNG = vstack([Table.read(output_file, format = 'ascii.fixed_width') for output_file in output_filelist])  
#tbl_hosse = Table.read('/data1/supernova_rawdata/SN2021aefx/photometry/Hosseinzadeh2022.dat', format = 'ascii.fixed_width')
#tbl_ashall = Table.read('/data1/supernova_rawdata/SN2021aefx/photometry/Ashall2022.dat', format = 'ascii.fixed_width')
tbl_IMSNG.write('/data1/supernova_rawdata/SN2021aefx/photometry/all_IMSNG.dat', format = 'ascii.fixed_width', overwrite = True)
#%%
from astropy.table import Table, MaskedColumn

# Assuming tbl_hosse is already defined
"""
tbl_hosse_new = Table()
tbl_hosse_new['obsdate'] = tbl_hosse['obsdate']
tbl_hosse_new['mag'] = tbl_hosse['mag']
tbl_hosse_new['e_mag'] = tbl_hosse['e_mag']
tbl_hosse_new['magsys'] = ['Vega' if component in ['U', 'B', 'V', 'U_S', 'B_S', 'V_S'] else 'AB' for component in tbl_hosse['filter']]
tbl_hosse_new['filter'] = tbl_hosse['filter']
tbl_hosse_new['depth_5sig'] = MaskedColumn(data=[None]*len(tbl_hosse), mask=True)
tbl_hosse_new['zp'] = MaskedColumn(data=[None]*len(tbl_hosse), mask=True)
tbl_hosse_new['observatory'] = tbl_hosse['observatory']
tbl_hosse_new['detected'] = True
tbl_hosse_new.sort('obsdate')

tbl_hosse_new.write('/data1/supernova_rawdata/SN2021aefx/photometry/Hosseinzadeh2022_new.dat', format='ascii.fixed_width', overwrite=True)

tbl_ashall_new = Table()
tbl_ashall_new['obsdate'] = tbl_ashall['obsdate']
tbl_ashall_new['mag'] = tbl_ashall['mag']
tbl_ashall_new['e_mag'] = tbl_ashall['e_mag']
tbl_ashall_new['magsys'] = ['Vega' if component in ['U', 'B', 'V', 'U_S', 'B_S', 'V_S'] else 'AB' for component in tbl_ashall['filter']]
tbl_ashall_new['filter'] = tbl_ashall['filter']
tbl_ashall_new['depth_5sig'] = MaskedColumn(data=[None]*len(tbl_ashall), mask=True)
tbl_ashall_new['zp'] = MaskedColumn(data=[None]*len(tbl_ashall), mask=True)
tbl_ashall_new['observatory'] = tbl_ashall['observatory']
tbl_ashall_new['detected'] = True
tbl_ashall_new.sort('obsdate')
tbl_ashall_new.write('/data1/supernova_rawdata/SN2021aefx/photometry/Ashall2022_new.dat', format='ascii.fixed_width', overwrite=True)
"""
#%%
tbl_phot = vstack([tbl_IMSNG, tbl_hosse, tbl_ashall])
tbl_phot.write('/data1/supernova_rawdata/SN2021aefx/photometry/all_phot.dat', format = 'ascii.fixed_width', overwrite = True)
tbl_spec = Table.read('/home/hhchoi1022/Desktop/Gitrepo/Research/spectroscopy/timeseriesspec.synphot', format = 'ascii.fixed_width')
tbl_spec.write('/data1/supernova_rawdata/SN2021aefx/photometry/all_spec.dat', format = 'ascii.fixed_width', overwrite = True)
tbl_photspec = vstack([tbl_phot, tbl_spec])
tbl_photspec.write('/data1/supernova_rawdata/SN2021aefx/photometry/all_photspec.dat', format = 'ascii.fixed_width', overwrite = True)
tbl_filters = tbl_photspec.group_by('filter').groups
# %%
import matplotlib.pyplot as plt

i = 12

tbl_show_RASA36 = tbl_filters[i][tbl_filters[i]['observatory'] == 'RASA36']
tbl_show_KCT = tbl_filters[i][tbl_filters[i]['observatory'] == 'KCT']
tbl_show_LSGT = tbl_filters[i][tbl_filters[i]['observatory'] == 'LSGT']
tbl_show_LC1 = tbl_filters[i][tbl_filters[i]['observatory'] == 'LCO(H22)']
tbl_show_SWIFT = tbl_filters[i][tbl_filters[i]['observatory'] == 'Swift']
tbl_show_Swope = tbl_filters[i][tbl_filters[i]['observatory'] == 'Swope']
tbl_show_en12 = tbl_filters[i][tbl_filters[i]['observatory'] == 'en12']
tbl_show_salt = tbl_filters[i][tbl_filters[i]['observatory'] == 'SALT']
tbl_show_SOAR = tbl_filters[i][tbl_filters[i]['observatory'] == 'GHTS_RED']
tbl_show_RASA36
plt.title(f'filter = {tbl_filters[i][0]["filter"]}')
# Black = RASA36
plt.scatter(Time(tbl_show_RASA36['obsdate'], format = 'mjd').datetime, tbl_show_RASA36['mag'], facecolor = 'none', edgecolor = 'k',  s = 50, marker = 's')
plt.errorbar(Time(tbl_show_RASA36['obsdate'], format = 'mjd').datetime, tbl_show_RASA36['mag'], tbl_show_RASA36['e_mag'], fmt = 'none', c= 'k')
# Red = KCT
plt.scatter(Time(tbl_show_KCT['obsdate'], format = 'mjd').datetime, tbl_show_KCT['mag'],facecolor = 'none', edgecolor = 'r', s = 50, marker = 'o')
plt.errorbar(Time(tbl_show_KCT['obsdate'], format = 'mjd').datetime, tbl_show_KCT['mag'], tbl_show_KCT['e_mag'], fmt = 'none', c= 'r')
# Blue = LSGT
plt.scatter(Time(tbl_show_LSGT['obsdate'], format = 'mjd').datetime, tbl_show_LSGT['mag'], facecolor = 'none', edgecolor = 'b', s = 50, marker = 'o')
plt.errorbar(Time(tbl_show_LSGT['obsdate'], format = 'mjd').datetime, tbl_show_LSGT['mag'], tbl_show_LSGT['e_mag'], fmt = 'none', c= 'b', marker = 'o')
# green = LasCumbres1m#
plt.scatter(Time(tbl_show_LC1['obsdate'], format = 'mjd').datetime, tbl_show_LC1['mag'], facecolor = 'none', edgecolor = 'g', s = 50, marker = 'o')
plt.errorbar(Time(tbl_show_LC1['obsdate'], format = 'mjd').datetime, tbl_show_LC1['mag'], tbl_show_LC1['e_mag'], fmt = 'none', c= 'g', marker = 'o')
# Yellow = Swope
plt.scatter(Time(tbl_show_Swope['obsdate'], format = 'mjd').datetime, tbl_show_Swope['mag'], facecolor = 'none', edgecolor = 'magenta', s = 50, marker = 'o')
plt.errorbar(Time(tbl_show_Swope['obsdate'], format = 'mjd').datetime, tbl_show_Swope['mag'], tbl_show_Swope['e_mag'], fmt = 'none', c= 'magenta', marker = 'o')
# Black = en12
#plt.scatter(Time(tbl_show_en12['obsdate'], format = 'mjd').datetime, tbl_show_en12['mag'], facecolor = 'none', edgecolor = 'k', s = 150, marker = 'o')
#plt.errorbar(Time(tbl_show_en12['obsdate'], format = 'mjd').datetime, tbl_show_en12['mag'], tbl_show_en12['e_mag'], fmt = 'none', c= 'magenta', marker = 'o')
# Black = SALT
#plt.scatter(Time(tbl_show_salt['obsdate'], format = 'mjd').datetime, tbl_show_salt['mag'], facecolor = 'none', edgecolor = 'k', s = 100, marker = 's')
#plt.errorbar(Time(tbl_show_salt['obsdate'], format = 'mjd').datetime, tbl_show_salt['mag'], tbl_show_salt['e_mag'], fmt = 'none', c= 'magenta', marker = 's')
# Black = SOAR
#plt.scatter(Time(tbl_show_SOAR['obsdate'], format = 'mjd').datetime, tbl_show_SOAR['mag'], facecolor = 'none', edgecolor = 'k', s = 150, marker = 'o')
#plt.errorbar(Time(tbl_show_SOAR['obsdate'], format = 'mjd').datetime, tbl_show_SOAR['mag'], tbl_show_SOAR['e_mag'], fmt = 'none', c= 'magenta', marker = 'o')

#plt.scatter(Time(tbl_LSGT['obsdate'], format ='mjd').datetime, tbl_LSGT['mag'], c= 'g')
#plt.scatter(Time(tbl_past['obsdate'], format ='mjd').datetime, tbl_past['mag'], c= 'r')
#plt.scatter(Time(tbl_ashall_g['obsdate'], format ='mjd').datetime, tbl_ashall_g['mag'], c= 'b')
#plt.errorbar(Time(tbl_ashall_g['obsdate'], format ='mjd').datetime, tbl_ashall_g['mag'], tbl_ashall_g['e_mag'], fmt = 'none', c = 'b')
#plt.scatter(Time(tbl_Hosse_g['obsdate'], format ='mjd').datetime, tbl_Hosse_g['mag'], c= 'r')
#plt.grid()
#plt.xlim(datetime(year = 2021, month = 11, day = 10), datetime(year = 2022, month = 1, day = 10))
#plt.xlim(datetime(year = 2022, month = 1, day = 10), datetime(year = 2022, month = 4, day = 20))
#plt.ylim(14.5,15.5)
import numpy as np
#plt.yticks(np.arange(17, 16, -1))

plt.axvline(datetime(year = 2021, month = 11, day = 25))
plt.axvline(datetime(year = 2022, month = 1, day = 25))
Time(datetime(year = 2022, month = 1, day = 25)).mjd
# %%
# Get the g-band and r-band data
i_g = 12  # g-band index
i_r = 14  # r-band index

tbl_g = tbl_filters[i_g]
tbl_r = tbl_filters[i_r]

# Observatories and their markers/colors
observatories = {
    'RASA36': {'marker': 's', 'color': 'black'},
    'KCT': {'marker': 'o', 'color': 'red'},
    'LSGT': {'marker': 'v', 'color': 'blue'},
    'LasCumbres1m': {'marker': '^', 'color': 'green'},
    'Swope': {'marker': 'p', 'color': 'orange'},
    'en12': {'marker': 'h', 'color': 'purple'},
    'SALT': {'marker': 'D', 'color': 'magenta'},
    'GHTS_RED': {'marker': '*', 'color': 'cyan'},
}

from Research.helper import PhotometryHelper
phot_helper = PhotometryHelper()
plt.figure(figsize=(10, 6))

# Iterate over each observatory
for obs, style in observatories.items():
    # Filter data by observatory
    tbl_g_obs = tbl_g[tbl_g['observatory'] == obs]
    tbl_r_obs = tbl_r[tbl_r['observatory'] == obs]
    matched_tbl = phot_helper.match_table(tbl_g_obs, tbl_r_obs, key = 'obsdate', tolerance = 0.3) 

    if len(matched_tbl) > 0:
        dates_g = Time(matched_tbl['obsdate_1'], format = 'mjd').datetime
        dates_r = Time(matched_tbl['obsdate_2'], format = 'mjd').datetime
        # Extract magnitudes for common dates
        g_mags = matched_tbl['mag_1']
        r_mags = matched_tbl['mag_2']

        # Calculate g-r magnitudes
        g_r_mags = g_mags - r_mags

        # Get the errors in g and r bands
        g_errors = matched_tbl['e_mag_1']
        r_errors = matched_tbl['e_mag_2']

        # Propagate the errors for g-r
        g_r_errors = np.sqrt(g_errors**2 + r_errors**2)

        # Plot the g-r light curve with different markers for each observatory
        plt.errorbar(dates_g, g_r_mags, yerr=g_r_errors, fmt=style['marker'], 
                     color=style['color'], label=obs, markersize=8)


plt.title('g-r Light Curve by Observatory')
plt.xlabel('Date')
plt.ylabel('g-r Magnitude')
plt.gca().invert_yaxis()  # Magnitudes decrease upwards
plt.grid(True)
plt.legend(loc='best')
plt.xlim(datetime(year = 2021, month = 11, day = 10), datetime(year = 2022, month = 2, day = 16))
plt.ylim(-0.5, 1.5)
plt.show()

# %%

################ SN 2023rve

output_filelist = []
# KCT in g band
output_file = write_photometry_file(
                                    key='/mnt/data1/supernova_rawdata/SN2023rve/analysis/KCT*/g/*.phot',
                                    savepath='/mnt/data1/supernova_rawdata/SN2023rve/analysis/IMSNG_KCT_g.dat',
                                    filter_='g',
                                    observatory='KCT'
                                    )   
output_filelist.append(output_file)
# KCT in r band
output_file = write_photometry_file(
                                    key='/mnt/data1/supernova_rawdata/SN2023rve/analysis/KCT*/r/*.phot',
                                    savepath='/mnt/data1/supernova_rawdata/SN2023rve/analysis/IMSNG_KCT_r.dat',
                                    filter_='r',
                                    observatory='KCT'
                                    )
output_filelist.append(output_file)
# RASA36 in r band 

output_file = write_photometry_file(
                                    key='/mnt/data1/supernova_rawdata/SN2023rve/analysis/RASA36/r/*.phot',
                                    savepath='/mnt/data1/supernova_rawdata/SN2023rve/analysis/IMSNG_RASA36_r.dat',
                                    filter_='r',
                                    observatory='RASA36'
                                    )
output_filelist.append(output_file)

# LSGT in r band 
output_file = write_photometry_file(
                                    key='/mnt/data1/supernova_rawdata/SN2023rve/analysis/LSGT_ASI1600MM/r/*.phot',
                                    savepath='/mnt/data1/supernova_rawdata/SN2023rve/analysis/IMSNG_LSGT_r.dat',
                                    filter_='r',
                                    observatory='LSGT'
                                    )
output_filelist.append(output_file)
# %%
output_filelist = glob.glob('/mnt/data1/supernova_rawdata/SN2023rve/analysis/IMSNG*.dat')
from astropy.table import vstack
tbl_IMSNG = vstack([Table.read(output_file, format = 'ascii.fixed_width') for output_file in output_filelist])  
tbl_IMSNG.write('/mnt/data1/supernova_rawdata/SN2023rve/analysis/all_IMSNG.dat', format = 'ascii.fixed_width', overwrite = True)
#%%
tbl_phot = vstack([tbl_IMSNG])
tbl_phot.write('/mnt/data1/supernova_rawdata/SN2023rve/analysis/all_phot.dat', format = 'ascii.fixed_width', overwrite = True)
tbl_filters = tbl_phot.group_by('filter').groups
# %%
import matplotlib.pyplot as plt

i = 1

tbl_show_RASA36 = tbl_filters[i][tbl_filters[i]['observatory'] == 'RASA36']
tbl_show_KCT = tbl_filters[i][tbl_filters[i]['observatory'] == 'KCT']
tbl_show_LSGT = tbl_filters[i][tbl_filters[i]['observatory'] == 'LSGT']

plt.figure(dpi = 300, figsize = (8,5))
plt.title(f'filter = {tbl_filters[i][0]["filter"]}')
# Black = RASA36
plt.scatter(Time(tbl_show_RASA36['obsdate'], format = 'mjd').datetime, tbl_show_RASA36['mag'], facecolor = 'none', edgecolor = 'k',  s = 50, marker = 'o')
plt.errorbar(Time(tbl_show_RASA36['obsdate'], format = 'mjd').datetime, tbl_show_RASA36['mag'], tbl_show_RASA36['e_mag'], fmt = 'none', c= 'k')
# Red = KCT
plt.scatter(Time(tbl_show_KCT['obsdate'], format = 'mjd').datetime, tbl_show_KCT['mag'],facecolor = 'none', edgecolor = 'r', s = 50, marker = 'o')
plt.errorbar(Time(tbl_show_KCT['obsdate'], format = 'mjd').datetime, tbl_show_KCT['mag'], tbl_show_KCT['e_mag'], fmt = 'none', c= 'r')
# Red = LSGT
plt.scatter(Time(tbl_show_LSGT['obsdate'], format = 'mjd').datetime, tbl_show_LSGT['mag'],facecolor = 'none', edgecolor = 'r', s = 50, marker = 's')
plt.errorbar(Time(tbl_show_LSGT['obsdate'], format = 'mjd').datetime, tbl_show_LSGT['mag'], tbl_show_LSGT['e_mag'], fmt = 'none', c= 'r')
# Blue = LSGT
plt.scatter(Time(tbl_show_RASA36['obsdate'], format = 'mjd').datetime, tbl_show_RASA36['depth_5sig'], facecolor = 'none', edgecolor = 'k',  s = 50, marker = 'o', alpha = 0.3)
plt.errorbar(Time(tbl_show_RASA36['obsdate'], format = 'mjd').datetime, tbl_show_RASA36['depth_5sig'], tbl_show_RASA36['e_mag'], fmt = 'none', c= 'k', alpha = 0.3)
# Red = KCT
plt.scatter(Time(tbl_show_KCT['obsdate'], format = 'mjd').datetime, tbl_show_KCT['depth_5sig'],facecolor = 'none', edgecolor = 'r', s = 50, marker = 'o', alpha = 0.3)
plt.errorbar(Time(tbl_show_KCT['obsdate'], format = 'mjd').datetime, tbl_show_KCT['depth_5sig'], tbl_show_KCT['e_mag'], fmt = 'none', c= 'r', alpha = 0.3)

#plt.scatter(Time(tbl_show_LSGT['obsdate'], format = 'mjd').datetime, tbl_show_LSGT['mag'], facecolor = 'none', edgecolor = 'b', s = 50, marker = 'o')
#plt.errorbar(Time(tbl_show_LSGT['obsdate'], format = 'mjd').datetime, tbl_show_LSGT['mag'], tbl_show_LSGT['e_mag'], fmt = 'none', c= 'b', marker = 'o')

plt.grid()
#plt.xlim(datetime(year = 2023, month = 10, day = 1), datetime(year = 2024, month = 1, day = 10))
#plt.xlim(datetime(year = 2022, month = 1, day = 10), datetime(year = 2022, month = 4, day = 20))
plt.ylim(22,13.5)
import numpy as np
# %%