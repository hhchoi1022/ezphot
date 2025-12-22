

#%%
from ezphot.methods import *
from ezphot.imageobjects import *
from ezphot.helper import Helper
from ezphot.utils import DataBrowser
from ezphot.skycatalog import *
import psutil, os
from pympler import asizeof

from tqdm import tqdm
import gc
import time
import numpy as np
#%%
from ezphot.utils import SDTDataQuerier
sdtquerier = SDTDataQuerier()
sdtquerier.sync_scidata(targetname = 'T22956')
#%%
db = DataBrowser('scidata')
db.observatory = '7DT'
db.objname = 'T22956'
db.keys
#%%
target_catset = db.search(pattern='calib*com.fits.cat', return_type='catalog')
# %%
target_ra = 233.857430764
target_dec = 12.0577222937
target_catset.select_sources(target_ra, target_dec, radius = 10)
# %%
from ezphot.dataobjects import LightCurve
lc = LightCurve(target_catset)
lc.extract_source_info(target_ra, target_dec)
# %%
lc.plt_params.figure_figsize = (14, 10)
lc.plt_params.ylim = [27, 8]
lc.plot(target_ra, target_dec, flux_key = 'MAGSKY_APER_4', fluxerr_key = 'MAGERR_APER_4')
#%%
m825_tbl = lc.data[lc.data['filter'] == 'm875']
telname_set = list(set(m825_tbl['telname']))
m825_unit4 = m825_tbl[m825_tbl['telname'] == telname_set[0]]
m825_unit10 = m825_tbl[m825_tbl['telname'] == telname_set[1]]
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.scatter(m825_unit4['mjd'], m825_unit4['MAGSKY_APER_4'], color='red')
plt.scatter(m825_unit10['mjd'], m825_unit10['MAGSKY_APER_4'], color='blue')
plt.errorbar(m825_unit4['mjd'], m825_unit4['MAGSKY_APER_4'], yerr=(m825_unit4['MAGERR_APER_4']**2 + m825_unit4['zp_err']**2)**0.5, color='red', label=f'{telname_set[0]}', fmt = 'none')
plt.errorbar(m825_unit10['mjd'], m825_unit10['MAGSKY_APER_4'], yerr=(m825_unit10['MAGERR_APER_4']**2 + m825_unit10['zp_err']**2)**0.5, color='blue', label=f'{telname_set[1]}', fmt = 'none')
plt.scatter(m825_unit4['mjd'], m825_unit4['MAGSKY_APER_4'] + (m825_unit4['MAGERR_APER_4']**2 + m825_unit4['zp_err']**2)**0.5, color='red', marker='x')
plt.ylim(18, 13)
plt.legend()
plt.show()
#%%
tbl_data = lc.data
tbl_g = tbl_data[tbl_data['filter'] == 'g']
tbl_r = tbl_data[tbl_data['filter'] == 'r']
tbl_i = tbl_data[tbl_data['filter'] == 'i']
helper_ezphot = Helper()
merged_tbl = helper_ezphot.match_table(tbl_g, tbl_r, key1 = 'mjd', key2 = 'mjd', tolerance = 0.1)
# Data
merged_tbl.sort('mjd_1')
#%%
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Example fix
x = merged_tbl['mjd_1']
y = merged_tbl['MAGSKY_APER_4_1'] - merged_tbl['MAGSKY_APER_4_2']

# Create a mask of valid entries only
mask = (~y.mask) & (~x.mask)  # keep only where both x and y are valid

x_valid = x[mask].data
y_valid = y[mask].data

# Sort
sorted_idx = np.argsort(x_valid)
x_sorted, y_sorted = x_valid[sorted_idx], y_valid[sorted_idx]

# Sort by x (important for spline fitting)
x_postpeak = x_sorted[x_sorted > 60810]
y_postpeak = y_sorted[x_sorted > 60810]

# Fit spline (s is smoothing factor; smaller s = tighter fit)
spl = UnivariateSpline(x_sorted, y_sorted, s=0.01)
linear = np.polyfit(x_postpeak, y_postpeak, 1)
linear_fn = np.poly1d(linear)

# Evaluate
x_fit = np.linspace(x_sorted.min(), x_sorted.max()+30, 10000)
y_fit = spl(x_fit)
x_fit_postpeak = np.linspace(60810, 60900, 500)
y_fit_postpeak = linear_fn(x_fit_postpeak)
# --- Plot ---
plt.figure(figsize=(6,4))
plt.scatter(x, y, color='gray', label='Observed (g - r)')
plt.plot(x_fit, y_fit, color='red', label='Spline fit')
plt.plot(x_fit_postpeak, y_fit_postpeak, color='blue', label='Post-peak linear fit')
plt.axvline(60810, ls='--', color='black', alpha=0.3, label='Post-peak boundary')
plt.xlabel('MJD')
plt.ylabel('g - r')
plt.legend()
plt.tight_layout()
plt.show()
#%%
from ezphot.imageobjects import ScienceImage
mag_keys = ['MAGSKY_AUTO', 'MAGSKY_APER', 'MAGSKY_APER_1', 'MAGSKY_APER_2', 'MAGSKY_APER_3', 'MAGSKY_APER_4']
corr_mag_dict = {mag: [] for mag in mag_keys}
for data in tbl_data:
    target_img = ScienceImage(data['target_img'])
    k = target_img.header['K_COLOR_APER_2_G-R']
    c = target_img.header['C_COLOR_APER_2_G-R']
    
    gr = spl(target_img.mjd)
    corrected_mag_offset = k * gr + c
    print("FILTER:", data['filter'], "MJD:", int(data['mjd']), "CORRECTION:", corrected_mag_offset)
    for mag in mag_keys:
        corr_mag_dict[mag].append(data[mag] + corrected_mag_offset)
for mag in mag_keys:
    mag_key_new = mag + '_CORR'
    tbl_data[mag_key_new] = corr_mag_dict[mag]
#%%
lc_corr = LightCurve()
lc_corr.data = tbl_data
# %%
lc_corr.plt_params.figure_figsize = (14, 10)
lc_corr.plt_params.ylim = [27, 8]
lc_corr.plot(target_ra, target_dec, flux_key = 'MAGSKY_APER_4_CORR', fluxerr_key = 'MAGERR_APER_4')

# %%
