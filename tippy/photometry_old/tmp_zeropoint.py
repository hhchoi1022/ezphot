#%%
from astropy.table import Table
from photometry import Photometry
from update_fitsinfo import Update_fitsinfo
# %%

imkey = '/data2/SN2021aefx_220627/RASA36/r/HIGH/com_align_Calib-RASA36-NGC1566-20220430-232917-r-60.fits'
#imkey = '/data2/SN2021aefx_220627/KCT_STX16803/r/Calib-KCT_STX16803-NGC1566-20220301-015834-r-120.fits'
imkey = '/data2/SN2021aefx_220627/LSGT/r/com_align_Calib-LSGT-NGC1566-20220320-105432-r-180.fits'
# 
updater = Update_fitsinfo(imkey = imkey, telescope = 'RASA36', ccd = 'KL4040')
updater = Update_fitsinfo(imkey = imkey, telescope = 'KCT', ccd = 'STX16803')
updater = Update_fitsinfo(imkey = imkey, telescope = 'LSGT', ccd = 'STX16803')
# %%
target, filter, binning = updater.get_image_info(updater.imlist[0])
apass_tbl, _ = updater.load_APASS(target = target, filter_ = filter, mag_upper = 15, mag_lower = 12)
smss_tbl, _ = updater.load_SMSS(target = target, filter_ = filter, class_star = 0.9, flag = 1, n_good = 2, mag_upper = 15, mag_lower = 12)
from HHsupport_phot import load_sexconfig
conf_param = load_sexconfig()
# %%
conf_param['CATALOG_NAME'] = 'zeropoint.cat'
conf_param['PARAMETERS_NAME'] = 'zeropoint.SEparam'
conf_param['DETECT_THRESH'] = updater.threshold
conf_param['SEEING_FWHM'] = updater.seeing_guess
conf_param['GAIN'] = float(updater.obsinfo['gain'])
conf_param['SATUR_LEVEL'] = 65536
conf_param['PIXEL_SCALE'] = float(updater.obsinfo['pixelscale'])
conf_param['BACK_SIZE'] = float(128)
# %%
from HHsupport_phot import run_sextractor
# %%
obj_tbl1 = run_sextractor(updater.imlist[0], conf_param)
obj_tbl1_cut = obj_tbl1[
                    (obj_tbl1['FLAGS'] == 0)&
                    (obj_tbl1['MAGERR_AUTO'] < 0.1)&
                    (3600*obj_tbl1['FWHM_WORLD'] > 1)#&
                    #(3600*obj_tbl1['FWHM_WORLD'] < 5) 
                    ]
#%%
from HHsupport_phot import to_skycoord
from HHsupport_phot import cross_match
from astropy.stats import sigma_clip
import numpy as np
sky_coords = to_skycoord(apass_tbl['ra'], apass_tbl['dec'])
sky_coords = to_skycoord(smss_tbl['ra'], smss_tbl['dec'])

obj_coords1 = to_skycoord(obj_tbl1_cut['ALPHA_J2000'], obj_tbl1_cut['DELTA_J2000'])
matched_obj_idx1, matched_sky_idx1, _ = cross_match(obj_coords1, sky_coords, 4*float(updater.obsinfo['pixelscale']))
seeing1 = round(3600*np.median(sigma_clip(obj_tbl1_cut[matched_obj_idx1]['FWHM_WORLD'],sigma=3,maxiters=1).data),3)
conf_param['PHOT_APERTURES'] = round(updater.aperfactor*seeing1/float(updater.obsinfo['pixelscale']), 3)
conf_param['SEEING_FWHM'] = seeing1
# %%
obj_tbl2 = run_sextractor(updater.imlist[0], conf_param)
obj_tbl2_cut = obj_tbl2[
                (obj_tbl2['FLAGS'] <= 1)&
                (obj_tbl2['MAGERR_AUTO'] < 0.1)&
                (3600*obj_tbl2['FWHM_WORLD'] > 1)&
                #(3600*obj_tbl2['FWHM_WORLD'] < 5)&
                (obj_tbl2['CLASS_STAR'] > 0.9)
                ]
# %%
obj_coords2 = to_skycoord(obj_tbl2_cut['ALPHA_J2000'], obj_tbl2_cut['DELTA_J2000'])
matched_obj_idx2, matched_sky_idx2, _ = cross_match(obj_coords2, sky_coords, 1.5*seeing1)
# %%
obj_tbl = obj_tbl2_cut[matched_obj_idx2]
sky_tbl = apass_tbl[matched_sky_idx2]
sky_tbl = smss_tbl[matched_sky_idx2]

# %%

def func(x,a,b):
    y = a*x+b
    return y

# %%
from scipy.optimize import curve_fit
# %%
x = sky_tbl['g_mag']-sky_tbl['r_mag']
y = sky_tbl['r_mag']-obj_tbl['MAG_APER']
yerr = np.sqrt(sky_tbl['e_r_mag']**2+obj_tbl['MAGERR_APER']**2)
xerr = np.sqrt(sky_tbl['e_r_mag']**2+sky_tbl['e_g_mag']**2)
# %%
popt, pcov = curve_fit(func, x, y, sigma = yerr)

# %%
xgrid = np.linspace(np.min(sky_tbl['g_mag']-sky_tbl['r_mag']),np.max(sky_tbl['g_mag']-sky_tbl['r_mag']),100)
# %%
yvalue = func(xgrid, popt[0], popt[1])
# %%
import matplotlib.pyplot as plt


# %%
plt.figure(figsize = (6,4), dpi = 300)
plt.title('LSGT')
plt.plot(xgrid, yvalue, c = 'r', linestyle = '--', linewidth = 1, label = f'ZP = {round(popt[0],3)}(g-r)+{round(popt[1],3)}')
plt.scatter(sky_tbl['g_mag']-sky_tbl['r_mag'], sky_tbl['r_mag']-obj_tbl['MAG_APER'], c = 'k', s = 1)
plt.errorbar(sky_tbl['g_mag']-sky_tbl['r_mag'], sky_tbl['r_mag']-obj_tbl['MAG_APER'],xerr = np.sqrt(sky_tbl['e_r_mag']**2+sky_tbl['e_g_mag']**2), yerr = np.sqrt(sky_tbl['e_r_mag']**2+obj_tbl['MAGERR_APER']**2), fmt = 'none', c = 'k', elinewidth = 0.2)
plt.ylabel(r' ZP$_r$ [APASS - obs]')
plt.legend()
plt.ylim(26.5,27.75)
plt.xlabel('g-r [APASS]')
# %%
plt.figure(figsize = (6,4), dpi = 300)
plt.title('LSGT')
plt.plot(xgrid, yvalue, c = 'r', linestyle = '--', linewidth = 1, label = f'ZP = {round(popt[0],3)}(g-r)+{round(popt[1],3)}')
plt.scatter(sky_tbl['g_mag']-sky_tbl['r_mag'], sky_tbl['r_mag']-obj_tbl['MAG_APER'], c = 'k', s = 1)
plt.errorbar(sky_tbl['g_mag']-sky_tbl['r_mag'], sky_tbl['r_mag']-obj_tbl['MAG_APER'],xerr = np.sqrt(sky_tbl['e_r_mag']**2+sky_tbl['e_g_mag']**2), yerr = np.sqrt(sky_tbl['e_r_mag']**2+obj_tbl['MAGERR_APER']**2), fmt = 'none', c = 'k', elinewidth = 0.2)
plt.ylabel(r' ZP$_r$ [SMSS - obs]')
plt.legend()
plt.ylim(26.5,27.75)
plt.xlabel('g-r [SMSS]')
# %%
