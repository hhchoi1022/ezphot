#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 20:04:52 2022

@author: hhchoi1022
"""


from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import os, sys
from photutils.aperture import CircularAperture
from astropy.stats import sigma_clip

#%%
filter_ = 'b'
conf_param = dict(
    #####system configuration #####
    IMAGE_FILE = f'/data2/psfex/n2041{filter_}.fits',
    EXEDIR = '/data2/psfex/',
    REFERENCE_FILE = '/data2/psfex/ref_n2041.cat',
    #####SExtractor configuration #####
    SECONFIG_FILE = 'assignment2_SEconfig',
    SEOUTPUT_FILE = f'/data2/psfex/n2041{filter_}.cat',
    FACTOR_APERTURE = 2,
    #####PSF configuration #####
    PSFCONFIG_FILE = 'univ.PSFconfig',
    PSFOUTPUT_FILE = f'/data2/psfex/n2041{filter_}.psf',
    PSFPHOT_PARAMFILE = '/data2/psfex/psfphot.SEparam'
    )
#%%
os.chdir(conf_param['EXEDIR'])
SEcommand1 =  f'source-extractor {conf_param["IMAGE_FILE"]} -c {conf_param["SECONFIG_FILE"]} -CATALOG_NAME {conf_param["SEOUTPUT_FILE"]}'

os.system(SEcommand1)
firdet = ascii.read(conf_param['SEOUTPUT_FILE'])
firdet = firdet[firdet['MAG_APER']<50]
#%%
seeingclip = sigma_clip(firdet['FWHM_IMAGE'],sigma=2,maxiters=3)
selected_seeing = firdet[~seeingclip.mask]

plt.figure(dpi = 100)
plt.ylabel('Seeing [pix]')
plt.xlabel(r'$mag_{instrument} $')
plt.scatter(firdet['MAG_APER'], firdet['FWHM_IMAGE'], marker = '.', linewidth = 0, c = 'k')
plt.scatter(selected_seeing['MAG_APER'].data, selected_seeing['FWHM_IMAGE'].data, marker = '.', linewidth = 0, c = 'r', label = f'median = %.2f' %(np.median(selected_seeing["FWHM_IMAGE"])))
plt.legend()
plt.show()

seeing = round(np.median(selected_seeing['FWHM_IMAGE']),3) * conf_param['FACTOR_APERTURE']

SEcommand2 = f'source-extractor {conf_param["IMAGE_FILE"]} -c {conf_param["SECONFIG_FILE"]} -CATALOG_NAME {conf_param["SEOUTPUT_FILE"]} -PHOT_APERTURES {seeing}'
os.system(SEcommand2)


#%%
os.chdir(conf_param['EXEDIR'])
SEcommand3 = f'source-extractor {conf_param["IMAGE_FILE"]} -c {conf_param["SECONFIG_FILE"]} -CATALOG_TYPE FITS_LDAC -CATALOG_NAME {conf_param["SEOUTPUT_FILE"]} -PHOT_APERTURES {seeing} -DETECT_THRESH 5 -DETECT_MAXAREA 0'
os.system(SEcommand3)

PSFcommand1 = f'psfex {conf_param["SEOUTPUT_FILE"]} -c {conf_param["PSFCONFIG_FILE"]}'
os.system(PSFcommand1)
PSFcommand1
SEcommand4 =  f'source-extractor {conf_param["IMAGE_FILE"]} -c {conf_param["SECONFIG_FILE"]} -PHOT_APERTURES 10 -CATALOG_NAME {conf_param["SEOUTPUT_FILE"]} -PHOT_APERTURES {seeing} -PARAMETERS_NAME {conf_param["PSFPHOT_PARAMFILE"]} -DEBLEND_MINCONT 0.000001 -PSF_NAME {conf_param["PSFOUTPUT_FILE"]}'
os.system(SEcommand4)

#SEcommand4 =  f'source-extractor {conf_param["IMAGE_FILE"]},/data2/psfex/n2041v.fits  -c {conf_param["SECONFIG_FILE"]} -CATALOG_NAME {conf_param["SEOUTPUT_FILE"]} -PHOT_APERTURES {seeing} -PARAMETERS_NAME {conf_param["PSFPHOT_PARAMFILE"]} -DEBLEND_MINCONT 0.000001 -PSF_NAME {conf_param["PSFOUTPUT_FILE"]}'
SEcommand4 =  f'source-extractor {conf_param["IMAGE_FILE"]} -c {conf_param["SECONFIG_FILE"]} -CATALOG_NAME {conf_param["SEOUTPUT_FILE"]} -PHOT_APERTURES {seeing} -PARAMETERS_NAME {conf_param["PSFPHOT_PARAMFILE"]} -DEBLEND_MINCONT 0.000001 -PSF_NAME {conf_param["PSFOUTPUT_FILE"]}'

SEcommand4
os.system(SEcommand4)
#%%
detections_b = ascii.read(conf_param['SEOUTPUT_FILE'])
detections_v = ascii.read(conf_param['SEOUTPUT_FILE'])

#%% ZP calculation
from astropy.stats import sigma_clip
reftbl = ascii.read(conf_param['REFERENCE_FILE'])

refidx_b=  []
for i in range(len(reftbl)):
    refcoord_x, refcoord_y = reftbl['x','y'][i]
    match_idx = np.argmin(np.abs((refcoord_x-detections_b['X_IMAGE'])**2+(refcoord_y-detections_b['Y_IMAGE'])**2))
    refidx_b.append(match_idx)
zp_b = np.median(np.array(reftbl['B[mag]'].astype(float))-np.array(detections_b[refidx_b]['MAG_PSF']))
zperr_b = np.std(np.array(detections_b[refidx_b]['MAG_PSF'])-np.array(reftbl['B[mag]'].astype(float)))
plt.figure(dpi = 300)
plt.title('B band')
plt.errorbar(range(len(reftbl)),np.array(reftbl['B[mag]'].astype(float))-np.array(detections_b[refidx_b]['MAG_PSF']),yerr = np.sqrt((reftbl['errB[mag]'].astype(float))**2+(detections_b[refidx_b]['MAGERR_PSF']**2)), fmt = '.', c= 'k', capsize = 3)
plt.axhline(zp_b, c= 'k', linestyle = '--', linewidth = 1, label = 'zp = %.3f'%zp_b)
plt.fill_between([-1,15], [zp_b-zperr_b], [zp_b+zperr_b], color='k', alpha=0.2, label = 'e_ZP = %.3f'%zperr_b)
plt.xlim(-1,15)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
plt.ylim(zp_b-1,zp_b+1)
plt.legend()
plt.show()

#%%

refidx_v=  []
for i in range(len(reftbl)):
    refcoord_x, refcoord_y = reftbl['x','y'][i]
    match_idx = np.argmin(np.abs((refcoord_x-detections_v['X_IMAGE'])**2+(refcoord_y-detections_v['Y_IMAGE'])**2))
    refidx_v.append(match_idx)
zp_v = np.array(reftbl['V[mag]'].astype(float))-np.array(detections_v[refidx_v]['MAG_PSF'])
zpclip = sigma_clip(zp_v,sigma=2,maxiters=1)
zp_vclip = zpclip.data[~zpclip.mask]
zp_v = np.median(zp_v)
zperr_v = np.std(np.array(detections_v[refidx_v]['MAG_PSF'])-np.array(reftbl['V[mag]'].astype(float)))
zperr_v = np.std(zp_vclip)
plt.figure(dpi = 300)
plt.title('V band')
plt.errorbar(range(len(reftbl)),np.array(reftbl['V[mag]'].astype(float))-np.array(detections_v[refidx_v]['MAG_PSF']),yerr = np.sqrt((reftbl['errV[mag]'].astype(float))**2+(detections_v[refidx_v]['MAGERR_PSF']**2)), fmt = '.', c= 'k', capsize = 3)
plt.axhline(zp_v, c= 'k', linestyle = '--', linewidth = 1, label = 'zp = %.3f'%zp_v)
plt.fill_between([-1,15], [zp_v-zperr_v], [zp_v+zperr_v], color='k', alpha=0.2, label = 'e_ZP = %.3f'%zperr_v)
plt.xlim(-1,15)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
plt.ylim(zp_v-1,zp_v+1)
plt.legend()
plt.show()
#%%

detections_b['MAG_REAL'] = detections_b['MAG_PSF'] + zp_b
detections_v['MAG_REAL'] = detections_v['MAG_PSF'] + zp_v

#detections_b['MAG_REAL']-detections_v['MAG_REAL']
#%%
plt.figure(figsize = (10,6), dpi = 300)
plt.title('Color-Magnitude Diagram')
plt.xlim(-1,2.5)
plt.ylim(22,13)
plt.ylabel('V [mag]')
plt.xlabel('B-V [mag]')
plt.errorbar(detections_b['MAG_REAL']-detections_v['MAG_REAL'], detections_v['MAG_REAL'],xerr = np.sqrt(detections_b['MAGERR_PSF']**2+detections_v['MAGERR_PSF']**2),yerr = np.sqrt(detections_v['MAGERR_PSF']**2+zperr_v**2), c= 'k', fmt= '.', linewidth  =0.2)
detections_v
zperr_v
#%%
from astropy.table import join

detections_bv = join(detections_b,detections_v,'NUMBER')
detections_bv['B-V'] = detections_bv['MAG_REAL_1']-detections_bv['MAG_REAL_2']
detections_bv['B-Verr'] = np.sqrt(detections_bv['MAGERR_PSF_1']**2+detections_bv['MAGERR_PSF_2']**2)
detections_bv['Verr'] = np.sqrt(detections_bv['MAGERR_PSF_2']**2+zperr_v**2)
#%%
plt.figure(figsize = (6,4), dpi = 300)
plt.title('Color-Magnitude Diagram')
plt.xlim(-1,2.5)
plt.ylim(22,13)
plt.ylabel('V [mag]')
plt.xlabel('B-V [mag]')
plt.errorbar(detections_blue['B-V'], detections_blue['MAG_REAL_2'],xerr = detections_blue['B-Verr'],yerr = detections_blue['Verr'], c= 'b', fmt= '.', linewidth  =0.2)
plt.errorbar(detections_faint['B-V'], detections_faint['MAG_REAL_2'],xerr = detections_faint['B-Verr'],yerr = detections_faint['Verr'], c= 'r', fmt= '.', linewidth  =0.2)
plt.errorbar(detections_bright['B-V'], detections_bright['MAG_REAL_2'],xerr = detections_bright['B-Verr'],yerr = detections_bright['Verr'], c= 'y', fmt= '.', linewidth  =0.2)
plt.axvline(0.5, c= 'k', linestyle = '--')
plt.axhline(17, 0.43, 2.5, c= 'k', linestyle = '--')
#%%
blue_cut = 0.5
faint_cut = 17
detections_blue = detections_bv[detections_bv['B-V']<blue_cut]
detections_red = detections_bv[detections_bv['B-V']>blue_cut]
detections_faint = detections_red[detections_red['MAG_REAL_2']>faint_cut]
detections_bright = detections_red[detections_red['MAG_REAL_2']<faint_cut]

#%%
#detections_b = detections_b[(detections_b['MAG_APER']<50)&(detections_b['MAG_PSF']<50)]

refidx_b=  []
for i in range(len(reftbl)):
    refcoord_x, refcoord_y = reftbl['x','y'][i]
    match_idx = np.argmin(np.abs((refcoord_x-detections_b['X_IMAGE'])**2+(refcoord_y-detections_b['Y_IMAGE'])**2))
    refidx_b.append(match_idx)

#plt.scatter(np.array(detections_b[refidx_b]['MAG_PSF']+zp_b),np.array(detections_b[refidx_b]['MAG_APER']+zp_b))
plt.figure(dpi = 300)
plt.title('B')
plt.plot([14,22],[14,22], linestyle = '--', c='k')
plt.errorbar(np.array(detections_b['MAG_PSF']+zp_b),np.array(detections_b['MAG_APER']+zp_b), xerr = np.array(np.sqrt(detections_b['MAGERR_PSF']**2+zperr_b**2)),yerr = np.array(np.sqrt(detections_b['MAGERR_APER']**2+zperr_b**2)), alpha = 0.1, fmt = '.', c= 'k', label = f'detections[{len(detections_b)}]')
plt.scatter(np.array(detections_b[refidx_b]['MAG_PSF']+zp_b),np.array(detections_b[refidx_b]['MAG_APER']+zp_b), c='r', marker = 's', s =20, label = 'reference stars')
plt.xlabel(r'$m_{PSF}$')
plt.ylabel(r'$m_{APER, 10pix}$')
plt.legend()
plt.grid()


#%%
#detections_v = detections_v[(detections_v['MAG_APER']<50)&(detections_v['MAG_PSF']<50)]
refidx_v=  []
for i in range(len(reftbl)):
    refcoord_x, refcoord_y = reftbl['x','y'][i]
    match_idx = np.argmin(np.abs((refcoord_x-detections_v['X_IMAGE'])**2+(refcoord_y-detections_v['Y_IMAGE'])**2))
    refidx_v.append(match_idx)

#plt.scatter(np.array(detections_b[refidx_b]['MAG_PSF']+zp_b),np.array(detections_b[refidx_b]['MAG_APER']+zp_b))
plt.figure(dpi = 300)
plt.title('V')
plt.plot([14,22],[14,22], linestyle = '--', c='k')
plt.errorbar(np.array(detections_v['MAG_PSF']+zp_v),np.array(detections_v['MAG_APER']+zp_v), xerr = np.array(np.sqrt(detections_v['MAGERR_PSF']**2+zperr_v**2)),yerr = np.array(np.sqrt(detections_v['MAGERR_APER']**2+zperr_v**2)), alpha = 0.1, fmt = '.', c= 'k', label = f'detections[{len(detections_b)}]')
plt.scatter(np.array(detections_v[refidx_v]['MAG_PSF']+zp_v),np.array(detections_v[refidx_v]['MAG_APER']+zp_v), c='r', marker = 's', s =20, label = 'reference stars')
plt.xlabel(r'$m_{PSF}$')
plt.ylabel(r'$m_{APER, 10pix}$')
plt.legend()
plt.grid()


































#%%

detections_blue['X']
#%%
#detections = ascii.read(conf_param['SEOUTPUT_FILE'], guess = True)
detections = firdet
#detections = detections[detections['FLAGS'] == 0]
plt.figure(dpi = 1000)
positions_blue = np.transpose((detections_blue['X_IMAGE_1']-1, detections_blue['Y_IMAGE_1']-1))
apertures_blue = CircularAperture(positions_blue, r=5.)
positions_bright = np.transpose((detections_bright['X_IMAGE_1']-1, detections_bright['Y_IMAGE_1']-1))
apertures_bright = CircularAperture(positions_bright, r=5.)
positions_faint = np.transpose((detections_faint['X_IMAGE_1']-1, detections_faint['Y_IMAGE_1']-1))
apertures_faint = CircularAperture(positions_faint, r=5.)
#norm = ImageNormalize(stretch=SqrtStretch())
data = fits.getdata(conf_param['IMAGE_FILE'])
norm = simple_norm(data, 'sqrt', percent = 90)
plt.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
Tapertures_blue.plot(color='blue', lw=0.5, alpha=1)
apertures_faint.plot(color='red', lw=0.5, alpha=1)
apertures_bright.plot(color='yellow', lw=0.5, alpha=1)










#%%







len(detections)

































