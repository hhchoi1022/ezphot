#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 16:50:29 2022

@author: hhchoi1022
"""
#%%
from astropy.io import fits
import numpy as np
import os, glob

hdu = fits.open('/data2/temp/SN/NGC1566/RASA36/r/NOINFO/Calib-RASA36-NGC1566-20211015-055944-r-60.fits')[0]
hdu.data = hdu.data[~np.isnan(hdu.data)]
hdu.data.sort()
mean = hdu.data[-500:-100]
#%%



def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100): 
    import sys
    
    formatStr = "{0:." + str(decimals) + "f}" 
    percent = formatStr.format(100 * (iteration / float(total))) 
    filledLength = int(round(barLength * iteration / float(total))) 
    bar = '#' * filledLength + '-' * (barLength - filledLength) 
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)), 
    if iteration == total: 
        sys.stdout.write('\n') 
    sys.stdout.flush() 
    
def RASA_dividemode(imlist, criteria, divide = True):
    from astropy.io import fits
    import os, glob
    import matplotlib.pyplot as plt
    
    meanlist = []
    highlist = []
    mergelist = []
    for i, image in enumerate(imlist):
        hdu = fits.open(image)[0]
        hdu.data = hdu.data[~np.isnan(hdu.data)]
        hdu.data.sort()
        mean = np.mean(hdu.data[-500:-100])
        meanlist.append(mean)
        printProgress(i, len(imlist), 'Progress', 'Complete', 1, 50)
        if mean < criteria:
            highlist.append(image)
        elif mean > criteria:
            mergelist.append(image)
    plt.scatter(range(len(meanlist)), meanlist)
        
    if divide == True:
        imdir = os.path.dirname(image)
        for image in highlist:
            os.makedirs(f'{imdir}/HIGH', exist_ok = True)
            os.system(f'cp {image} {imdir}/HIGH')
        for image in mergelist:
            os.makedirs(f'{imdir}/MERGE', exist_ok = True)
            os.system(f'cp {image} {imdir}/MERGE')
    return highlist, mergelist


#%%

imlist = glob.glob('/data1/supernova_rawdata/SN2021aefx/photometry/RASA36/r/*.fits')
#%%
RASA_dividemode(imlist, criteria = 20000, divide = True)

# %%
