#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 17:33:39 2022

@author: hhchoi1022
"""
#%%
from astropy.io import ascii, fits

# %%

home_dir = '/data2/SN2021aefx/Data/Hosseinzadeh/'
data = ascii.read(home_dir+'anc/SN2021aefx_phot.tex')

group_tel = data.group_by('Telescope')
group_tel.groups[1]
group_fil = data.group_by('Filter')
group_fil.groups[2]
group_fil.groups[0]['Apparent Magnitude']
import matplotlib.pyplot as plt
color = set(group_fil['Filter'])
color
len(color)
color = ['g','r','y','k']
group_fil.groups[9]
#%%
group_g = group_fil.groups[10]
group_i = group_fil.groups[11]
group_r = group_fil.groups[12]
group_c = group_fil.groups[9]
#%%
def find_mag(tbl):
    import re
    result = []
    for value in tbl:
        val = float(re.findall('(\d\d.\d\d)', value)[0])
        result.append(val)
    return result
def find_magerr(tbl):
    import re
    result = []
    for value in tbl:
        try:
            val = float(re.findall(' (\d.\d\d)', value)[0])
        except:
            val = 0.01
        result.append(val)
    return result
#%%
data_my = ascii.read('/home/hhchoi1022/Desktop/Gitrepo/photometry/fit_tbl.dat')
data_my = data_my[data_my['mag']!=data_my['UL5_4']]
group_fil_my = data_my.group_by('filter')
group_g_my = group_fil_my.groups[0]
group_i_my = group_fil_my.groups[1]
group_r_my = group_fil_my.groups[2]

#%%
import numpy as np
plt.figure(dpi = 300)
plt.errorbar(group_g['MJD'], np.array(find_mag(group_g['Apparent Magnitude']))-1,yerr = find_magerr(group_g['Apparent Magnitude']), c='g',fmt = 'None', alpha = 0.3)
plt.errorbar(group_r['MJD'], np.array(find_mag(group_r['Apparent Magnitude'])),yerr = find_magerr(group_r['Apparent Magnitude']), c='r', fmt = 'None', alpha = 0.3)
plt.errorbar(group_i['MJD'], np.array(find_mag(group_i['Apparent Magnitude']))+2,yerr = find_magerr(group_i['Apparent Magnitude']), c='y', fmt = 'None', alpha = 0.3)
plt.errorbar(group_c['MJD'], np.array(find_mag(group_c['Apparent Magnitude']))+1,yerr = find_magerr(group_c['Apparent Magnitude']), c='k', fmt = 'None', alpha = 0.3)
plt.scatter(group_g['MJD'], np.array(find_mag(group_g['Apparent Magnitude']))-1, s = 30, c = 'g', marker = 'D', alpha = 0.3,label = 'g-1 [Hosseinzadeh + 2022]')
plt.scatter(group_r['MJD'], np.array(find_mag(group_r['Apparent Magnitude'])), s = 30, c = 'r', marker = 'D', alpha = 0.3,label = 'r [Hosseinzadeh + 2022]')
plt.scatter(group_i['MJD'], np.array(find_mag(group_i['Apparent Magnitude']))+2, s = 30, c = 'y', marker = 'D', alpha = 0.3,label = 'i+2 [Hosseinzadeh + 2022]')
plt.scatter(group_c['MJD'], np.array(find_mag(group_c['Apparent Magnitude']))+1, s = 30, c = 'k', marker = 'D', alpha = 0.3,label = 'Clear+1 [Hosseinzadeh + 2022]')
plt.errorbar(group_g_my['obsdate'], np.array(group_g_my['mag']-1),yerr = group_g_my['e_mag'], c='g',  fmt = 'none')
plt.errorbar(group_r_my['obsdate'], np.array(group_r_my['mag']),yerr = group_r_my['e_mag'], c='r', fmt = 'none')
plt.errorbar(group_i_my['obsdate'], np.array(group_i_my['mag']+2),yerr = group_i_my['e_mag'], c='k', fmt = 'none')
plt.scatter(group_g_my['obsdate'], np.array(group_g_my['mag'])-1, s = 30, facecolors = 'none', edgecolor = 'g', marker = 's',label = 'g-1 [IMSNG]')
plt.scatter(group_r_my['obsdate'], np.array(group_r_my['mag']), s = 30,facecolors = 'none', edgecolor = 'r', marker = 's',label = 'r [IMSNG]')
plt.scatter(group_i_my['obsdate'], np.array(group_i_my['mag'])+2, s = 30,facecolors = 'none', edgecolor = 'k', marker = 's',label = 'i+2 [IMSNG]')

plt.grid()
plt.xlim(59525,59540)
plt.ylim(19,11)
plt.xlabel('MJD')
plt.ylabel('Apparent magnitude')
plt.legend()
plt.show()
#%%
