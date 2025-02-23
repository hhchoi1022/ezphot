#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:35:53 2022

@author: hhchoi1022
"""

from astropy.io import ascii
from astropy.table import Table
from astropy.table import join
import matplotlib.pyplot as plt
from matplotlib import gridspec
from astropy.table import vstack
import os, glob, sys
from astropy.table.pprint import conf
conf.max_lines = -1
conf.max_width = -1



tbl_path ='/data2/SN2021aefx/Data/'
tbllist = glob.glob(f'{tbl_path}*.dat')
all_tbl = Table()
for tblname in tbllist:
    tbl = ascii.read(tblname)
    all_tbl = vstack([all_tbl, tbl])
    
all_tbl['index'] = list(range(len(all_tbl)))
all_tbl.add_index('index')
all_tbl['Limit'] = 0
all_tbl['mode'] = 'MERGE'
for i in range(len(all_tbl)):
    if all_tbl['UL5_4'][i] == all_tbl['mag'][i]:
        all_tbl['Limit'][i] = 1
    else:
        all_tbl['Limit'][i] = 0
for i in range(len(all_tbl)):
    if 'HIGH' in all_tbl['image'][i]:
        all_tbl['mode'][i] = 'HIGH'
    else:
        all_tbl['mode'][i] = 'MERGE'

all_tbl['int_obsdate'] = all_tbl['obsdate'].astype(int)
all_tbl.add_index

def show_filter(tbl, filter_):
    index = (tbl['filter'] == filter_)
    table = tbl[index]
    plt.scatter(table['obsdate'], table['mag'], label = filter_)
    return table

def show_obs(tbl, observatory):
    index = (tbl['observatory'] == observatory)
    table = tbl[index]
    plt.scatter(table['obsdate'], table['mag'], label = observatory)
    return table
    
def show_obsfilt(tbl, observatory, filter_):
    index = (tbl['observatory'] == observatory) & (tbl['filter'] == filter_)
    table = tbl[index]
    plt.scatter(table['obsdate'], table['mag'], label = f'[{observatory}]{filter_}')
    return table

def match_obsdate(tbl1, tbl2, timeterm):
    tbl3 = join(tbl1,tbl2,'int_obsdate')
    #tbl3[]
    

obslist = set(all_tbl['observatory'])
filterlist = set(all_tbl['filter'])

det_tbl = all_tbl[all_tbl['Limit'] == 0]
nondet_tbl = all_tbl[all_tbl['Limit'] == 1]


obsgroup = det_tbl.group_by('observatory')
KCT_tbl = obsgroup.groups[0]
LSGT_tbl = obsgroup.groups[1]
RASA_tbl = obsgroup.groups[2]
noobsgroup = nondet_tbl.group_by('observatory')
noKCT_tbl = noobsgroup.groups[0]
noLSGT_tbl = noobsgroup.groups[1]
noRASA_tbl = noobsgroup.groups[2]
match_obsdate(KCT_tbl, RASA_tbl, 0.3)

KCT_r = show_filter(KCT_tbl, 'r')
KCT_g = show_filter(KCT_tbl, 'g')
KCT_i = show_filter(KCT_tbl, 'i')
RASA_r = show_filter(RASA_tbl, 'r')
LSGT_r = show_filter(LSGT_tbl, 'r')
LSGT_g = show_filter(LSGT_tbl, 'g')
LSGT_i = show_filter(LSGT_tbl, 'i')
noKCT_r = show_filter(noKCT_tbl, 'r')
noKCT_g = show_filter(noKCT_tbl, 'g')
noKCT_i = show_filter(noKCT_tbl, 'i')
noRASA_r = show_filter(noRASA_tbl, 'r')
noLSGT_r = show_filter(noLSGT_tbl, 'r')
noLSGT_g = show_filter(noLSGT_tbl, 'g')
noLSGT_i = show_filter(noLSGT_tbl, 'i')

len(RASA_r)
noRASA_r
peak = 59548.06

#%%
os.chdir('/data2/SN2021aefx/')
all_tbl.write('all_tbl.dat',format = 'ascii')


#%% Comparison between observatories

from astropy import table
from astropy.table import join_distance
KCT_gr = join(KCT_g, KCT_r, keys = 'int_obsdate')
KCT_gr['g-r'] = KCT_gr['mag_1'] - KCT_gr['mag_2'] 
t1 = Table([list(KCT_gr['obsdate_2'])], names = ['obsdate'])
t2 = Table([list(RASA_tbl['obsdate'])], names = ['obsdate'])
t12 = table.join(t1, t2, join_funcs = {'obsdate':join_distance(0.1)})
KCT_gr['obsdate_1'] = KCT_gr['obsdate_2']
RASA_tbl['obsdate_2'] = RASA_tbl['obsdate']
tbl_1 = join(left = t12, right = KCT_gr, keys = 'obsdate_1')
tbl_2 = join(left = t12, right = RASA_tbl, keys = 'obsdate_2')
plt.scatter(tbl_1['g-r'], tbl_2['mag']-tbl_1['mag_2'], c = tbl_1['obsdate_1']-tbl_2['obsdate_2'])
plt.grid()
plt.colorbar()



#%%
# KCT_gr color
KCT_gr = join(KCT_g, KCT_r, keys = 'int_obsdate')
KCT_gr['g-r'] = KCT_gr['mag_1'] - KCT_gr['mag_2'] 
plt.scatter(KCT_gr['int_obsdate'], KCT_gr['g-r'])
plt.show()
# LSGT_gr color
LSGT_gr = join(LSGT_g, LSGT_r, keys = 'int_obsdate')
LSGT_gr['g-r'] = LSGT_gr['mag_1'] - LSGT_gr['mag_2']
plt.scatter(LSGT_gr['int_obsdate'], LSGT_gr['g-r']) 
plt.show()
# KCT_ri color
KCT_ri = join(KCT_r, KCT_i, keys = 'int_obsdate')
KCT_ri['r-i'] = KCT_ri['mag_1'] - KCT_ri['mag_2'] 
plt.scatter(KCT_ri['int_obsdate'], KCT_ri['r-i'])
plt.show()
# LSGT_ri color
LSGT_ri = join(LSGT_r, LSGT_i, keys = 'int_obsdate')
LSGT_ri['r-i'] = LSGT_ri['mag_1'] - LSGT_ri['mag_2']
plt.scatter(LSGT_ri['int_obsdate'], LSGT_ri['r-i']) 
plt.show()

#%%
# RASA_KCT mag comparison
plt.figure(figsize = (6,4), dpi = 300)
RK_r = join(RASA_r, KCT_gr, keys = 'int_obsdate')
RK_r['RASA-KCT'] = RK_r['mag'] - RK_r['mag_2'] 
RK_r['delobsdate'] = RK_r['obsdate']-RK_r['obsdate_1']
plt.xlabel(r'$g-r_{KCT}$')
plt.ylabel(r'$\bigtriangleup r_{RASA36-KCT}$')
plt.scatter(RK_r['g-r'], RK_r['RASA-KCT'], c = RK_r['delobsdate'])#RK_r['int_obsdate']-peak)
plt.grid()
cbar = plt.colorbar()
cbar.ax.set_title('Phase')
plt.show()

# RASA_KCT obsdate comparison
plt.figure(figsize = (6,4), dpi = 300)
RK_r = join(RASA_r, KCT_gr, keys = 'int_obsdate')
RK_r['RASA-KCT'] = RK_r['mag'] - RK_r['mag_2'] 
RK_r['delobsdate'] = RK_r['obsdate']-RK_r['obsdate_1']
plt.xlabel(r'$Phase$(days)')
plt.ylabel(r'$ \bigtriangleup t_{obs, RASA36-KCT}(days)$')
plt.scatter(RK_r['int_obsdate']-peak, RK_r['delobsdate'], c= 'k')
plt.grid()
plt.show()
#%%
def spline(tbl, smooth_factor = 0.01):
    from scipy.interpolate import UnivariateSpline
    import numpy as np
    tbl.sort('obsdate')
    x = [tbl['obsdate'][k]-59528 for k in range(len(tbl))]
    y = [tbl['mag'][k] for k in range(len(tbl))]
    spl = UnivariateSpline(x, y)
    spl.set_smoothing_factor(smooth_factor)
    yp = spl(x)
    xs = np.linspace(0, x[-1]+10, 1000)
    
    plt.figure(figsize = (8,5), dpi = 300)
    
    plt.gca().invert_yaxis()
    gs = gridspec.GridSpec(nrows = 2, ncols =1, height_ratios= [8, 3], width_ratios = [8])
    ax0 = plt.subplot(gs[0])
    ax0.set_title('SN2021aefx, KCT[g]')
    ax0.set_ylabel('Apparent magnitude')
    ax0.plot(xs, spl(xs), 'k', lw=1, linestyle = '--')
    ax0.scatter(x,y, c='k')
    ax1 = plt.subplot(gs[1])
    ax1.scatter(x, y-yp, c='k')
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('residual')
    plt.show()
    
    return spl

#%%
KCT_g_spl = spline(KCT_g, 0.04)
KCT_r_spl = spline(KCT_r, 0.2)
RASA_r_spl = spline(RASA_r, 0.08)
LSGT_g_spl = spline(LSGT_g)
LSGT_r_spl = spline(LSGT_r, 0.08)

#%% LSGT_KCT mag comparison

LK_g = join(LSGT_g, KCT_gr, keys = 'int_obsdate')
LK_r = join(LSGT_r, KCT_gr, keys = 'int_obsdate')
LK_g['obsdate'] = LK_g['obsdate']- 59528
LK_g['obsdate_1'] = LK_g['obsdate_1']- 59528
LK_g['obsdate_2'] = LK_g['obsdate_2']- 59528
LK_g['mag_1'] = KCT_g_spl(LK_g['obsdate'])
LK_g['LSGT-KCT'] = LK_g['mag'] - LK_g['mag_1'] 

LK_r['obsdate'] = LK_r['obsdate']- 59528
LK_r['obsdate_1'] = LK_r['obsdate_1']- 59528
LK_r['obsdate_2'] = LK_r['obsdate_2']- 59528
LK_r['mag_2'] = KCT_r_spl(LK_r['obsdate'])
LK_r['LSGT-KCT'] = LK_r['mag'] - LK_r['mag_2'] 

#%%
plt.figure(figsize = (6,4), dpi = 300)
plt.xlabel(r'$g-r_{KCT}$')
plt.ylabel(r'$\bigtriangleup r_{LSGT-KCT}$')
plt.scatter(LK_r['g-r'], LK_r['LSGT-KCT'], c = LK_r['obsdate_2']-peak+59528)
plt.grid()
cbar = plt.colorbar()
cbar.ax.set_title('Phase')
plt.show()
#%%
plt.figure(figsize = (6,4), dpi = 300)
plt.xlabel(r'$g-r_{KCT}$')
plt.ylabel(r'$\bigtriangleup g_{LSGT-KCT}$')
plt.scatter(LK_g['g-r'], LK_g['LSGT-KCT'], c = LK_g['obsdate_1']-peak+59528)
plt.grid()
cbar = plt.colorbar()
cbar.ax.set_title('Phase')
plt.show()


#%% LSGT_KCT mag comparison

RK_r = join(RASA_r, KCT_gr, keys = 'int_obsdate')

RK_r['obsdate'] = RK_r['obsdate']- 59528
RK_r['obsdate_1'] = RK_r['obsdate_1']- 59528
RK_r['obsdate_2'] = RK_r['obsdate_2']- 59528
RK_r['mag_2'] = KCT_r_spl(RK_r['obsdate'])
RK_r['RASA-KCT'] = RK_r['mag'] - RK_r['mag_2'] 

#%%
plt.figure(figsize = (6,4), dpi = 300)
plt.xlabel(r'$Phase$')
plt.ylabel(r'$\bigtriangleup r_{RASA36-KCT}$')
plt.scatter(RK_r['obsdate_2']-peak+59528 , RK_r['RASA-KCT'], c = RK_r['g-r'])
plt.grid()
cbar = plt.colorbar()
cbar.ax.set_title('Color')
plt.show()


#%% All light curve

from matplotlib import gridspec
import numpy as np
exptime = 59528.135
markersize = 10

plt.figure(figsize = (6,6), dpi = 1000)
gs = gridspec.GridSpec(nrows = 3, ncols =1, height_ratios= [10, 3, 3], width_ratios = [6])
plt.xticks(visible=False)
plt.subplots_adjust(hspace=0)
ax0 = plt.subplot(gs[0])
plt.gca().invert_yaxis()

ax0.scatter(noKCT_r['obsdate']-exptime, noKCT_r['mag'], marker = 's',  alpha = 0.1, s = markersize, facecolor = 'none', edgecolor = 'r', linewidth = 0.8)
ax0.scatter(noRASA_r['obsdate']-exptime, noRASA_r['mag'], marker = 'x', alpha = 0.1, s = markersize, facecolor = 'none', edgecolor = 'r', linewidth = 0.8)
ax0.scatter(noKCT_g['obsdate']-exptime, noKCT_g['mag']-2, marker = 's',  alpha = 0.1, s = markersize, facecolor = 'none', edgecolor = 'g', linewidth = 0.8)
ax0.scatter(noLSGT_g['obsdate']-exptime, noLSGT_g['mag']-2, marker = '^', c ='g', alpha = 0.1, s = markersize, facecolor = 'none', edgecolor = 'g', linewidth = 0.8)
ax0.scatter(noKCT_i['obsdate']-exptime, noKCT_i['mag']+1, marker = 's', alpha = 0.1, s = markersize, facecolor = 'none', edgecolor = 'y', linewidth = 0.8)
ax0.scatter(noLSGT_i['obsdate']-exptime, noLSGT_i['mag']+1, marker = '^', c ='y', alpha = 0.1, s = markersize, facecolor = 'none', edgecolor = 'y', linewidth = 0.8)

ax0.scatter(KCT_r['obsdate']-exptime, KCT_r['mag'], marker = 's',label = '[KCT]r band', s = markersize, facecolor = 'none', edgecolor = 'r', linewidth = 0.8)
ax0.scatter(RASA_r['obsdate']-exptime, RASA_r['mag'], marker = 'x',  c ='r', label = '[RASA36]r band', s = markersize, facecolor = 'none', edgecolor = 'r', linewidth = 0.8)
ax0.scatter(KCT_g['obsdate']-exptime, KCT_g['mag']-2, marker = 's',  label = '[KCT]g band', s = markersize, facecolor = 'none', edgecolor = 'g', linewidth = 0.8)
ax0.scatter(LSGT_g['obsdate']-exptime, LSGT_g['mag']-2, marker = '^',  label = '[LSGT]g band', s = markersize, facecolor = 'none', edgecolor = 'g', linewidth = 0.8)
ax0.scatter(KCT_i['obsdate']-exptime, KCT_i['mag']+1, marker = 's',  label = '[KCT]i band', s = markersize, facecolor = 'none', edgecolor = 'y', linewidth = 0.8)
ax0.scatter(LSGT_i['obsdate']-exptime, LSGT_i['mag']+1, marker = '^', label = '[LSGT]i band', s = markersize, facecolor = 'none', edgecolor = 'y', linewidth = 0.8)

import matplotlib.patches as mpatches
marker = ['s', '^', 'x']
color = ['g','r','y']
label_column = ['KCT','LSGT','RASA36']
label_row = ['g-2','r','i+1']
rows = [mpatches.Patch(color=color[i][0]) for i in range(3)]
columns = [plt.plot([], [], marker[i], markerfacecolor='w',markeredgecolor='k')[0] for i in range(3)]
ax0.legend(rows + columns, label_row + label_column, loc=1)

ax0.grid()

ax0.set_ylabel(' Apparent magnitude[AB]')
ax0.set_title(r'$SN\ 2021aefx$')


#%%









































































'''
g1_tbl = ascii.read('/data2/temp/SN/NGC1566/KCT_STX16803/g/KCT_g.dat')
r1_tbl = ascii.read('/data2/temp/SN/NGC1566/KCT_STX16803/r/KCT_r.dat')
i1_tbl = ascii.read('/data2/temp/SN/NGC1566/KCT_STX16803/i/KCT_i.dat')
g2_tbl = ascii.read('/data2/temp/SN/NGC1566/LSGT/g/LSGT_g.dat')
r2_tbl = ascii.read('/data2/temp/SN/NGC1566/LSGT/r/LSGT_r.dat')
i2_tbl = ascii.read('/data2/temp/SN/NGC1566/LSGT/i/LSGT_i.dat')
r3_tbl = ascii.read('/data2/temp/SN/NGC1566/RASA36/r/HIGH/RASA36_r.dat')
r4_tbl = ascii.read('/data2/temp/SN/NGC1566/RASA36/r/NOINFO/MERGE/RASA36_r.dat')
r5_tbl = ascii.read('/data2/temp/SN/NGC1566/RASA36/r/NOINFO/HIGH/RASA36_r.dat')

all_tbl = vstack([g1_tbl, r1_tbl, i1_tbl, g2_tbl, r2_tbl, i2_tbl, r3_tbl, r4_tbl, r5_tbl])
all_tbl[all_tbl['obsdate']<59595].write('fit_tbl.dat', format = 'ascii', overwrite = True)
depth_tbl = all_tbl[all_tbl['UL5_4'] == all_tbl['mag']]
all_tbl = all_tbl[all_tbl['UL5_4'] != all_tbl['mag']]

filtergroup = all_tbl.group_by('filter')
filtergroup_depth = depth_tbl.group_by('filter')
filtergroup['int_obsdate'] = filtergroup['obsdate'].astype(int)
filtergroup_depth['int_obsdate'] = filtergroup_depth['obsdate'].astype(int)

g_group = filtergroup[filtergroup['filter'] == 'g']
r_group = filtergroup[filtergroup['filter'] == 'r']
i_group = filtergroup[filtergroup['filter'] == 'i']
g_group_depth = filtergroup_depth[filtergroup_depth['filter'] == 'g']
r_group_depth = filtergroup_depth[filtergroup_depth['filter'] == 'r']
i_group_depth = filtergroup_depth[filtergroup_depth['filter'] == 'i']
color_tbl = join(g_group, r_group, keys = ['int_obsdate', 'observatory'])
color_tbl2 = join(r_group, i_group, keys = ['int_obsdate', 'observatory'])
color_tbl['mag_2']
color_tbl['g-r'] = color_tbl['mag_1'] - color_tbl['mag_2']
color_tbl2 = join(r_group, i_group, keys = ['int_obsdate', 'observatory'])
color_tbl2['r-i'] = color_tbl2['mag_1'] - color_tbl2['mag_2']
rasa_group = filtergroup[filtergroup['observatory'] == 'RASA36']
kct_group = filtergroup[filtergroup['observatory'] == 'KCT']
diff_tbl = join(rasa_group, kct_group, keys = 'int_obsdate')
diff_tbl['delmag'] = diff_tbl['mag_1'] - diff_tbl['mag_2']



#%% LC and color 
c = dict(g='g', r = 'r', i = 'y')
shape = dict(KCT = 'o', LSGT = 's', RASA36 = 'D')
plt.figure(figsize = (9,12), dpi = 1000)
plt.title('SN2021aefx')
plt.grid()
plt.xlim(-30,90)
plt.gca().invert_yaxis()
peak = 59548.012
fir = 59528.07
gs = gridspec.GridSpec(nrows = 3, ncols =1, height_ratios= [8, 3, 3], width_ratios = [8])
plt.xticks(visible=False)
plt.subplots_adjust(hspace=0)
ax0 = plt.subplot(gs[0])
ax0.set_title('SN2021aefx')
g_obsgroup_depth = g_group_depth.group_by('observatory')
for obsname, group in zip(g_obsgroup_depth.groups.keys, g_obsgroup_depth.groups):
    ax0.errorbar(group['obsdate']-peak, group['mag']-1 ,yerr = group['e_mag'] , fmt = '.', capsize = 4, c = c['g'], marker = shape[obsname['observatory']], alpha = 0.2)
r_obsgroup_depth = r_group_depth.group_by('observatory')
for obsname, group in zip(r_obsgroup_depth.groups.keys, r_obsgroup_depth.groups):
    ax0.errorbar(group['obsdate']-peak, group['mag']+1 ,yerr = group['e_mag'] , fmt = '.', capsize = 4, c = c['r'], marker = shape[obsname['observatory']], alpha = 0.2)
g_obsgroup = g_group.group_by('observatory')
for obsname, group in zip(g_obsgroup.groups.keys, g_obsgroup.groups):
    ax0.errorbar(group['obsdate']-peak, group['mag']-1 ,yerr = group['e_mag'] , fmt = '.', capsize = 4, c = c['g'], marker = shape[obsname['observatory']], label = f'{obsname["observatory"]}[g-1]]', alpha = 0.8)
r_obsgroup = r_group.group_by('observatory')
for obsname, group in zip(r_obsgroup.groups.keys, r_obsgroup.groups):
    ax0.errorbar(group['obsdate']-peak, group['mag']+1 ,yerr = group['e_mag'] , fmt = '.', capsize = 4, c = c['r'], marker = shape[obsname['observatory']], label = f'{obsname["observatory"]}[r+1]]', alpha = 0.8)
i_obsgroup = i_group.group_by('observatory')
for obsname, group in zip(i_obsgroup.groups.keys, i_obsgroup.groups):
    ax0.errorbar(group['obsdate']-peak, group['mag'] ,yerr = group['e_mag'] , fmt = '.', capsize = 4, c = c['i'], marker = shape[obsname['observatory']], label = f'{obsname["observatory"]}[i]]', alpha = 0.8)
ax0.legend()
ax0.grid()
ax0.set_ylabel('Apparent magnitude (AB)')
ax0.set_xlim(-30,90)
ax0.set_ylim(22,10)
import numpy as np
ax1 = plt.subplot(gs[1], sharex = ax0)
ax1.set_xlabel('days from B maximum')
ax1.set_ylabel('g-r (AB)')
ax1.errorbar(color_tbl['int_obsdate']-peak, color_tbl['g-r'], yerr = np.sqrt(color_tbl['e_mag_1']**2+color_tbl['e_mag_2']**2), fmt = '.', capsize = 5,  label = 'g-r', marker = '.', c = 'k')
ax1.grid()


plt.show()
#%%
color1_tbl = color_tbl[color_tbl['observatory'] == 'KCT']
color2_tbl = color_tbl[color_tbl['observatory'] == 'LSGT']

plt.figure(figsize = (6,5), dpi = 500)
plt.ylabel(r'$g-r\ (mag)$')
plt.xlabel('days since first light')
plt.xlim(0,20)
plt.ylim(-0.8,1.2)
plt.errorbar(color1_tbl['obsdate_1']-peak+19.93, color1_tbl['g-r'], yerr = np.sqrt(color1_tbl['e_mag_1']**2+color1_tbl['e_mag_2']**2), fmt = '.', capsize = 5, label = 'KCT', marker = 'o', c = 'k')
plt.errorbar(color2_tbl['obsdate_1']-peak+19.93, color2_tbl['g-r'], yerr = np.sqrt(color2_tbl['e_mag_1']**2+color2_tbl['e_mag_2']**2), fmt = '.', capsize = 5, label = 'LSGT', marker = 'D', c = 'k')
plt.legend()
plt.grid()
color_tbl['observatory']
color_tbl['g-r']
color1_tbl['obsdate_1'][:7]
(color1_tbl[6]['g-r']-color1_tbl[0]['g-r'])/(color1_tbl[6]['obsdate_1']-color1_tbl[0]['obsdate_1'])
#%%Observatory offset
rasa_group = filtergroup[filtergroup['observatory'] == 'RASA36']
kct_group = filtergroup[(filtergroup['observatory'] == 'KCT')&(filtergroup['filter'] == 'r')]
offset_tbl = join(rasa_group, kct_group, keys = ['int_obsdate'])
offset_tbl['offset'] = offset_tbl['mag_1'] - offset_tbl['mag_2']

from matplotlib import gridspec
plt.figure(figsize = (6,4), dpi = 1000)

gs = gridspec.GridSpec(nrows = 2, ncols =1, height_ratios= [10, 3], width_ratios = [6])
plt.xticks(visible=False)
plt.subplots_adjust(hspace=0)
ax0 = plt.subplot(gs[0])
ax0.errorbar(offset_tbl['mag_1'], offset_tbl['mag_2'], xerr = offset_tbl['e_mag_1'],yerr = offset_tbl['e_mag_2'], fmt = '.', c ='k', label = 'r band')
ax0.grid()
ax0.legend()
ax0.set_ylabel(' Apparent magnitude (KCT)')
ax0.set_title('SN2021aefx')
ax1 = plt.subplot(gs[1], sharex = ax0)
ax1.scatter(offset_tbl['mag_1'], offset_tbl['offset'], c= 'k', marker = 'o', s = 5)
ax1.errorbar(offset_tbl['mag_1'], offset_tbl['offset'], yerr = np.sqrt(offset_tbl['e_mag_1']**2+offset_tbl['e_mag_2']**2), c= 'k', fmt = 'None', alpha = 0.3)
ax1.set_xlabel('Apparent magnitude (RASA36)')
ax1.set_ylabel('Residual')
ax1.set_ylim(-0.2,0.2)
ax1.grid()

#%%Observatory offset with phase
rasa_group = filtergroup[filtergroup['observatory'] == 'RASA36']
kct_group = filtergroup[(filtergroup['observatory'] == 'KCT')&(filtergroup['filter'] == 'r')]
kct_group_g = filtergroup[(filtergroup['observatory'] == 'KCT')&(filtergroup['filter'] == 'r')]
offset_tbl = join(rasa_group, kct_group, keys = ['int_obsdate'])
offset_tbl1 = join(offset_tbl, kct_group_g , keys = ['int_obsdate'])
offset_tbl1['offset'] = offset_tbl1['mag_1'] - offset_tbl1['mag_2']
offset_tbl1['g-r'] = offset_tbl1['mag'] - offset_tbl1['mag_2']

from matplotlib import gridspec
plt.figure(figsize = (6,4), dpi = 1000)

gs = gridspec.GridSpec(nrows = 2, ncols =1, height_ratios= [10, 3], width_ratios = [6])
plt.xticks(visible=False)
plt.subplots_adjust(hspace=0)
ax0 = plt.subplot(gs[0])
ax0.errorbar(offset_tbl1['int_obsdate'], offset_tbl1['mag_1'],yerr = offset_tbl1['e_mag_1'], fmt = '.', c ='r', label = 'RASA(r)', alpha = 0.3)
ax0.errorbar(offset_tbl1['int_obsdate'], offset_tbl1['mag_2'],yerr = offset_tbl1['e_mag_2'], fmt = '.', c ='r', label = 'KCT(r)')

ax0.grid()
ax0.legend()
ax0.set_ylabel(' Apparent magnitude (KCT)')
ax0.set_title('SN2021aefx')
ax1 = plt.subplot(gs[1], sharex = ax0)
ax1.scatter(offset_tbl1['int_obsdate'], offset_tbl1['offset'], c= 'k', marker = 'o', s = 5)
ax1.errorbar(offset_tbl1['int_obsdate'], offset_tbl1['offset'], yerr = np.sqrt(offset_tbl1['e_mag_1']**2+offset_tbl1['e_mag_2']**2), c= 'k', fmt = 'None', alpha = 0.3)
ax1.set_xlabel('Apparent magnitude (RASA36)')
ax1.set_ylabel('Residual')
ax1.set_ylim(-0.2,0.2)
ax1.grid()

#%%Observatory offset with color
rasa_group = filtergroup[filtergroup['observatory'] == 'RASA36']
kct_group = filtergroup[(filtergroup['observatory'] == 'KCT')&(filtergroup['filter'] == 'r')]
kct_group_g = filtergroup[(filtergroup['observatory'] == 'KCT')&(filtergroup['filter'] == 'g')]
offset_tbl = join(rasa_group, kct_group, keys = ['int_obsdate'])
offset_tbl1 = join(offset_tbl, kct_group_g , keys = ['int_obsdate'])
offset_tbl1['offset'] = offset_tbl1['mag_1'] - offset_tbl1['mag_2']
offset_tbl1['g-r'] = offset_tbl1['mag'] - offset_tbl1['mag_2']
kct_group_g['obsdate']
kct_group['obsdate']
offset_tbl1['int_obsdate']
from scipy.optimize import curve_fit
def conversion(x,a,b):
    diff = a* x + b
    return diff
popt_, pcov_ = curve_fit(conversion, offset_tbl1['g-r'], offset_tbl1['offset'], sigma = np.sqrt(offset_tbl1['e_mag']**2+offset_tbl1['e_mag_2']**2))
x1 = np.linspace(-0.1,0.4,100)
from matplotlib import gridspec
plt.figure(figsize = (6,6), dpi = 1000)

gs = gridspec.GridSpec(nrows = 3, ncols =1, height_ratios= [10, 3, 3], width_ratios = [6])
plt.xticks(visible=False)
plt.subplots_adjust(hspace=0)
ax0 = plt.subplot(gs[0])
#ax0.errorbar(offset_tbl1['int_obsdate'], offset_tbl1['mag_1'],yerr = offset_tbl1['e_mag_1'], fmt = '.', c ='r', label = 'RASA(r)', alpha = 0.3)
#ax0.errorbar(offset_tbl1['int_obsdate'], offset_tbl1['mag_2'],yerr = offset_tbl1['e_mag_2'], fmt = '.', c ='r', label = 'KCT(r)')

# color vs offset
ax0.errorbar(offset_tbl1['g-r'], offset_tbl1['offset'],yerr = np.sqrt(offset_tbl1['e_mag']**2+offset_tbl1['e_mag_2']**2), fmt = '.', c ='r', alpha = 0.3)
ax0.plot(x1, conversion(x1, popt_[0], popt_[1]), label = 'r(RASA) = r(KCT) + %.2f(g-r) + %.2f' %(popt_[0],popt_[1]), linewidth  =1, c = 'k', linestyle = '--')
ax0.grid()
ax0.legend()
ax0.set_xlabel(' g-r (KCT)')
ax0.set_ylabel(' Offset (RASA-KCT)')
ax0.set_title('SN 2021aefx')
## color vs offset

ax1 = plt.subplot(gs[1], sharex = ax0)
ax1.scatter(offset_tbl1['int_obsdate'], offset_tbl1['offset'], c= 'k', marker = 'o', s = 5)
ax1.errorbar(offset_tbl1['int_obsdate'], offset_tbl1['offset'], yerr = np.sqrt(offset_tbl1['e_mag_1']**2+offset_tbl1['e_mag_2']**2), c= 'k', fmt = 'None', alpha = 0.3)
ax1.set_xlabel('Apparent magnitude (RASA36)')
ax1.set_ylabel('Residual')
ax1.set_ylim(-0.2,0.2)
ax1.grid()
ax2 = plt.subplot(gs[2], sharex = ax0)
ax2.scatter(offset_tbl1['int_obsdate'], offset_tbl1['g-r'], c= 'k', marker = 'o', s = 5)
ax2.errorbar(offset_tbl1['int_obsdate'], offset_tbl1['g-r'], yerr = np.sqrt(offset_tbl1['e_mag_1']**2+offset_tbl1['e_mag_2']**2), c= 'k', fmt = 'None', alpha = 0.3)
ax2.set_xlabel('Apparent magnitude (RASA36)')
ax2.set_ylabel('g-r')
ax2.set_ylim(-0.5,0.5)
ax2.grid()



#%%Observatory offset by Paek
file = '/home/hhchoi1022/Downloads/all.phot.dat'
from astropy.time import Time
paek_tbl = ascii.read(file)
paek_tbl['mjd'] = Time(paek_tbl['jd'], format = 'jd').mjd
paek_tbl['int_obsdate'] = paek_tbl['mjd'].astype(int)
rasa_group = paek_tbl[(paek_tbl['obs'] == 'RASA36') & (paek_tbl['mag']>0)]
kct_group = paek_tbl[(paek_tbl['obs'] == 'KCT_STX16803')&(paek_tbl['filter'] == 'r')& (paek_tbl['mag']>0)]
offset_tbl = join(rasa_group, kct_group, keys = ['int_obsdate'])
offset_tbl['offset'] = offset_tbl['mag_1'] - offset_tbl['mag_2']
offset_tbl.columns
from matplotlib import gridspec
plt.figure(figsize = (6,4), dpi = 1000)

gs = gridspec.GridSpec(nrows = 2, ncols =1, height_ratios= [10, 3], width_ratios = [6])
plt.xticks(visible=False)
plt.subplots_adjust(hspace=0)
ax0 = plt.subplot(gs[0])
ax0.errorbar(offset_tbl['mag_1'], offset_tbl['mag_2'], xerr = offset_tbl['magerr_1'],yerr = offset_tbl['magerr_2'], fmt = '.', c ='k', label = 'r band')
ax0.grid()
ax0.legend()
ax0.set_ylabel(' Apparent magnitude (KCT)')
ax0.set_title('SN2021aefx by Paek')
ax1 = plt.subplot(gs[1], sharex = ax0)
ax1.scatter(offset_tbl['mag_1'], offset_tbl['offset'], c= 'k', marker = 'o', s = 5)
ax1.errorbar(offset_tbl['mag_1'], offset_tbl['offset'], yerr = np.sqrt(offset_tbl['magerr_1']**2+offset_tbl['magerr_2']**2), c= 'k', fmt = 'None', alpha = 0.3)
ax1.set_xlabel('Apparent magnitude (RASA36)')
ax1.set_ylabel('Residual')
ax1.set_ylim(-0.2,0.2)
ax1.grid()
#%% Comparison with Paek
file = '/home/hhchoi1022/Downloads/all.phot.dat'
from astropy.time import Time
from astropy.table import join
hh_tbl = vstack([g1_tbl, r1_tbl, i1_tbl, g2_tbl, r2_tbl, i2_tbl, r3_tbl, r4_tbl, r5_tbl])
hh_tbl = hh_tbl[hh_tbl['UL5_4'] != hh_tbl['mag']]
hh_tbl['int_obsdate'] = hh_tbl['obsdate'].astype(int)
paek_tbl = ascii.read(file)
obslist = []
for i in range(len(paek_tbl)):
    if paek_tbl['obs'][i] == 'KCT_STX16803':
        obs = 'KCT' 
    else: 
        obs = paek_tbl['obs'][i]
    obslist.append(obs)
    obslist
paek_tbl['observatory'] = obslist
paek_tbl['mjd'] = Time(paek_tbl['jd'], format = 'jd').mjd
paek_tbl['int_obsdate'] = paek_tbl['mjd'].astype(int)
diff_tbl = join(left = hh_tbl, right = paek_tbl, keys = ['int_obsdate', 'filter','observatory'])
diff_tbl['delmag'] = diff_tbl['mag_1'] - diff_tbl['mag_2']

from matplotlib import gridspec
plt.figure(figsize = (6,4), dpi = 1000)
gs = gridspec.GridSpec(nrows = 2, ncols =1, height_ratios= [10, 3], width_ratios = [6])
plt.xticks(visible=False)
plt.subplots_adjust(hspace=0)
ax0 = plt.subplot(gs[0])
ax0.errorbar(diff_tbl['mag_1'], diff_tbl['mag_2'], xerr = diff_tbl['e_mag'],yerr = diff_tbl['magerr'], fmt = '.', c ='k')
ax0.grid()
ax0.set_title('SN2021aefx')
ax0.set_ylabel('Apparent magnitude (Paek)')
ax1 = plt.subplot(gs[1], sharex = ax0)
ax1.scatter(diff_tbl['mag_1'], diff_tbl['delmag'], c= 'k', marker = 'o', s = 5)
ax1.errorbar(diff_tbl['mag_1'], diff_tbl['delmag'], yerr = np.sqrt(diff_tbl['e_mag']**2+diff_tbl['magerr']**2), c= 'k', fmt = 'None', alpha = 0.3)
ax1.set_xlabel('Apparent magnitude (HH)')
ax1.set_ylabel('Residual')

ax1.grid()

#%% Conversion with conversion table 
from astropy.table import Table

color_tbl['g-r'] =color_tbl['mag_1']-color_tbl['mag_2']
color_tbl['e_g-r'] = np.sqrt(color_tbl['e_mag_1']**2 + color_tbl['e_mag_2']**2)
color_tbl['mag_1'] + 0.313*color_tbl['g-r']+0.219
color_tbl['e_B_mag'] = np.sqrt(color_tbl['e_mag_1']**2 + (0.313*color_tbl['e_g-r'])**2)
color_tbl['V_mag'] = color_tbl['mag_1'] - 0.565*color_tbl['g-r']-0.016
color_tbl['e_V_mag'] = np.sqrt(color_tbl['e_mag_1']**2 + (0.565*color_tbl['e_g-r'])**2)
color_tbl['B-V_mag'] = color_tbl['B_mag'] - color_tbl['V_mag']
color_tbl['BQ_mag'] = color_tbl['mag_1'] + 0.1*color_tbl['g-r']+0.12
color_tbl['e_BQ_mag'] = np.sqrt(color_tbl['e_mag_1']**2 + (0.1*color_tbl['e_g-r'])**2)
color_tbl['VQ_mag'] = color_tbl['mag_1'] - 0.52*color_tbl['g-r']-0.03
color_tbl['e_VQ_mag'] = np.sqrt(color_tbl['e_mag_1']**2 + (0.52*color_tbl['e_g-r'])**2)
color_tbl['B-VQ_mag'] = color_tbl['BQ_mag'] - color_tbl['VQ_mag']
peakcolor = color_tbl[color_tbl['int_obsdate'] == int(peak)]

plt.figure(figsize = (4,4), dpi = 500)
plt.errorbar(color_tbl['obsdate_1']-peak,color_tbl['g-r']-peakcolor['g-r'], yerr = color_tbl['e_g-r'] , fmt = '.', linewidth = 1, c = 'k')
plt.xlim(-20,35)
plt.ylim(-0.3,1.2)
plt.xlabel('$t-T_{B,max}(d)$')
plt.ylabel(r'$(g-r)_0\ [mag]$')
plt.grid()


color_tbl2['r-i'] =color_tbl2['mag_1']-color_tbl2['mag_2']
color_tbl2['e_r-i'] = np.sqrt(color_tbl2['e_mag_1']**2 + color_tbl2['e_mag_2']**2)
color_tbl2['R_mag'] = color_tbl2['mag_1'] -0.153 *color_tbl2['r-i']+0.003
color_tbl2['e_R_mag'] = np.sqrt(color_tbl2['e_mag_1']**2 + (0.153*color_tbl2['e_r-i'])**2)
color_tbl2['RQ_mag'] = color_tbl2['mag_1'] + 0.38*color_tbl2['r-i']+0.27
color_tbl2['e_RQ_mag'] = np.sqrt(color_tbl2['e_mag_1']**2 + (0.38*color_tbl2['e_r-i'])**2)

peakcolor['B_mag']
c = dict(g='g', r = 'r', i = 'y')
shape = dict(KCT = 'o', LSGT = 's', RASA36 = 'D')
plt.figure(figsize = (10,14), dpi = 1000)
plt.title('SN2021aefx')
plt.grid()
plt.xlim(-30,90)
plt.gca().invert_yaxis()
peak = 59548.012
gs = gridspec.GridSpec(nrows = 3, ncols =1, height_ratios= [8, 3, 3], width_ratios = [8])
plt.xticks(visible=False)
plt.subplots_adjust(hspace=0)
ax0 = plt.subplot(gs[0])
ax0.set_title('SN2021aefx')
g_obsgroup_depth = g_group_depth.group_by('observatory')
for obsname, group in zip(g_obsgroup_depth.groups.keys, g_obsgroup_depth.groups):
    ax0.errorbar(group['obsdate']-peak, group['mag']-1 ,yerr = group['e_mag'] , fmt = '.', capsize = 4, c = c['g'], marker = shape[obsname['observatory']], alpha = 0.2)
r_obsgroup_depth = r_group_depth.group_by('observatory')
for obsname, group in zip(r_obsgroup_depth.groups.keys, r_obsgroup_depth.groups):
    ax0.errorbar(group['obsdate']-peak, group['mag']+1 ,yerr = group['e_mag'] , fmt = '.', capsize = 4, c = c['r'], marker = shape[obsname['observatory']], alpha = 0.2)
g_obsgroup = g_group.group_by('observatory')
for obsname, group in zip(g_obsgroup.groups.keys, g_obsgroup.groups):
    ax0.errorbar(group['obsdate']-peak, group['mag']-1 ,yerr = group['e_mag'] , fmt = '.', capsize = 4, c = c['g'], marker = shape[obsname['observatory']], label = f'{obsname["observatory"]}[g-1]]', alpha = 0.8)
r_obsgroup = r_group.group_by('observatory')
for obsname, group in zip(r_obsgroup.groups.keys, r_obsgroup.groups):
    ax0.errorbar(group['obsdate']-peak, group['mag']+1 ,yerr = group['e_mag'] , fmt = '.', capsize = 4, c = c['r'], marker = shape[obsname['observatory']], label = f'{obsname["observatory"]}[r+1]]', alpha = 0.8)
#i_obsgroup = i_group.group_by('observatory')
#for obsname, group in zip(i_obsgroup.groups.keys, i_obsgroup.groups):
#    ax0.errorbar(group['obsdate']-peak, group['mag'] ,yerr = group['e_mag'] , fmt = '.', capsize = 4, c = c['i'], marker = shape[obsname['observatory']], label = f'{obsname["observatory"]}[i]]', alpha = 0.8)

ax0.errorbar(color_tbl['obsdate_1']-peak, color_tbl['B_mag']-1, yerr = color_tbl['e_B_mag'], fmt = '.', capsize = 4, c = c['g'], marker = 'x', label = 'KCT, LSGT[B-1]', alpha = 0.2)
ax0.errorbar(color_tbl['obsdate_1']-peak, color_tbl['V_mag']+1, yerr = color_tbl['e_V_mag'], fmt = '.', capsize = 4, c = c['r'], marker = 'x', label = 'KCT, LSGT[V-1]', alpha = 0.2)
#ax0.errorbar(color_tbl['obsdate_1']-peak, color_tbl['BQ_mag']-1, yerr = color_tbl['e_BQ_mag'], fmt = '.', capsize = 4, c = c['g'], marker = 'x', label = 'KCT, LSGT[B-1, QSO]', alpha = 0.5)
#ax0.errorbar(color_tbl['obsdate_1']-peak, color_tbl['VQ_mag']+1, yerr = color_tbl['e_VQ_mag'], fmt = '.', capsize = 4, c = c['r'], marker = 'x', label = 'KCT, LSGT[V+1, QSO]', alpha = 0.5)

ax0.legend()
ax0.grid()
ax0.set_ylabel('Apparent magnitude (AB)')
ax0.set_xlim(-30,90)
ax0.set_ylim(22,10)

ax1 = plt.subplot(gs[1], sharex = ax0)

ax1.set_ylabel('Color (AB)')
ax1.errorbar(color_tbl['int_obsdate']-peak, color_tbl['g-r'], yerr = np.sqrt(color_tbl['e_mag_1']**2+color_tbl['e_mag_2']**2), fmt = '.', capsize = 5,  label = 'g-r', marker = '.', c = 'k')
ax1.errorbar(color_tbl['int_obsdate']-peak, color_tbl['B-V_mag'], yerr = np.sqrt(color_tbl['e_B_mag']**2+color_tbl['e_V_mag']**2), fmt = '.', capsize = 5,  label = 'B-V', marker = '.', c = 'g', alpha = 0.2)
ax1.errorbar(color_tbl['int_obsdate']-peak, color_tbl['B-VQ_mag'], yerr = np.sqrt(color_tbl['e_BQ_mag']**2+color_tbl['e_VQ_mag']**2), fmt = '.', capsize = 5,  label = 'B-V[QSO]', marker = '.', c = 'r', alpha = 0.5)
ax1.legend()
ax1.grid()

ax2 = plt.subplot(gs[1], sharex = ax0)
ax2.set_ylabel('Difference')
ax2.set_xlabel('days from B maximum')
ax2.errorbar(color_tbl['int_obsdate']-peak, (color_tbl['B_mag']+color_tbl['V_mag'])/2 - color_tbl['mag_1'],fmt = '.', capsize = 5,  label = '(B+V)/2-g', marker = 's', c = 'g', alpha =0.5)
ax2.errorbar(color_tbl['int_obsdate']-peak, color_tbl['B_mag'] - color_tbl['mag_1'],fmt = '.', capsize = 5,  label = 'B-g', marker = 'D', c = 'g', alpha =0.5)
ax2.errorbar(color_tbl['int_obsdate']-peak, color_tbl['V_mag'] - color_tbl['mag_2'],fmt = '.', capsize = 5,  label = 'V-r', marker = 'x', c = 'r', alpha =0.5)
ax2.errorbar(color_tbl2['int_obsdate']-peak, color_tbl2['R_mag'] - color_tbl2['mag_1'],fmt = '.', capsize = 5,  label = 'R-r', marker = 'o', c = 'r', alpha =0.5)
#ax2.errorbar(color_tbl['int_obsdate']-peak, (color_tbl['BQ_mag']+color_tbl['VQ_mag'])/2 - color_tbl['mag_1'],fmt = '.', capsize = 5,  label = '(B+V)/2-g [QSO]', marker = 's', c = 'r', alpha =0.5)
#ax2.errorbar(color_tbl['int_obsdate']-peak, color_tbl['BQ_mag'] - color_tbl['mag_1'],fmt = '.', capsize = 5,  label = 'B-g [QSO]', marker = 'D', c = 'r', alpha =0.5)
#ax2.errorbar(color_tbl['int_obsdate']-peak, color_tbl['VQ_mag'] - color_tbl['mag_2'],fmt = '.', capsize = 5,  label = 'V-r [QSO]', marker = 'x', c = 'r', alpha =0.5)
#ax2.errorbar(color_tbl2['int_obsdate']-peak, color_tbl2['RQ_mag'] - color_tbl2['mag_1'],fmt = '.', capsize = 5,  label = 'R-r [QSO]', marker = 'o', c = 'r', alpha =0.5)
ax2.grid()
ax2.legend()
plt.show()

#%%
plt.figure(figsize = (10,8), dpi = 500)

plt.ylabel('Difference')
plt.xlabel('days from B maximum')
plt.errorbar(color_tbl['int_obsdate']-peak, (color_tbl['B_mag']+color_tbl['V_mag'])/2 - color_tbl['mag_1'],fmt = '.', capsize = 5,  label = '(B+V)/2-g', marker = 's', c = 'g', alpha =0.5)
plt.errorbar(color_tbl['int_obsdate']-peak, color_tbl['B_mag'] - color_tbl['mag_1'],fmt = '.', capsize = 5,  label = 'B-g', marker = 'D', c = 'g', alpha =0.5)
plt.errorbar(color_tbl['int_obsdate']-peak, color_tbl['V_mag'] - color_tbl['mag_2'],fmt = '.', capsize = 5,  label = 'V-r', marker = 'x', c = 'g', alpha =0.5)
plt.errorbar(color_tbl2['int_obsdate']-peak, color_tbl2['R_mag'] - color_tbl2['mag_1'],fmt = '.', capsize = 5,  label = 'R-r', marker = 'o', c = 'g', alpha =0.5)
plt.errorbar(color_tbl['int_obsdate']-peak, (color_tbl['BQ_mag']+color_tbl['VQ_mag'])/2 - color_tbl['mag_1'],fmt = '.', capsize = 5,  label = '(B+V)/2-g [QSO]', marker = 's', c = 'r', alpha =0.5)
plt.errorbar(color_tbl['int_obsdate']-peak, color_tbl['BQ_mag'] - color_tbl['mag_1'],fmt = '.', capsize = 5,  label = 'B-g [QSO]', marker = 'D', c = 'r', alpha =0.5)
plt.errorbar(color_tbl['int_obsdate']-peak, color_tbl['VQ_mag'] - color_tbl['mag_2'],fmt = '.', capsize = 5,  label = 'V-r [QSO]', marker = 'x', c = 'r', alpha =0.5)
plt.errorbar(color_tbl2['int_obsdate']-peak, color_tbl2['RQ_mag'] - color_tbl2['mag_1'],fmt = '.', capsize = 5,  label = 'R-r [QSO]', marker = 'o', c = 'r', alpha =0.5)
plt.grid()
plt.legend()
plt.show()
#%%
import csv
f = open('/home/hhchoi1022/Desktop/2015F.B.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
obsdatelist=[]
maglist=[]
for line in rdr:
    obsdate, mag, _ = line
    obsdatelist.append(float(obsdate))
    maglist.append(float(mag)-1.5)
f.close()    
obsdatelist
maglist
len(maglist)
#%%
template_tbl_g = Table()
template_tbl_g['obsdate'] = obsdatelist
template_tbl_g['mag'] = maglist
#%%
template_tbl_r = Table()
template_tbl_r['obsdate'] = obsdatelist
template_tbl_r['mag'] = maglist
#%%
template_tbl_B = Table()
template_tbl_B['obsdate'] = obsdatelist
template_tbl_B['mag'] = maglist

#%%
plt.figure(figsize = (10,8))
plt.subplot(2,1,1)
plt.scatter(master_tbl['obsdate'], master_tbl['g_mag']-1, label = 'g_mag')
plt.scatter(master_tbl['obsdate'], master_tbl['r_mag']+1, label = 'r_mag')
#plt.scatter(master_tbl['obsdate'], master_tbl['B_mag']-1, label  ='B_mag(Converted)')
#plt.scatter(master_tbl['obsdate'], master_tbl['V_mag']+1, label  ='R_mag(Converted)')
plt.grid()
plt.xlabel('Days from g band Peak')
plt.ylabel('mag')
plt.legend()
plt.ylim(17,10)
plt.subplot(2,1,2)
plt.scatter(master_tbl['obsdate'], master_tbl['g-r_mag'], label = 'g-r')
#plt.scatter(master_tbl['obsdate'], master_tbl['B-V_mag'], label = 'B-V')
plt.grid()
plt.xlabel('Days from g band Peak')
plt.ylabel('color')
plt.ylim(-0.2,1)
plt.legend()

#%%

plt.figure(figsize = (10,8))
plt.subplot(2,1,1)
plt.scatter(template_tbl_g['obsdate'], template_tbl_g['mag'], label = 'g_mag')
#plt.scatter(template_tbl_r['obsdate'], template_tbl_r['mag'], label = 'r_mag')
plt.scatter(template_tbl_B['obsdate'], template_tbl_B['mag'], label  ='B_mag')
plt.xlabel('Days from B band Peak')
plt.ylabel('mag')
plt.legend()
plt.grid()
plt.xlim(-20,75)
plt.ylim(19,12)
plt.xlabel('Days from g band Peak')
plt.ylabel('mag')#%%
plt.xlabel('Days from g band Peak')
plt.ylabel('mag')#%%
plt.subplot(2,1,2)
plt.scatter(master_tbl['obsdate'], master_tbl['g-r_mag'], label = 'g-r')
plt.scatter(master_tbl['obsdate'], master_tbl['B-V_mag'], label = 'B-V')
plt.legend()

'''