
#%%
import matplotlib.pyplot as plt
from astropy.io import ascii
import os, glob

#%%
unsubfilelist = glob.glob('/home/hhchoi1022/Desktop/SN2021aefx_test/output/unsubtracted/*')
singlefilelist = glob.glob('/home/hhchoi1022/Desktop/SN2021aefx_test/output/single/*')
combinedfilelist = glob.glob('/home/hhchoi1022/Desktop/SN2021aefx_test/output/combined/*')
# %%
combinedfile = combinedfilelist[0]
combinedfilelist
# %%
from astropy.table import vstack
from astropy.table import Table
master_combined = Table()
master_single = Table()
master_unsub = Table()

for combinedfile in combinedfilelist:
    tbl = ascii.read(combinedfile, format = 'fixed_width')
    master_combined = vstack([master_combined, tbl])

for singlefile in singlefilelist:
    tbl = ascii.read(singlefile, format = 'fixed_width')
    master_single = vstack([master_single, tbl])
    
master_unsub = Table()
for unsubfile in unsubfilelist:
    tbl = ascii.read(unsubfile, format = 'fixed_width')
    master_unsub = vstack([master_unsub, tbl])
    
# %%
obs_combined = master_combined.group_by('observatory').groups
obs_single = master_single.group_by('observatory').groups
obs_unsub = master_unsub.group_by('observatory').groups
for obs in obs_combined:
    plt.scatter(obs['obsdate'], obs['mag'])
for obs in obs_single:
    plt.scatter(obs['obsdate'], obs['mag'] , alpha =0.2)
for obs in obs_unsub:
    plt.scatter(obs['obsdate'], obs['mag'] , alpha =0.2)

plt.ylim(14,16)
    
#%%
import numpy as np
plt.figure(dpi = 300)
plt.gca().invert_yaxis()
plt.ylim(16,14)
for i in range(3):
    combined = obs_combined[i]
    single = obs_single[i]
    unsub = obs_unsub[i]
    markerkey = ['s','.','*']
    obskey = ['KCT','LSGT','RASA36']
    #plt.scatter(single['obsdate'], single['mag'], alpha = 0.3, c = 'k', marker = markerkey[i])
    #plt.errorbar(single['obsdate'], single['mag'], fmt = 'none', yerr = np.sqrt(single['e_mag']**2+single['e_zp']**2), alpha = 0.1, c = 'k')
    plt.scatter(unsub['obsdate'], unsub['mag'], c = 'b', marker = markerkey[i])
    plt.errorbar(unsub['obsdate'], unsub['mag'], fmt = 'none', yerr = np.sqrt(unsub['e_mag']**2+unsub['e_zp']**2), alpha = 0.5, c = 'b')
    plt.scatter(combined['obsdate'], combined['mag'], c = 'r', marker = markerkey[i], label =obskey[i])
    plt.errorbar(combined['obsdate'], combined['mag'], fmt = 'none', yerr = np.sqrt(combined['e_mag']**2+combined['e_zp']**2), alpha = 0.5, c = 'r')
    
plt.legend()
plt.xlabel('JD')
plt.ylabel('Apparent mag')
# %%
plt.figure(figsize = (6,2), dpi = 300)
plt.ylabel('Depth')
plt.xlabel('JD')
plt.gca().invert_yaxis()
for i in range(3):
    combined = obs_combined[i]
    single = obs_single[i]
    unsub = obs_unsub[i]
    plt.scatter(unsub['obsdate'], unsub['depth'], c = 'k', marker = markerkey[i], alpha = 0.5)
# %%
unsub
# %%
