

#%%
DEFAULT_CONFIG_PATH = './default.swarp'
DEFAULT_OBSINFO_PATH = '../CCD.dat'
#%%
from tippy.helper import Helper
from astropy.io import ascii
# %%

helper = Helper()
# %%
all_obsinfo = ascii.read(DEFAULT_OBSINFO_PATH, format = 'fixed_width')
default_config = helper.load_config(DEFAULT_CONFIG_PATH)
for observatory_info in all_obsinfo:
    observatory = observatory_info['obs']
    ccd = observatory_info['ccd']
    binning = observatory_info['binning']
    obsmode = observatory_info['mode']
    if not isinstance(obsmode, str):
        telkey = observatory + '_' + ccd + '_' + str(binning) + 'x' + str(binning)
    else:
        telkey = observatory + '_' + ccd + '_' + str(binning) + 'x' + str(binning) + '_' + observatory_info['mode']
# %%
