

#%%
target_path = '/home/hhchoi1022/7DT11_20260419_065907_T02080_m500_1x1_100.0s_0000.resamp.fits'
from ezphot.imageobjects import ScienceImage
#%%
from astropy.io import fits
with fits.open(target_path) as hdul:
    hdul[0].header['GAIN'] = 2750
    hdul[0].header['EGAIN'] = 0.2561
    hdul.writeto(target_path, overwrite=True)

target_image = ScienceImage(path = target_path)
# %%
target_image.show()
# %%
target_srcmask = target_image.calculate_sourcemask()
#%%
target_srcmask.show()
# %%
target_bkgmap = target_image.calculate_bkgmap(target_srcmask)
#%%
target_bkgmap.show()
# %%
target_bkgrms = target_image.calculate_bkgrms(target_srcmask)
#%%
target_bkgrms.show()
#%%