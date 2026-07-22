

#%%
from ezphot.imageobjects import ScienceImage, CalibrationImage
# %%
import glob
target_directorylist = glob.glob('/qso/data6/obsdata/LOAO/2026_*')
# %%
import os
target_directory = target_directorylist[-1]
bias_pathlist = glob.glob(os.path.join(target_directory, 'zero*.fits'))
dark_pathlist = glob.glob(os.path.join(target_directory, 'dark*.fits'))
flat_pathlist = glob.glob(os.path.join(target_directory, 'ef*.fits'))
sci_pathlist = glob.glob(os.path.join(target_directory, 'obj*.fits'))
# %%
from ezphot.methods import Preprocess
preprocess = Preprocess()
#%%
bias_imglist = [CalibrationImage(bias_path) for bias_path in bias_pathlist]
dark_imglist = [CalibrationImage(dark_path) for dark_path in dark_pathlist]
flat_imglist = [CalibrationImage(flat_path) for flat_path in flat_pathlist]
new_bias_imglist = []
new_dark_imglist = []
new_flat_imglist = []
for bias_img in bias_imglist:
    bias_img.header['XBINNING'] = 2
    bias_img.header['YBINNING'] = 2
    bias_img.header['GAIN'] = 1.0
    bias_img.write()
    new_bias_imglist.append(CalibrationImage(bias_img.savepath.savepath))
for dark_img in dark_imglist:
    dark_img.header['XBINNING'] = 2
    dark_img.header['YBINNING'] = 2
    dark_img.header['GAIN'] = 1.0
    dark_img.write()
    new_dark_imglist.append(CalibrationImage(dark_img.savepath.savepath))
for flat_img in flat_imglist:
    flat_img.header['XBINNING'] = 2
    flat_img.header['YBINNING'] = 2
    flat_img.header['GAIN'] = 1.0
    flat_img.write()
    new_flat_imglist.append(CalibrationImage(flat_img.savepath.savepath))
#%%
mbias = preprocess.generate_masterframe(new_bias_imglist, save= False)
#%%
mdark = preprocess.generate_masterframe(dark_imglist, mbias = mbias)
#%%
mflat = preprocess.generate_masterframe(flat_imglist, mbias = mbias, mdark = mdark)

#%%
preprocess.generate_master_frames(bias_pathlist, dark_pathlist, flat_pathlist)
#%%
bias_image = CalibrationImage(bias_pathlist[0])
#%%
bias_image.show()
# %%
dark_image = CalibrationImage(dark_pathlist[0])
# %%
dark_image.show()
# %%
flat_image = CalibrationImage(flat_pathlist[0])
# %%
flat_image.show()
# %%
sci_image = ScienceImage(sci_pathlist[0])
# %%
sci_image.show('pixel')
# %%
