#!/usr/bin/env python
"""
Process LOAO 2x2 (E2V) observation nights: generate master frames, then
bias/dark/flat-correct the science images.

Raw LOAO headers lack XBINNING/YBINNING/GAIN, so they are injected before
processing (see 260621_preocess_loao.py draft).

Master frames are saved/registered under CALIBDATA_MASTERDIR, corrected
science images under SCIDATA_DIR.

Usage:
    python 260710_process_loao.py 2026_0709
    python 260710_process_loao.py 2026_0620 2026_0709   # inclusive range
"""
#%%
import gc
import glob
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from ezphot.helper import Helper
from ezphot.imageobjects import ScienceImage, CalibrationImage, MasterImage
from ezphot.methods import Preprocess
from ezphot.methods.stack import Combiner
#%%
RAW_BASE = Path('/qso/data6/obsdata/LOAO')

helper = Helper()
TELINFO = helper.get_telinfo(telescope='LOAO', ccd='E2V', binning=2)
HEADER_FIX = dict(XBINNING=2, YBINNING=2, GAIN=float(TELINFO['gain']))

preprocess = Preprocess()
combiner = Combiner(n_proc=4)
#%%
def load_calib(path):
    img = CalibrationImage(path, telinfo=TELINFO, load=False)
    img.header  # trigger lazy load
    for key, value in HEADER_FIX.items():
        img.header[key] = value
    return img

def release(img):
    img.clear(clear_data=True, verbose=False)


def combine_to_master(imglist, verbose=True):
    """Median-combine (extrema-clipped) a list of calibration images into a MasterImage."""
    datalist = [np.asarray(img.data, dtype=np.float32) for img in imglist]
    clip_type = 'extrema' if len(datalist) >= 5 else None
    combined, _ = combiner.combine_images_parallel(
        image_list=datalist,
        bkgrms_list=None,
        combine_method='median',
        clip_method=clip_type,
        nlow=1,
        nhigh=1,
        verbose=verbose,
    )
    header = imglist[0].header.copy()
    header['NCOMBINE'] = len(imglist)
    for i, img in enumerate(imglist):
        header[f'COMBIM{i + 1}'] = Path(img.path).name

    master_name = Path(imglist[0].path).with_suffix('.com.fits').name
    master = MasterImage(path=master_name, telinfo=TELINFO, load=False)
    master.data = combined.astype(np.float32)
    master.header = header
    return master


def get_or_make_master(imglist, verbose=True):
    """Return an existing registered master for this frame set, or build/register a new one."""
    probe = MasterImage(
        path=Path(imglist[0].path).with_suffix('.com.fits').name,
        telinfo=TELINFO, load=False)
    probe.header = imglist[0].header.copy()
    if probe.savepath.savepath.exists():
        print(f'  [SKIP] master exists: {probe.savepath.savepath.name}')
        return MasterImage(probe.savepath.savepath, telinfo=TELINFO), False
    master = combine_to_master(imglist, verbose=verbose)
    master.write(verbose=verbose)
    master.register(verbose=False)
    return master, True


def process_night(night, carried_flats, verbose=True, sci_key: str = '*NGC5907*'):
    """Process one raw night directory. Returns dict of failures."""
    night_dir = RAW_BASE / night
    bias_paths = sorted(glob.glob(str(night_dir / 'zero*.fits')))
    dark_paths = sorted(glob.glob(str(night_dir / 'dark*.fits')))
    flat_paths = sorted(glob.glob(str(night_dir / 'ef*.fits')) +
                        glob.glob(str(night_dir / 'mf*.fits')))
    sci_paths = sorted(glob.glob(str(night_dir / f'*{sci_key}*.fits')))
    print(f'\n===== {night}: {len(bias_paths)} bias, {len(dark_paths)} dark, '
          f'{len(flat_paths)} flat, {len(sci_paths)} sci =====')

    failures = {}

    # --- Master BIAS ---
    if not bias_paths:
        raise RuntimeError(f'{night}: no bias frames found')
    bias_imgs = [load_calib(p) for p in bias_paths]
    mbias, _ = get_or_make_master(bias_imgs, verbose=verbose)
    for img in bias_imgs:
        release(img)

    # --- Master DARK (per exposure time) ---
    if not dark_paths:
        raise RuntimeError(f'{night}: no dark frames found')
    dark_imgs = [load_calib(p) for p in dark_paths]
    darks_by_exptime = {}
    for img in dark_imgs:
        darks_by_exptime.setdefault(img.exptime, []).append(img)

    mdarks = {}
    for exptime, group in darks_by_exptime.items():
        probe = MasterImage(
            path=Path(group[0].path).with_suffix('.com.fits').name,
            telinfo=TELINFO, load=False)
        probe.header = group[0].header.copy()
        if probe.savepath.savepath.exists():
            print(f'  [SKIP] master exists: {probe.savepath.savepath.name}')
            mdarks[exptime] = MasterImage(probe.savepath.savepath, telinfo=TELINFO)
            continue
        bcorr = [preprocess.correct_bias(img, mbias, save=False, verbose=False)
                 for img in group]
        for img in group:
            release(img)
        mdark = combine_to_master(bcorr, verbose=verbose)
        mdark.write(verbose=verbose)
        mdark.register(verbose=False)
        mdarks[exptime] = mdark
        del bcorr
        gc.collect()
    # Use the longest-exposure master dark (subtract_dark scales by EXPTIME)
    mdark = mdarks[max(mdarks.keys())]

    # --- Master FLAT (per filter) ---
    mflats = {}
    if flat_paths:
        flat_imgs = [load_calib(p) for p in flat_paths]
        flats_by_filter = {}
        for img in flat_imgs:
            flats_by_filter.setdefault(img.filter, []).append(img)
        for filter_name, group in flats_by_filter.items():
            probe = MasterImage(
                path=Path(group[0].path).with_suffix('.com.fits').name,
                telinfo=TELINFO, load=False)
            probe.header = group[0].header.copy()
            if probe.savepath.savepath.exists():
                print(f'  [SKIP] master exists: {probe.savepath.savepath.name}')
                mflats[filter_name] = MasterImage(probe.savepath.savepath, telinfo=TELINFO)
                continue
            dbcorr = [preprocess.correct_bd(img, mbias, mdark, save=False, verbose=False)
                      for img in group]
            for img in group:
                release(img)
            mflat = combine_to_master(dbcorr, verbose=verbose)
            mflat.write(verbose=verbose)
            mflat.register(verbose=False)
            mflats[filter_name] = mflat
            del dbcorr
            gc.collect()
    carried_flats.update(mflats)

    # --- Science images ---
    n_done = n_skip = 0
    for path in sci_paths:
        try:
            sci = ScienceImage(path, telinfo=TELINFO, load=False)
            sci.header
            for key, value in HEADER_FIX.items():
                sci.header[key] = value
            if sci.savepath.savepath.exists():
                n_skip += 1
                release(sci)
                continue
            mflat = mflats.get(sci.filter) or carried_flats.get(sci.filter)
            if mflat is None:
                raise RuntimeError(f'no master flat for filter {sci.filter}')
            calib = preprocess.correct_bdf(sci, mbias, mdark, mflat,
                                           save=True, verbose=False)
            calib.data = None
            release(sci)
            n_done += 1
        except Exception as e:
            failures[path] = f'{type(e).__name__}: {e}'
            print(f'  [FAIL] {Path(path).name}: {failures[path]}')
        if (n_done + n_skip) % 50 == 0:
            gc.collect()
    print(f'  science: {n_done} processed, {n_skip} skipped, {len(failures)} failed')

    # Release masters
    release(mbias)
    for m in mdarks.values():
        release(m)
    for m in mflats.values():
        m.data = None
    gc.collect()
    return failures
#%%
all_nights = sorted(p.name for p in RAW_BASE.glob('2026_*') if p.is_dir())
#%%
sci_key = '*NGC5907*'
night = all_nights[53]
night = all_nights[-1]
night_dir = RAW_BASE / night
bias_paths = sorted(glob.glob(str(night_dir / 'zero*.fits')))
dark_paths = sorted(glob.glob(str(night_dir / 'dark*.fits')))
flat_paths = sorted(glob.glob(str(night_dir / 'ef*.fits')) + glob.glob(str(night_dir / 'mf*.fits')))
sci_paths = sorted(glob.glob(str(night_dir / f'{sci_key}.fits')))
#%%
from astropy.io import fits
hdr = fits.getheader(bias_paths[0])
#%%
telinfo = CalibrationImage(bias_paths[0]).telinfo
#%%
bias_imglist = []
dark_imglist = []
flat_imglist = []
sci_imglist = []
for p in bias_paths:
    img = CalibrationImage(p, telinfo = telinfo)
    img.header  # trigger lazy load
    for key, value in HEADER_FIX.items():
        img.header[key] = value
    img.write()
    bias_imglist.append(CalibrationImage(img.savepath.savepath, telinfo = telinfo))
for p in dark_paths:
    img = CalibrationImage(p, telinfo = telinfo)
    img.header  # trigger lazy load
    for key, value in HEADER_FIX.items():
        img.header[key] = value
    img.write()
    dark_imglist.append(CalibrationImage(img.savepath.savepath, telinfo = telinfo))
for p in flat_paths:
    img = CalibrationImage(p, telinfo = telinfo)
    img.header  # trigger lazy load
    for key, value in HEADER_FIX.items():
        img.header[key] = value
    img.write()
    flat_imglist.append(CalibrationImage(img.savepath.savepath, telinfo = telinfo))
for p in sci_paths:
    img = ScienceImage(p, telinfo = telinfo)
    img.header  # trigger lazy load
    for key, value in HEADER_FIX.items():
        img.header[key] = value
    img.write()
    sci_imglist.append(ScienceImage(img.savepath.savepath, telinfo = telinfo))
#%%
preprocess = Preprocess()
#%%
mbias_result = preprocess.generate_masterframe(bias_imglist, save = True, verbose = True)
mbias = list(mbias_result.values())[0]['BIAS']
#%%
mbias.show()
#%%
mdark_result = preprocess.generate_masterframe(dark_imglist, mbias = mbias, save = True, verbose = True)
mdark = list(list(mdark_result.values())[0]['DARK'].values())[0]
#%%
mdark.show()
#%%
mflat_result = preprocess.generate_masterframe(flat_imglist, mbias = mbias, mdark = mdark, save = True, verbose = True)
mflat_dict = list(mflat_result.values())[0]['FLAT']
mflat_list = list(mflat_dict.values())
#%%
for mflat in mflat_list:
    mflat.show()
#%%
target_img = sci_imglist[0]
target_img.show('pixel')
#%%
for target_img in sci_imglist:
    calib_img = preprocess.correct_bdf(target_img = target_img, bias_image = mbias, dark_image = mdark, flat_image = mflat_dict[target_img.filter], save = False, verbose = True)
    calib_img.show('pixel')

