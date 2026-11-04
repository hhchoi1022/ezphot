#!/usr/bin/env python
"""
visualize: per-stage figures of a process_images run, collected into a static HTML gallery
for looking at the products by eye. Read-only on the products; writes only under --out.

A stage is shown when its products are on disk (or when it is named in --stages):
    preprocess  header BIASCOR/DARKCOR/FLATCOR + master paths in .info -> raw (rebuilt) vs corrected, masters
    platesolve  SCAMP PV1_* in header -> detected-vs-reference offsets over the field, 3x3 cutouts
    mask        .srcmask / .invalidmask -> image, background source mask (.srcmask), invalid object mask (.invalidmask)
    background  .bkgmap / .bkgrms
    photometry  .cat + ZP_APER_3 -> ZP figures written by ezphot (tab "ZP Calibration")
    run         summary-level ZP / seeing / depth vs time (always, >1 image)
    lightcurve  only with --lightcurve (target_photometry table): lightcurve.png when >1 epoch, and
                interactive light-curve / SED tabs of MAGSKY_APER_3 (10") from 1 row on (a table with no usable
                row gives the tabs with a note): frames are combined in the page per time bin and filter
                (default 1 day, width changeable); filter wavelength = response-weighted mean of the transmission
                curve in SYNPHOT_FILTERDIR (<observatory>_<filter>.csv, then <filter>.csv)
    sources     .cat with MAGSKY_APER_3 -> detections with SNR > 3 (MAGERR_APER_3 < 1.0857/3: X_WORLD, Y_WORLD,
                MAGSKY_APER_3, MAGERR_APER_3, SATDIST) of all frames in sources.json next to index.html; the light-curve /
                SED tabs then take a coordinate (default --ra/--dec; with --lightcurve the table is shown first) and use
                the nearest detection within a radius (default 5") in each frame, fetched from sources.json over http;
                fov.jpg (about 1200 px wide) shows the whole field of the frame with the most of them in those tabs, and a
                click on it takes the RA/Dec of the nearest detection of that frame

    python -m ezphot.recipes.visualize summary.csv [--ra deg --dec deg] [--lightcurve lc.csv] \
        [--stages platesolve photometry] [--max-images N] [--root DIR] [--n-proc 8] [--skip-unchanged | --pages-only]

Images are drawn in --n-proc processes (default 8; 1 = one after the other): the first image of a gallery
alone, so that its platesolve figure fills the sky-catalog cache, then the rest in parallel. The
preprocess figure reads the whole science image, the mask figure all of it in bands of 8 rows; the other figures
read its shape and cutouts. In the page the figures of a stage are sized to fit the window height.
Each frame folder keeps .viz_done.json (stage figures drawn, file mtime/size of their inputs, FIG_REV), written by
every run that draws: --skip-unchanged then redraws only the stages whose inputs changed, --pages-only draws no
stage figure and rebuilds the pages from the figures on disk (the run-level figures, sources.json, fov.jpg and the
pages are made again in every mode).

Gallery: <root>/ezphot/<objname>/viz/index.html (root default ~/viz); a later run of the same
objname replaces the page. Frames are listed as "filter|telname|obsdate: raw file name", medium-band filters
(m###, m###w) before broad band, each by filter wavelength, then telname, obsdate (--max-images takes the first
N summary rows before sorting). Serve with
    python -m http.server 8765 --bind 127.0.0.1 --directory ~/viz
    -> http://127.0.0.1:8765/ezphot/<objname>/viz/
"""
from __future__ import annotations

import argparse
import html
import io
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS

from ezphot.recipes import defaults as D

STAGES = ('preprocess', 'platesolve', 'mask', 'background', 'photometry')
VIZ_ROOT = Path.home() / 'viz'
ZP_KEY = f'ZP_{D.APER10_KEY}'
MAG_KEY, MAGERR_KEY, MAGSKY_KEY = f'MAG_{D.APER10_KEY}', f'MAGERR_{D.APER10_KEY}', f'MAGSKY_{D.APER10_KEY}'
LC_BIN_DAYS = 1                                    # default time bin of the interactive light-curve / SED tabs
SOURCES_FILE = 'sources.json'                      # per-frame detections for the coordinate input of those tabs
SOURCES_SNR = 3                                    # detections kept: MAGERR_APER_3 < 1.0857 / SOURCES_SNR
SOURCES_RADIUS_ARCSEC = 5.0                        # default matching radius in the page (= target_photometry --radius)
FOV_FILE = 'fov.jpg'                               # whole field of one frame for the click input of those tabs
FOV_WIDTH = 1200                                   # its width is NAXIS1 / ceil(NAXIS1 / FOV_WIDTH) px
FIG_REV = 1                                        # bump only when a figure drawn here or its thumbnail changes content:
                                                   # stage records of another rev are redrawn by --skip-unchanged
DONE_FILE = '.viz_done.json'                       # per frame folder: stage figures drawn, from which inputs (not in manifest)
_CACHE = {}                                        # per process, cleared by visualize(): sky-catalog utility and tiles, masters
_FRAME_CATS = None                                 # .cat tables read while one frame is drawn (_frame), else no memo


# ---------------------------------------------------------------- helpers
def _bin(data: np.ndarray, f: int, func=np.nanmean) -> np.ndarray:
    ny, nx = (data.shape[0] // f) * f, (data.shape[1] // f) * f
    blocks = data[:ny, :nx].reshape(ny // f, f, nx // f, f)
    if func is np.nanmean:                          # np.mean is faster and the same where no block has a NaN
        out = np.mean(blocks, axis=(1, 3))
        return np.nanmean(blocks, axis=(1, 3)) if np.isnan(out).any() else out
    return func(blocks, axis=(1, 3))


def _binned(data, f: int, func=np.nanmean) -> np.ndarray:
    """_bin of a 2-D array read in bands of f rows, so that a memory map or _Cutouts is not converted whole."""
    return np.vstack([_bin(np.asarray(data[r:r + f], dtype=np.float32), f, func)
                      for r in range(0, data.shape[0] // f * f, f)])


def _show(ax, data, title=None, cmap='gray', **kw):
    vmin, vmax = ZScaleInterval().get_limits(data[np.isfinite(data)])
    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, **kw)
    if title:
        ax.set_title(title, fontsize=9)
    return im


def _wcs(header) -> WCS:
    """SCAMP writes PV1_* with CTYPE TAN; astropy applies them only as TPV."""
    h = header.copy()
    if any(k.startswith('PV1_') for k in h):
        h['CTYPE1'], h['CTYPE2'] = 'RA---TPV', 'DEC--TPV'
    return WCS(h)


def _info(path: Path) -> dict:
    p = Path(str(path) + '.info')
    return json.loads(p.read_text()) if p.exists() else {}


def _product(path: Path, suffix: str) -> Path:
    return Path(str(path) + suffix)


def _link(src: Path, dst_dir: Path) -> Optional[str]:
    """Symlink an existing ezphot figure into the gallery (http.server follows links)."""
    if not src.exists():
        return None
    dst = dst_dir / src.name
    if dst.is_symlink():
        if os.readlink(dst) == str(src):            # the same link: kept
            return dst.name
        dst.unlink()
    elif dst.exists():
        dst.unlink()
    os.symlink(src, dst)
    return dst.name


def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return path.name


def _read_cat(path: Path) -> Table:
    """While a frame is drawn (_frame) a .cat is parsed once for fig_platesolve and _sources (both only read it)."""
    if _FRAME_CATS is None:
        return Table.read(str(path), format='ascii.basic')
    if str(path) not in _FRAME_CATS:
        _FRAME_CATS[str(path)] = Table.read(str(path), format='ascii.basic')
    return _FRAME_CATS[str(path)]


def _catalog_util():
    """One SkyCatalogUtility per process whose tiles (SkyCatalog of an archive summary row) are kept by objname,
    catalog_type, catalog_version, fov_ra, fov_dec, so that frames of the same field load them once; the archive
    summary is still read by every get_catalogs. Cleared by visualize(); forked workers start with the parent's."""
    if 'util' not in _CACHE:
        from ezphot.skycatalog import SkyCatalogUtility

        class _TileCache(SkyCatalogUtility):
            @staticmethod
            def _load_catalog_worker(row):
                key = (str(row['objname']), str(row['catalog_type']), str(row['catalog_version']),
                       float(row['fov_ra']), float(row['fov_dec']))
                tiles = _CACHE.setdefault('tiles', {})
                if key not in tiles:
                    tiles[key] = SkyCatalogUtility._load_catalog_worker(row)
                return tiles[key]

        _CACHE['util'] = _TileCache()
    return _CACHE['util']


def _masters(paths):
    """bias, dark, flat / mean(flat) and the dark EXPTIME of one master set (paths as in .info), kept in this process
    for the next frame of the same set: key = (path, mtime_ns) of each; only the latest set is kept."""
    key = tuple((str(p), os.stat(p).st_mtime_ns) for p in paths)
    if _CACHE.get('masters', (None,))[0] != key:
        _CACHE.pop('masters', None)                 # free the previous set before reading the next
        bias, dark, flat = (fits.getdata(p).astype(np.float32, copy=False) for p in paths)
        _CACHE['masters'] = (key, (bias, dark, flat / np.nanmean(flat), fits.getheader(paths[1])['EXPTIME']))
    return _CACHE['masters'][1]


class _Cutouts:
    """Stands in for data.astype(np.float32) in the stage figures: slices are converted on access, so of a
    memory-mapped fits.getdata array only the rows a cutout spans are read (in mmap pages); fig_preprocess takes
    the whole image with [:, :] once its masters exist, fig_mask reads it through _binned.
    (Compressed or BZERO/BSCALE/BLANK-scaled images are decoded in full by fits.getdata, as before.)"""
    def __init__(self, data: np.ndarray):
        self.shape, self._data, self._blocks = data.shape, data, {}

    def __getitem__(self, key) -> np.ndarray:
        return self._data[key].astype(np.float32)

    def blocks(self, f: int) -> np.ndarray:
        """_binned(self, f), computed once per frame (fig_preprocess and fig_mask both show the 8x8 image)."""
        if f not in self._blocks:
            self._blocks[f] = _binned(self, f)
        return self._blocks[f]


def _blocks(data, f: int) -> np.ndarray:
    return data.blocks(f) if isinstance(data, _Cutouts) else _binned(data, f)


# ---------------------------------------------------------------- stage figures
def fig_preprocess(path: Path, data, header, info, outdir: Path) -> List[str]:
    paths = [info.get(k) for k in ('BIASPATH', 'DARKPATH', 'FLATPATH')]
    if not all(p and Path(p).exists() for p in paths):
        return []
    f = 8
    corrected = _blocks(data, f)                    # = _bin of the whole image (rows in bands of f), shared with fig_mask
    data = data[:, :]                               # whole image: float32 from _Cutouts, a view of an ndarray
    bias, dark, flat_norm, dark_exptime = _masters(paths)   # flat_norm: ccdproc.flat_correct divides by flat / mean(flat)
    dark_scale = header['EXPTIME'] / dark_exptime
    raw = data * flat_norm + dark * dark_scale + bias
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    _show(axes[0, 0], _bin(raw, f), 'raw (rebuilt = corr*flat + dark + bias)')
    _show(axes[0, 1], corrected, 'corrected')
    ax = axes[0, 2]
    for arr, lab in ((raw, 'raw'), (data, 'corrected')):
        v = arr[::7, ::7].ravel()
        lo, hi = np.nanpercentile(v, [0.5, 99.5])
        ax.hist(v, bins=200, range=(lo, hi), histtype='step', label=lab)
    ax.set_yscale('log'); ax.set_xlabel('ADU'); ax.legend(fontsize=8); ax.set_title('pixel histogram', fontsize=9)
    _show(axes[1, 0], _bin(bias, f), f'master bias\n{Path(paths[0]).name}')
    _show(axes[1, 1], _bin(dark * dark_scale, f), f'master dark x{dark_scale:.2f}\n{Path(paths[1]).name}')
    _show(axes[1, 2], _bin(flat_norm, f), f'master flat / mean\n{Path(paths[2]).name}')
    for ax in axes.ravel()[[0, 1, 3, 4, 5]]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(path.name, fontsize=10)
    return [_save(fig, outdir / 'preprocess.png')]


def fig_platesolve(path: Path, data, header, info, outdir: Path) -> List[str]:
    cat_path = _product(path, '.cat')
    if not cat_path.exists():
        return []
    w = _wcs(header)
    ny, nx = data.shape
    centre = w.pixel_to_world(nx / 2, ny / 2)
    util = _catalog_util()
    refs = [util.select_reference_sources(catalog=c, mag_lower=10, mag_upper=18)[0]
            for c in util.get_catalogs(ra=centre.ra.deg, dec=centre.dec.deg,
                                       fov_ra=info.get('FOVX', 1.4), fov_dec=info.get('FOVY', 0.9),
                                       catalog_type=D.PHOTCAL['catalog_type'],
                                       catalog_version=D.PHOTCAL['catalog_version'])]
    if not refs:
        return []
    ref = SkyCoord(np.concatenate([np.asarray(r['ra']) for r in refs]),
                   np.concatenate([np.asarray(r['dec']) for r in refs]), unit='deg')
    cat = _read_cat(cat_path)
    cat = cat[(cat['FLAGS'] == 0) & (cat['ELONGATION'] < 1.5)]
    det = SkyCoord(cat['ALPHA_J2000'], cat['DELTA_J2000'], unit='deg')   # SExtractor world coords (PV applied)
    idx, sep, _ = det.match_to_catalog_sky(ref)
    ok = sep < 3 * u.arcsec
    dra, ddec = det[ok].spherical_offsets_to(ref[idx[ok]])
    dra, ddec = dra.to_value(u.arcsec), ddec.to_value(u.arcsec)
    x, y = np.asarray(cat['X_IMAGE'])[ok], np.asarray(cat['Y_IMAGE'])[ok]
    r = np.hypot(x - nx / 2, y - ny / 2) * info.get('PIXELSCALE', 0.505) / 60
    names = []

    fig = plt.figure(figsize=(9.5, 5))              # width / height < 1.8: not a full-row strip; page: left of wcs_cutouts
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1], hspace=0.5, wspace=0.25, top=0.85)   # top: room for the 2-line suptitle
    axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]   # field | distance / histogram
    ax = axes[0]
    q = ax.quiver(x, y, dra, ddec, np.hypot(dra, ddec), cmap='viridis', angles='xy',
                  scale_units='xy', scale=1 / 2000, width=0.003)
    ax.quiverkey(q, 0.02, -0.14, 0.5, '0.5"', labelpos='E')     # below the tick labels, above the colour bar
    ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_aspect('equal')
    ax.set_title(f'detected -> reference (x2000), N={ok.sum()}', fontsize=9)
    fig.colorbar(q, ax=ax, label='|offset| [arcsec]', orientation='horizontal', fraction=0.06, pad=0.16)   # below: the field fills its column
    ax = axes[1]
    ax.scatter(r, np.hypot(dra, ddec), s=4, alpha=0.6)
    ax.set_xlabel('distance from centre [arcmin]'); ax.set_ylabel('|offset| [arcsec]')
    ax.set_title(f'median {np.median(np.hypot(dra, ddec)):.2f}"', fontsize=9)
    ax = axes[2]
    ax.hist(dra, bins=40, histtype='step', label=f'dRA* med {np.median(dra):+.2f}"')
    ax.hist(ddec, bins=40, histtype='step', label=f'dDec med {np.median(ddec):+.2f}"')
    ax.set_xlabel('arcsec'); ax.legend(fontsize=8)
    fig.suptitle(f'{path.name}\nCTYPE={header["CTYPE1"]}  PV1 terms={sum(k.startswith("PV1_") for k in header)}', fontsize=9)
    names.append(_save(fig, outdir / 'wcs_offsets.png'))

    half = 240                                      # cutout half-size [pix] = 2'
    px, py = w.world_to_pixel(ref)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, cy in enumerate((ny - 1.5 * half, ny / 2, 1.5 * half)):
        for j, cx in enumerate((1.5 * half, nx / 2, nx - 1.5 * half)):
            ax = axes[i, j]
            x0, y0 = int(cx - half), int(cy - half)
            _show(ax, data[y0:y0 + 2 * half, x0:x0 + 2 * half])
            sel = (px > x0) & (px < x0 + 2 * half) & (py > y0) & (py < y0 + 2 * half)
            ax.scatter(px[sel] - x0, py[sel] - y0, s=60, facecolors='none', edgecolors='r', lw=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'x={int(cx)} y={int(cy)}', fontsize=8)
    fig.suptitle('reference positions through the header WCS (red), 4\'x4\' cutouts', fontsize=10)
    names.append(_save(fig, outdir / 'wcs_cutouts.png'))
    return names


def fig_mask(path: Path, data, header, info, outdir: Path) -> List[str]:
    """Image, background source mask (.srcmask) and invalid object mask (.invalidmask) in 8x8 blocks (masks read-only,
    a block is masked when any of its pixels is); a missing mask file is noted in its panel."""
    masks = [_product(path, s) for s in ('.srcmask', '.invalidmask')]
    have = [p.exists() for p in masks]
    if not any(have):
        return []
    f = 8
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    _show(axes[0], _blocks(data, f), f'image, {f}x{f} mean')
    for ax, p, ok, lab in zip(axes[1:], masks, have, ('background source mask (.srcmask)',
                                                      'invalid object mask')):
        if ok:
            m = _binned(fits.getdata(p, memmap=True), f, lambda a, axis: np.mean(a > 0, axis=axis))   # masked fraction
            ax.imshow(m > 0, origin='lower', cmap='gray_r', vmin=0, vmax=1)
            ax.set_title(f'{lab}\n{f}x{f} blocks with any masked pixel ({100 * np.mean(m):.3f}% masked)', fontsize=9)
        else:
            ax.text(0.5, 0.5, f'none (no {p.suffix} file)', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, data.shape[1] // f); ax.set_ylim(0, data.shape[0] // f); ax.set_aspect('equal')
            ax.set_title(lab, fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(path.name, fontsize=10)
    return [_save(fig, outdir / 'mask.png')]


def fig_background(path: Path, data, header, info, outdir: Path) -> List[str]:
    return [n for n in (_link(_product(path, s), outdir) for s in ('.bkgmap.png', '.bkgrms.png')) if n]


def fig_photometry(path: Path, data, header, info, outdir: Path) -> List[str]:
    return [n for n in (_link(_product(path, s), outdir)
                        for s in ('.refcat.png', '.zp_mag.png', '.zp_color.png', '.zp_2d.png')) if n]


def fig_run(summary: pd.DataFrame, outdir: Path) -> List[str]:
    ok = summary[summary['success'].astype(bool)].copy()
    if len(ok) < 2:
        return []
    ok['t'] = pd.to_datetime(ok['obsdate'])
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
    for filt, g in ok.groupby('filter'):
        axes[0].errorbar(g['t'], g[ZP_KEY], g[f'ZPERR_{D.APER10_KEY}'], fmt='o', ms=4, label=filt)
        axes[1].plot(g['t'], g['SEEING'], 'o', ms=4)
        axes[2].plot(g['t'], g[f'UL5SKY_{D.APER10_KEY}'], 'o', ms=4)
    for ax, lab in zip(axes, (ZP_KEY, 'SEEING ["]', f'UL5SKY_{D.APER10_KEY}')):
        ax.set_title(lab, fontsize=10)
        ax.tick_params(axis='x', labelrotation=30, labelsize=8)
    axes[2].invert_yaxis(); axes[0].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    return [_save(fig, outdir / 'run_trends.png')]


def fig_lightcurve(lc: pd.DataFrame, outdir: Path, run_paths=()) -> List[str]:
    """Filled: catalogs of images in this summary; open: other catalogs found by target_photometry."""
    if len(lc) < 2 or MAGSKY_KEY not in lc:
        return []
    lc = lc.copy()
    lc['t'] = pd.to_datetime(lc['obsdate'])
    in_run = lc['target_img'].astype(str).isin(set(map(str, run_paths))) if 'target_img' in lc else np.ones(len(lc), bool)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for i, (filt, g) in enumerate(lc.groupby('filter')):
        c = f'C{i}'
        m = in_run[g.index]
        ax.errorbar(g['t'][m], g[MAGSKY_KEY][m], g[MAGERR_KEY][m], fmt='o', ms=5, color=c, label=f'{filt} (this run)')
        if (~m).any():
            ax.errorbar(g['t'][~m], g[MAGSKY_KEY][~m], g[MAGERR_KEY][~m], fmt='o', ms=5, color=c, mfc='none',
                        label=f'{filt} (other catalogs)')
    ax.invert_yaxis(); ax.set_ylabel(MAGSKY_KEY + ' (10")'); ax.legend(fontsize=7, ncol=4)
    ax.tick_params(axis='x', labelrotation=30, labelsize=8)
    fig.tight_layout()
    return [_save(fig, outdir / 'lightcurve.png')]


def _filter_wavelength(filterdir: Path, observatory: str, filt: str):
    """Response-weighted mean wavelength [nm] and file name of <observatory>_<filter>.csv, else <filter>.csv
    (columns wavelength [Angstrom], response); (None, None) when neither exists."""
    for f in (filterdir / f'{observatory}_{filt}.csv', filterdir / f'{filt}.csv'):
        if f.exists():
            t = pd.read_csv(f)
            w, r = t.iloc[:, 0].to_numpy(float), t.iloc[:, 1].to_numpy(float)
            dw = np.gradient(w)
            wl = np.sum(w * r * dw) / np.sum(r * dw) / 10
            if not np.isfinite(wl):
                raise ValueError(f'{f.name}: no finite response-weighted wavelength')
            return round(float(wl), 1), f.name
    return None, None


def _filters(obs: list, filt: list, filterdir: Path):
    """(observatory, filter) keys sorted by wavelength and their page entries (name, wavelength, source, note, broad)."""
    multi_obs = len(set(obs)) > 1                  # same filter name from two observatories = two filters
    filters = {}
    for o, f in set(zip(obs, filt)):
        note = ''
        try:
            wl, src = _filter_wavelength(filterdir, o, f)
        except Exception as e:
            wl, src, note = None, None, f'{type(e).__name__}: {e}'.replace(str(filterdir), 'SYNPHOT_FILTERDIR')
            print(f"[visualize] transmission curve of {f} unreadable: {note}")
        m = re.fullmatch(r'm(\d{3})w?', f)
        if wl is None and m:                       # shown as an estimate in the page
            wl, src = float(m.group(1)), 'name'
        filters[(o, f)] = dict(n=f'{o} {f}' if multi_obs else f, wl=wl, src=src, note=note,
                               broad=not re.fullmatch(r'm\d{3}', f))
    keys = sorted(filters, key=lambda k: (filters[k]['wl'] is None, filters[k]['wl'] or 0, k))
    return keys, filters


def _lc_payload(lc: pd.DataFrame, run_paths=(), verbose=True) -> dict:
    """Points and filter wavelengths for the interactive light-curve / SED tabs; binning and combining
    are done in the page. Rows with a non-finite magnitude, error or mjd, or MAGERR >= 99, are dropped (counted);
    the bin origin t0 is the earliest finite mjd of all rows, dropped ones included."""
    from ezphot.configuration import Configuration
    filterdir = Path(Configuration(_silent=True).config['SYNPHOT_FILTERDIR'])
    mag, err, mjd = (pd.to_numeric(lc[k], errors='coerce').to_numpy(float) for k in (MAGSKY_KEY, MAGERR_KEY, 'mjd'))
    ok = np.isfinite(mag) & np.isfinite(err) & np.isfinite(mjd) & (err < 99)
    col = lambda k: (lc[k].fillna('').astype(str) if k in lc else pd.Series('', index=lc.index))[ok].tolist()
    obs, filt, tel, img = col('observatory'), col('filter'), col('telname'), col('target_img')
    keys, filters = _filters(obs, filt, filterdir)
    index = {k: i for i, k in enumerate(keys)}
    run = set(map(str, run_paths))
    pts = [[round(float(t), 6), index[(o, f)], round(float(m), 4), round(float(e), 4), Path(p).stem if p else '', u,
            int(p in run or 'target_img' not in lc)]
           for t, o, f, m, e, p, u in zip(mjd[ok], obs, filt, mag[ok], err[ok], img, tel)]
    if verbose:
        print(f"[visualize] interactive light curve: {len(pts)} points, {len(keys)} filters, {int((~ok).sum())} rows dropped; "
              f"filter curves from {filterdir}")
    return dict(key=MAGSKY_KEY, bin=LC_BIN_DAYS, t0=round(float(np.nanmin(mjd)), 6) if np.isfinite(mjd).any() else None,
                n_bad=int((~ok).sum()), filters=[filters[k] for k in keys], pts=pts)


def _sources(path: Path, row, info) -> Optional[dict]:
    """Detections of one frame for the coordinate input of the light-curve / SED tabs: rows of the .cat with finite
    X_WORLD, Y_WORLD, MAGSKY_APER_3 and MAGERR_APER_3 < 1.0857 / SOURCES_SNR (SATDIST None when missing; x, y =
    X_IMAGE, Y_IMAGE for _fov, not in SOURCES_FILE); None without a .cat with MAGSKY_APER_3. mjd of the summary obsdate,
    as target_photometry takes it."""
    cat_path = _product(path, '.cat')
    if not cat_path.exists():
        return None
    cat = _read_cat(cat_path).to_pandas()
    if MAGSKY_KEY not in cat:
        return None
    col = lambda k: (pd.to_numeric(cat[k], errors='coerce').to_numpy(float, na_value=np.nan) if k in cat
                     else np.full(len(cat), np.nan))
    ra, dec, mag, err, sat, x, y = (col(k) for k in ('X_WORLD', 'Y_WORLD', MAGSKY_KEY, MAGERR_KEY, 'SATDIST',
                                                     'X_IMAGE', 'Y_IMAGE'))
    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mag) & (err < 1.0857 / SOURCES_SNR)
    filt = row.get('filter')
    return dict(t=round(float(Time(str(row['obsdate'])).mjd), 6), obs=str(info.get('OBSERVATORY') or ''),
                f='' if filt is None or pd.isna(filt) else str(filt),
                ra=np.round(ra[ok], 6).tolist(), dec=np.round(dec[ok], 6).tolist(),
                mag=np.round(mag[ok], 4).tolist(), err=np.round(err[ok], 4).tolist(),
                sat=[round(float(s), 3) if np.isfinite(s) else None for s in sat[ok]],
                x=np.round(x[ok], 2).tolist(), y=np.round(y[ok], 2).tolist())


def _sources_payload(srcs: list, verbose=True) -> dict:
    """Content of SOURCES_FILE: frames [index in the page's image list, mjd, filter index, number of rows] and the
    rows of those frames one after the other in column lists (ra, dec, mag, err, sat); filters as in _lc_payload,
    t0 = earliest frame mjd. No paths."""
    from ezphot.configuration import Configuration
    filterdir = Path(Configuration(_silent=True).config['SYNPHOT_FILTERDIR'])
    have = [(i, s) for i, s in enumerate(srcs) if s is not None]
    keys, filters = _filters([s['obs'] for _, s in have], [s['f'] for _, s in have], filterdir)
    index = {k: i for i, k in enumerate(keys)}
    frames = [[i, s['t'], index[(s['obs'], s['f'])], len(s['ra'])] for i, s in have]
    cols = {k: [v for _, s in have for v in s[k]] for k in ('ra', 'dec', 'mag', 'err', 'sat')}
    if verbose:
        print(f"[visualize] coordinate input: {len(cols['ra'])} detections with SNR > {SOURCES_SNR} "
              f"in {len(have)} frames -> {SOURCES_FILE}")
    return dict(key=MAGSKY_KEY, snr=SOURCES_SNR, t0=min(s['t'] for _, s in have),
                filters=[filters[k] for k in keys], frames=frames, **cols)


def _fov(path: Path, src: dict, out: Path) -> dict:
    """Whole field of one frame for the click input of the light-curve / SED tabs: FOV_FILE in out (JPEG of f x f
    nanmean blocks, f = ceil(NAXIS1 / FOV_WIDTH), read in bands of f rows from the read-only memory map; ZScale as in
    _show, NaN black, highest Y on top) and the frame's detections (_sources) in its pixels: u = (X_IMAGE - 0.5) / f
    from the left, v from the top. ValueError when X_IMAGE / Y_IMAGE do not all fall within the image."""
    from PIL import Image
    data = fits.getdata(path, memmap=True)          # compressed or scaled images are decoded in full, as in _frame
    ny, nx = data.shape
    x, y = np.asarray(src['x'], float), np.asarray(src['y'], float)
    bad = ~((x >= 0.5) & (x <= nx + 0.5) & (y >= 0.5) & (y <= ny + 0.5))
    if bad.any():
        raise ValueError(f'X_IMAGE/Y_IMAGE of {int(bad.sum())} of {len(x)} detections not within the {nx} x {ny} image')
    f = -(-nx // FOV_WIDTH)
    img = _binned(data, f)
    vmin, vmax = ZScaleInterval().get_limits(img[np.isfinite(img)])
    Image.fromarray(np.nan_to_num(np.clip((img[::-1] - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)).save(
        out / FOV_FILE, quality=85)
    h, w = img.shape
    return dict(file=FOV_FILE, w=w, h=h, u=np.round((x - 0.5) / f, 2).tolist(), v=np.round(h - (y - 0.5) / f, 2).tolist(),
                ra=src['ra'], dec=src['dec'])


# ---------------------------------------------------------------- driver
def _auto_stages(path: Path, header, info) -> List[str]:
    found = []
    if header.get('BIASCOR') and header.get('DARKCOR') and header.get('FLATCOR') and info.get('FLATPATH'):
        found.append('preprocess')
    if any(k.startswith('PV1_') for k in header):
        found.append('platesolve')
    if _product(path, '.invalidmask').exists() or _product(path, '.srcmask').exists():
        found.append('mask')
    if _product(path, '.bkgmap').exists() or _product(path, '.bkgrms').exists():
        found.append('background')
    if _product(path, '.cat').exists() and ZP_KEY in header:
        found.append('photometry')
    return found


STAGE_LABELS = dict(preprocess='전처리', platesolve='WCS', mask='마스크', background='배경', photometry='ZP Calibration')
CAPTIONS = {
    'preprocess.png': '보정 전(master로 재구성) / 보정 후, 픽셀 히스토그램, 사용된 master bias·dark·flat',
    'wcs_offsets.png': '검출 위치 → 기준성 위치 오프셋: 시야 분포, 중심 거리 의존성, 히스토그램',
    'wcs_cutouts.png': '헤더 WCS로 투영한 기준성 위치(빨강), 시야 9곳 4′×4′',
    'mask.png': '원본 이미지(8×8 평균) · background source mask(.srcmask) · invalid object mask(.invalidmask) — 마스크는 8×8 블록 중 하나라도 마스크된 곳',
    '.bkgmap.png': '2D 배경 맵',
    '.bkgrms.png': '배경 rms 맵',
    '.refcat.png': 'ZP 계산에 쓴 기준성 선택',
    '.zp_mag.png': 'ZP vs 기준성 등급',
    '.zp_color.png': 'ZP vs 색',
    '.zp_2d.png': 'ZP 공간 분포',
    'run_trends.png': f'{ZP_KEY}, SEEING, UL5SKY (10″) 시간 추이',
    'lightcurve.png': f'{MAGSKY_KEY} 광도곡선 — 채운 원: 이번 summary의 이미지, 빈 원: 디스크의 다른 카탈로그',
}
SUMMARY_COLS = [('filter', '필터'), ('telname', '망원경'), ('obsdate', '관측 시각 (UT)'), ('exptime', '노출'),
                ('SEEING', 'seeing'), (ZP_KEY, 'ZP'), (f'ZPERR_{D.APER10_KEY}', 'ZP err'),
                (f'UL5SKY_{D.APER10_KEY}', 'UL5 (10″)'), ('n_ref', 'n_ref')]


def _figure(outdir: Path, name: str, prefix: str = '', reuse: bool = False) -> dict:
    """Gallery entry with a JPEG thumbnail; the full PNG opens on click. reuse (--skip-unchanged / --pages-only, the
    stage's recorded inputs equal to the current ones): an existing thumbnail that decodes is kept, its size giving wide;
    otherwise (missing, unreadable) it is made again."""
    from PIL import Image
    thumb = f'thumb_{Path(name).stem}.jpg'
    caption = next((v for k, v in CAPTIONS.items() if name.endswith(k)), '')
    if reuse:
        try:
            with Image.open(outdir / thumb) as im:
                im.load()                           # a truncated or damaged JPEG fails here
                w, h = im.size
            return dict(src=prefix + name, thumb=prefix + thumb, caption=caption, wide=w > 1.8 * h)
        except Exception:
            pass
    with Image.open(outdir / name) as im:
        im.thumbnail((1400, 1400))
        bg = Image.new('RGB', im.size, 'white')
        bg.paste(im, mask=im.getchannel('A') if im.mode == 'RGBA' else None)
        buf = io.BytesIO()                          # one write, without the existence checks of save(<path>)
        bg.save(buf, format='JPEG', quality=85)
        tmp = outdir / (thumb + '.tmp')
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, outdir / thumb)             # never a half-written thumbnail under its name
        wide = im.size[0] > 1.8 * im.size[1]              # multi-panel strips span the full row
    return dict(src=prefix + name, thumb=prefix + thumb, caption=caption, wide=wide)


def _fmt(v, col):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '–'
    if col == 'obsdate':
        return str(v).replace('T', ' ')[:19]
    if col in ('SEEING', 'exptime'):
        return f'{float(v):.2f}'
    if isinstance(v, float):
        return f'{v:.3f}'
    return str(v)


def _html(title: str, data: dict) -> str:
    return (HTML_TEMPLATE.replace('__TITLE__', html.escape(title))
            .replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/')))


HTML_TEMPLATE = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1d21;--mute:#667085;--line:#e3e6ea;--acc:#2f6fde;--bad:#c4372f;--hl:#eef3fd}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--card:#1b1e23;--fg:#e6e8eb;--mute:#98a1ae;--line:#2c3138;--acc:#7aa7ff;--bad:#ff7b72;--hl:#22293a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Apple SD Gothic Neo",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:12px 16px 0}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mute);font-size:12px;margin-bottom:10px}
nav{display:flex;gap:2px;overflow-x:auto}nav button{border:0;background:none;color:var(--mute);padding:8px 12px;font:inherit;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap}
nav button.on{color:var(--acc);border-bottom-color:var(--acc);font-weight:600}nav button:disabled{opacity:.35;cursor:default}
main{padding:16px;max-width:1500px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:14px}
.card h2{font-size:14px;margin:0 0 8px}.cap{color:var(--mute);font-size:12px;margin:6px 0 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{color:var(--mute);font-weight:500}
td:nth-child(-n+2),th:nth-child(-n+2){text-align:left}tbody tr{cursor:pointer}tbody tr:hover{background:var(--hl)}
.wrap{overflow-x:auto}.bad{color:var(--bad)}
.bar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.bar select{flex:1;min-width:0;max-width:640px;padding:6px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.bar button{padding:6px 12px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;cursor:pointer}
.fbar{position:sticky;top:var(--hh,100px);z-index:4;background:var(--bg);padding:8px 0;margin-top:-8px}.fbar select{max-width:960px}
.meta{color:var(--mute);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,560px),1fr));gap:14px}
figure{margin:0}figure.wide{grid-column:1/-1;max-width:1400px}figure img{width:100%;height:auto;display:block;border-radius:4px;cursor:zoom-in;background:#fff}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;align-items:center;justify-content:center;z-index:10;cursor:zoom-out}
#lb img{max-width:96vw;max-height:94vh;background:#fff}
.empty{color:var(--mute);padding:24px;text-align:center}
.pl{width:100%;height:auto;display:block;user-select:none}.pl text{fill:var(--mute);font-size:11px}.pl .gr{stroke:var(--line)}.pl .ax{fill:none;stroke:var(--line)}.pl .zr{fill:var(--acc);opacity:.15;pointer-events:none}
.bar button.on{border-color:var(--acc);color:var(--acc);font-weight:600}
.bar button[data-act=tgt]{flex:1 1 0;min-width:4em;max-width:max-content;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}   /* target name: takes what is left of its row (no new row unless < 4em), shortened with "…"; whole name in title */
.bar input[type=number]{width:80px;padding:6px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.bar input[data-crd=ra],.bar input[data-crd=dec]{width:150px;padding:6px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin:0 0 10px}
.chip{display:inline-flex;align-items:center;gap:5px;padding:2px 8px;font:inherit;font-size:12px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:12px;cursor:pointer}
.chip.off{opacity:.35}.chip i{width:10px;height:10px;border-radius:50%}.chip i.sq{border-radius:1px}
.panels{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}.panels h3{font-size:12px;font-weight:500;margin:0}
.ramp{display:inline-block;width:120px;height:8px;border-radius:4px;vertical-align:middle}
.fit{gap:14px}.fit figure.wide{grid-column:auto;max-width:none}   /* stage figures: --rows rows within --av (hh()) */
.fit .cap{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}   /* one line each; whole caption in title */
.fit figure img{max-height:calc((var(--av,80vh) - var(--rows,1)*2.6em - (var(--rows,1) - 1)*14px) / var(--rows,1));object-fit:contain;background:none}
.mid figure:nth-child(odd) img{object-position:100% 50%}.mid figure:nth-child(even) img{object-position:0 50%}   /* 2 columns: figures meet in the middle */
.mid figure:nth-child(odd) .cap{text-align:right}
.fit.zp{max-width:1000px;margin:0 auto}.fit.zp figure img{max-height:none;aspect-ratio:4/3}.zp .cap{text-align:center}   /* ZP: 2x2 of equal 4:3 boxes as wide as the cell (grid <= 1000 px, centred), no one-screen height limit */
.lcgrid{display:grid;grid-template-columns:minmax(320px,38%) minmax(0,1fr);gap:14px;align-items:start}   /* LC / SED: coordinate + image | plot */
.lcl .cap:not(.bad),.lcl #cmsg{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.lcl #cmsg{flex:1 1 100%;min-width:0}   /* one line each; whole text in title */
.lcl #cmsg:empty{display:none}
.lcl .bar input[data-crd=ra],.lcl .bar input[data-crd=dec]{width:110px}   /* RA and Dec on one row in a 320 px column */
@media (max-width:760px){.fit{grid-template-columns:minmax(0,1fr)!important}.fit figure img{max-height:none}.fit.zp figure img{aspect-ratio:auto}.mid figure:nth-child(n) .cap{text-align:left}
  .lcgrid{grid-template-columns:minmax(0,1fr)}}
.fov{width:100%;height:auto;display:block;margin:0 auto;background:#000;border-radius:4px;cursor:crosshair;user-select:none}
.fov .fd circle{fill:none;stroke:#4ade80;stroke-width:.75;opacity:.5;vector-effect:non-scaling-stroke}.fov .pk{fill:none;stroke:#ff4d4f;stroke-width:2;vector-effect:non-scaling-stroke}
#tip{position:fixed;display:none;z-index:9;pointer-events:none;white-space:pre;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.25)}
</style></head><body>
<header><h1 id="t"></h1><div class="sub" id="s"></div><nav id="nav"></nav></header>
<main id="m"></main><div id="lb"><img alt=""></div><div id="tip"></div>
<script>
const D=__DATA__;
const TABS=[['overview','개요'],...D.stages.map(s=>[s,D.labels[s]]),...(D.lcdata||D.srcs?[['lc','광도곡선'],['sed','SED']]:[])];
let tab='overview',idx=0;
const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function fig(f){return `<figure${f.wide?' class="wide"':''}><img loading="lazy" src="${f.thumb}" data-full="${f.src}" alt="">${f.caption?`<p class="cap" title="${esc(f.caption)}">${esc(f.caption)}</p>`:''}</figure>`}
function card(t,figs){return `<div class="card"><h2>${esc(t)}</h2><div class="grid">${figs.map(fig).join('')}</div></div>`}
function overview(){
  const head=D.cols.map(c=>`<th>${esc(c[1])}</th>`).join('');
  const rows=D.images.map((im,i)=>`<tr data-i="${i}"><td>${esc(im.short)}</td>${D.cols.map(c=>`<td>${esc(im.row[c[0]])}</td>`).join('')}<td class="${im.ok?'':'bad'}" style="text-align:left">${esc(im.status)}</td></tr>`).join('');
  let h=`<div class="card"><h2>이미지 ${D.images.length}장</h2><div class="wrap"><table><thead><tr><th>프레임</th>${head}<th style="text-align:left">상태</th></tr></thead><tbody>${rows}</tbody></table></div><p class="cap">행을 누르면 그 이미지의 단계별 그림으로 이동합니다.</p></div>`;
  if(D.run.length)h+=card('실행 추이',D.run);
  if(D.lc.length)h+=card('광도곡선',D.lc);
  return h}
function stage(){ // figures in c columns sized to the window height (.fit); WCS: offsets (~1.6:1) left of the 3x3 cutouts; background: 2 rows
  const im=D.images[idx],figs=(im.stages[tab]||[]),n=figs.length,
    c=tab==='photometry'?(n<3?n:Math.ceil(n/2)):tab==='platesolve'&&n===2?n:1,
    cols=tab==='platesolve'&&n===2?'minmax(0,1.6fr) minmax(0,1fr)':`repeat(${c},minmax(0,1fr))`;
  const opts=D.images.map((x,i)=>`<option value="${i}"${i===idx?' selected':''}>${i+1}. ${esc(x.short)}</option>`).join('');
  return `<div class="bar fbar"><button id="pv">←</button><select id="sel">${opts}</select><button id="nx">→</button><span class="meta">${idx+1} / ${D.images.length}</span></div>
  <div class="meta" style="margin-bottom:10px">${esc(im.name)} — seeing ${esc(im.row.SEEING)}″, ZP ${esc(im.row[D.cols[5][0]])} ± ${esc(im.row[D.cols[6][0]])}, UL5 ${esc(im.row[D.cols[7][0]])} <span class="${im.ok?'':'bad'}">· ${esc(im.status)}</span></div>
  ${n?`<div class="grid fit${c===2&&tab==='platesolve'?' mid':''}${tab==='photometry'?' zp':''}" style="grid-template-columns:${cols};--rows:${Math.ceil(n/c)}">${figs.map(fig).join('')}</div>`:'<div class="empty">이 이미지에는 이 단계의 그림이 없습니다.</div>'}`}
function render(){
  $('#nav').innerHTML=TABS.map(([k,l])=>`<button data-k="${k}" class="${k===tab?'on':''}">${esc(l)}</button>`).join('');
  const lcv=tab==='lc'||tab==='sed';
  $('#m').innerHTML=tab==='overview'?overview():lcv?lcView():stage();
  if(lcv)draw();hh();
  history.replaceState(null,'',tab==='overview'?'#':lcv?`#${tab}`:`#${tab}/${idx+1}`)}
function go(i){idx=(i+D.images.length)%D.images.length;render()}
function hh(){document.documentElement.style.setProperty('--hh',$('header').offsetHeight+'px');   // frame bar sticks below the header
  const g=$('.fit'),pb=parseFloat(getComputedStyle($('main')).paddingBottom);
  if(g)document.documentElement.style.setProperty('--av',Math.max(240,innerHeight-g.getBoundingClientRect().top-scrollY-pb-12)+'px');   // window height below header, frame bar, name line, main padding
  const l=$('.lcl'),s=l&&$('svg[data-fov]');   // LC / SED: --av = height left for the image when the rest of the left column is in the window
  if(s){document.documentElement.style.setProperty('--av',Math.max(160,innerHeight-l.getBoundingClientRect().top-scrollY-(l.offsetHeight-s.getBoundingClientRect().height)-pb-12)+'px');fovView()}}
addEventListener('resize',hh);
document.addEventListener('click',e=>{
  const b=e.target.closest('nav button');if(b){tab=b.dataset.k;render();return}
  const r=e.target.closest('tbody tr');if(r){idx=+r.dataset.i;tab=D.stages.includes('photometry')?'photometry':D.stages[0];render();scrollTo(0,0);return}
  if(e.target.id==='pv')return go(idx-1);if(e.target.id==='nx')return go(idx+1);
  const img=e.target.closest('figure img');if(img){$('#lb img').src=img.dataset.full;$('#lb').style.display='flex';return}
  if(e.target.closest('#lb'))$('#lb').style.display='none'});
document.addEventListener('change',e=>{if(e.target.id==='sel')go(+e.target.value)});
document.addEventListener('keydown',e=>{if(e.key==='Escape')$('#lb').style.display='none';
  if(tab==='overview'||tab==='lc'||tab==='sed'||e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft')go(idx-1);if(e.key==='ArrowRight')go(idx+1)});
// interactive light curve / SED: L = D.lcdata (--lightcurve table) or detections in D.srcs.file matched to a coordinate;
// time binning and combining happen here
let L,FL,PT,FM,WL0,WL1;
const S={bin:(D.lcdata||D.srcs||{}).bin||1,comb:true,ind:false,mode:'abs',off:new Set(),zoom:null,sed:-1},DARK=matchMedia('(prefers-color-scheme:dark)').matches;
let TIPS=[],ZD=null;
const utc=t=>new Date((t-40587)*864e5).toISOString().slice(0,19).replace('T',' ');
const fwl=f=>f.wl==null?'파장 없음':f.src==='name'?`≈${f.wl} nm (이름 추정)`:`${f.wl} nm`;
const wt=p=>1/Math.max(p.e,1e-4)**2;                 // inverse variance; floor keeps MAGERR=0 finite
function fcol(i){const w=FL[i].wl;if(w==null)return 'var(--mute)';const k=WL1>WL0?(w-WL0)/(WL1-WL0):.5;
  return `hsl(${Math.round(270*(1-k))},${DARK?80:70}%,${DARK?62:42}%)`}   // violet (blue end) -> red
function setL(l){L=l;FL=L.filters||[];PT=(L.pts||[]).map(p=>({t:p[0],f:p[1],m:p[2],e:p[3],fr:p[4],tel:p[5],run:p[6],sep:p[7],sat:p[8]}));
  const wls=FL.map(f=>f.wl).filter(w=>w!=null);WL0=Math.min(...wls);WL1=Math.max(...wls);
  FM=FL.map((_,i)=>{let sw=0,sm=0;for(const p of PT)if(p.f===i){sw+=wt(p);sm+=wt(p)*p.m}return sm/sw});  // per-filter mean for dmag
  S.off=new Set();S.zoom=null;S.sed=-1}
function bins(){ // bin = floor((mjd - t0) / width); same bin and filter (any unit) -> inverse-variance mean in magnitudes
  const w=S.bin,g=new Map();
  for(const p of PT){const k=Math.floor((p.t-L.t0)/w)+'|'+p.f;if(!g.has(k))g.set(k,[]);g.get(k).push(p)}
  return [...g.values()].map(a=>{
    const n=a.length;let sw=0,sm=0;for(const p of a){sw+=wt(p);sm+=wt(p)*p.m}
    const mu=a.reduce((s,p)=>s+p.m,0)/n,ew=Math.sqrt(1/sw),es=n>1?Math.sqrt(a.reduce((s,p)=>s+(p.m-mu)**2,0)/(n-1)/n):0;
    return {t:a.reduce((s,p)=>s+p.t,0)/n,f:a[0].f,m:sm/sw,e:n>1?Math.max(ew,es):a[0].e,ew,es,n,a,
            b:Math.floor((a[0].t-L.t0)/w),run:a.some(p=>p.run)}})}
function mark(p,x,y,op,tip,c){return {x,y,e:p.e,c:c||fcol(p.f),sq:FL[p.f].broad,open:!p.run,op,tip}}
const dtxt=d=>d==null?'':` (Δ ${d>=0?'+':''}${d.toFixed(3)})`;
function stxt(a){if(a[0].sep==null)return '';const s=a.map(p=>p.sat).filter(v=>v!=null),n=a.length>1;  // catalog matching only
  return `\n매칭 거리${n?' 최대':''} ${Math.max(...a.map(p=>p.sep)).toFixed(2)}″ · SATDIST${n?' 최소':''} ${s.length?Math.min(...s).toFixed(2):'–'}`}
function ptip(p,d){const f=FL[p.f];
  return `${f.n} · ${fwl(f)}\n${p.m.toFixed(3)} ± ${p.e.toFixed(3)}${dtxt(d)}\n${p.fr}\n${p.tel} · ${utc(p.t)} UTC${p.run?'':'\n(다른 카탈로그)'}${stxt([p])}`}
function ctip(c,d){const f=FL[c.f],w=S.bin;
  return `${f.n} · ${fwl(f)}\n${c.m.toFixed(3)} ± ${c.e.toFixed(3)}${dtxt(d)} · n=${c.n}\n`+
    (c.n>1?`√(1/Σw) ${c.ew.toFixed(3)} · std/√n ${c.es.toFixed(3)}`:`${c.a[0].fr} (프레임 오차)`)+
    `\n${[...new Set(c.a.map(p=>p.tel))].join(', ')}\nbin ${c.b}: ${utc(L.t0+c.b*w)} – ${utc(L.t0+(c.b+1)*w)} UTC${stxt(c.a)}`}
function ext(a,pad){let lo=Math.min(...a),hi=Math.max(...a);if(!(hi>lo)){lo-=.5;hi+=.5}const d=(hi-lo)*pad;return [lo-d,hi+d]}
function ticks(a,b,n){const raw=(b-a)/Math.max(n,2),p=10**Math.floor(Math.log10(raw)),st=[1,2,5,10].map(k=>k*p).find(k=>k>=raw),r=[];
  for(let i=Math.ceil(a/st);i*st<=b;i++)r.push(i*st);r.d=Math.max(0,-Math.floor(Math.log10(st)));return r}
function plot(o){ // SVG scatter with error bars; smaller y on top (magnitudes); o.zoom: drag selects an x range
  const W=o.W,H=o.H,l=o.yl?66:46,r=10,t=8,b=o.xl?36:22,iw=W-l-r,ih=H-t-b,[x0,x1]=o.x;
  const ms=o.marks.filter(k=>k.x>=x0&&k.x<=x1),[y0,y1]=ms.length?ext(ms.flatMap(k=>[k.y-k.e,k.y+k.e]),.05):[0,1];
  const X=v=>(l+(v-x0)/(x1-x0)*iw).toFixed(1),Y=v=>(t+(v-y0)/(y1-y0)*ih).toFixed(1),xt=ticks(x0,x1,iw/100),yt=ticks(y0,y1,ih/40);
  let s=`<svg class="pl" viewBox="0 0 ${W} ${H}"${o.zoom?` data-zoom="${x0},${x1},${l},${iw},${W}"`:''}>`;
  for(const v of xt)s+=`<line class="gr" x1="${X(v)}" x2="${X(v)}" y1="${t}" y2="${t+ih}"/><text x="${X(v)}" y="${t+ih+14}" text-anchor="middle">${v.toFixed(xt.d)}</text>`;
  for(const v of yt)s+=`<line class="gr" x1="${l}" x2="${l+iw}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${l-5}" y="${+Y(v)+4}" text-anchor="end">${v.toFixed(yt.d)}</text>`;
  s+=`<rect class="ax" x="${l}" y="${t}" width="${iw}" height="${ih}"/>`;
  if(o.xl)s+=`<text x="${l+iw/2}" y="${H-4}" text-anchor="middle">${esc(o.xl)}</text>`;
  if(o.yl)s+=`<text transform="rotate(-90)" x="${-(t+ih/2)}" y="13" text-anchor="middle">${esc(o.yl)}</text>`;
  for(const ln of o.lines||[])s+=`<polyline points="${ln.p.map(([a,c])=>X(a)+','+Y(c)).join(' ')}" style="fill:none;stroke:${ln.c};opacity:.6"/>`;
  for(const k of ms){const i=TIPS.push(k.tip)-1,cx=X(k.x),cy=+Y(k.y),st=`stroke:${k.c};stroke-width:1.5;fill:${k.open?'var(--card)':k.c}`;
    s+=`<g data-tip="${i}" style="opacity:${k.op}"><line x1="${cx}" x2="${cx}" y1="${Y(k.y-k.e)}" y2="${Y(k.y+k.e)}" style="stroke:${k.c}"/>`+
      (k.sq?`<rect x="${cx-3.5}" y="${cy-3.5}" width="7" height="7" style="${st}"/>`:`<circle cx="${cx}" cy="${cy}" r="4" style="${st}"/>`)+
      `<circle cx="${cx}" cy="${cy}" r="9" style="fill:transparent"/></g>`}
  return s+`<rect class="zr" x="0" y="${t}" width="0" height="${ih}"/></svg>`}
function lcMarks(fi){ // fi: one filter (grid panel) or null = all shown filters
  const dl=S.mode==='delta',use=p=>!S.off.has(p.f)&&(fi==null||p.f===fi),y=p=>dl?p.m-FM[p.f]:p.m,out=[];
  if(S.ind)for(const p of PT)if(use(p))out.push(mark(p,p.t,y(p),S.comb?.3:1,ptip(p,dl?y(p):null)));
  if(S.comb)for(const c of bins())if(use(c))out.push(mark(c,c.t,y(c),1,ctip(c,dl?y(c):null)));
  return out}
function lcPlot(){
  const ms=lcMarks(null);if(!ms.length)return '<div class="empty">표시할 점이 없습니다 (필터·점 종류 선택 확인).</div>';
  const x=S.zoom||ext(ms.map(k=>k.x),.03);
  if(S.mode!=='grid')return plot({W:1100,H:440,x,marks:ms,xl:'MJD (UTC)',yl:S.mode==='delta'?'Δmag (필터별 가중 평균 기준)':`${L.key} (10″)`,zoom:1});
  return `<div class="panels">${FL.map((f,i)=>{const m=S.off.has(i)?[]:lcMarks(i);
    return m.length?`<div><h3>${esc(f.n)} · ${esc(fwl(f))}</h3>${plot({W:320,H:190,x,marks:m,zoom:1})}</div>`:''}).join('')}</div>`}
function sedPlot(){ // x = filter wavelength; one bin: filter colours, several bins: one hue, light (early) -> dark (late)
  const w=S.bin,cs=bins().filter(c=>!S.off.has(c.f)&&FL[c.f].wl!=null),bs=[...new Set(cs.map(c=>c.b))].sort((a,c)=>a-c);
  const shown=bs.filter(b=>S.sed<0||b===S.sed),one=shown.length<2,
    col=(p,b)=>{if(one)return fcol(p.f);const k=bs.indexOf(b)/(bs.length-1);return `hsl(215,75%,${DARK?35+45*k:72-44*k}%)`};
  const ms=[],lines=[];
  if(S.ind)for(const p of PT){const b=Math.floor((p.t-L.t0)/w);
    if(!S.off.has(p.f)&&FL[p.f].wl!=null&&shown.includes(b))ms.push(mark(p,FL[p.f].wl,p.m,S.comb?.3:1,ptip(p),col(p,b)))}
  if(S.comb)for(const b of shown){const g=cs.filter(c=>c.b===b).sort((a,c)=>FL[a.f].wl-FL[c.f].wl);
    lines.push({c:one?'var(--mute)':col(g[0],b),p:g.map(c=>[FL[c.f].wl,c.m])});for(const c of g)ms.push(mark(c,FL[c.f].wl,c.m,1,ctip(c),col(c,b)))}
  if(!ms.length)return '<div class="empty">표시할 점이 없습니다 (필터·점 종류·bin 선택 확인).</div>';
  let h=plot({W:1100,H:440,x:ext(ms.map(k=>k.x),.04),marks:ms,lines,xl:'투과곡선 응답 가중 평균 파장 (nm)',yl:`${L.key} (10″)`});
  if(!one)h+=`<p class="meta">색: bin 시각 순 <span class="ramp" style="background:linear-gradient(90deg,${col(null,bs[0])},${col(null,bs[bs.length-1])})"></span> ${utc(L.t0+bs[0]*w)} → ${utc(L.t0+bs[bs.length-1]*w)} UTC</p>`;
  return h}
function controls(sed){
  const mb=(v,lab)=>`<button data-act="mode" data-v="${v}" class="${S.mode===v?'on':''}">${lab}</button>`;
  let h='<div class="bar">'+(sed?'':mb('abs','등급')+mb('delta','Δmag')+mb('grid','필터별 패널')+(S.zoom?'<button data-act="unzoom">범위 리셋</button>':''));
  h+=`<label class="meta">bin 폭 <input type="number" min="0" step="any" value="${S.bin}" data-ctl="bin"> 일</label>`+
    `<label class="meta"><input type="checkbox" data-ctl="comb"${S.comb?' checked':''}> 합친 점</label>`+
    `<label class="meta"><input type="checkbox" data-ctl="ind"${S.ind?' checked':''}> 개별 점</label>`;
  if(sed){const bs=[...new Set(bins().map(c=>c.b))].sort((a,c)=>a-c);
    if(bs.length>1)h+=`<select data-ctl="sed"><option value="-1">모든 bin 겹쳐 보기 (${bs.length})</option>${bs.map(b=>`<option value="${b}"${b===S.sed?' selected':''}>bin ${b} · ${utc(L.t0+b*S.bin)} UTC</option>`).join('')}</select>`}
  return h+'</div>'}
function chips(){
  return `<div class="chips"><button class="chip" data-act="all">전체</button><button class="chip" data-act="none">없음</button>${FL.map((f,i)=>
    `<button class="chip${S.off.has(i)?' off':''}" data-act="flt" data-v="${i}" title="${esc(fwl(f)+(f.src&&f.src!=='name'?` · ${f.src}`:''))}"><i class="${f.broad?'sq':''}" style="background:${fcol(i)}"></i>${esc(f.n)}</button>`).join('')}</div>`}
function notes(sed){
  const est=FL.filter(f=>f.src==='name').map(f=>f.n),no=FL.filter(f=>f.wl==null).map(f=>f.n),bad=FL.filter(f=>f.note).map(f=>`${f.n}: ${f.note}`);
  let h=`<p class="cap">${esc(L.key)} (10″) · 점 ${PT.length}개${L.n_bad?`, 제외 ${L.n_bad}개 (MAGERR ≥ 99 또는 NaN)`:''} · bin = floor((MJD − ${L.t0==null?'–':L.t0.toFixed(5)}) / 폭) · `+
    `합친 점 = 같은 bin·필터(유닛 무관)의 등급 역분산 가중 평균, 오차 max(√(1/Σw), std/√n) · 채운 기호: 이번 summary의 이미지, 빈 기호: 다른 카탈로그 · ○ 중대역, □ 광대역`+
    `${sed?'':' · 그림을 가로로 끌면 시간 범위를 확대합니다'}</p>`;
  if(L.cat){const c=L.cat;h+=`<p class="cap">검출 카탈로그 매칭: RA ${c.ra.toFixed(6)}°, Dec ${c.dec.toFixed(6)}° 반경 ${c.r}″ 안에서 프레임마다 가장 가까운 검출 (SNR > ${c.snr}) · `+
    `카탈로그 있는 프레임 ${c.nf}장 중 ${c.nm}장에서 찾음${D.images.length>c.nf?`, 카탈로그 없는 프레임 ${D.images.length-c.nf}장`:''} · `+
    `<span class="${c.n2?'bad':''}">반경 안 검출 2개 이상 ${c.n2}장</span> · SNR ≤ ${c.snr}인 프레임은 빠짐 · target_photometry(카탈로그 사이 군집 선택)와 혼잡한 곳에서 다를 수 있음</p>`}
  if(est.length)h+=`<p class="cap">투과곡선이 없어 이름에서 파장을 추정: ${esc(est.join(', '))}</p>`;
  if(no.length)h+=`<p class="cap bad">파장 없음${sed?' (SED에서 제외)':''}: ${esc(no.join(', '))}</p>`;
  if(bad.length)h+=`<p class="cap bad">투과곡선 읽기 실패: ${esc(bad.join('; '))}</p>`;
  return h}
function lcView(){const l=coordBar()+fovCard(),r=L.error?`<div class="card"><div class="empty bad">광도곡선·SED 데이터를 만들지 못했습니다: ${esc(L.error)}</div></div>`:'<div class="card" id="lcc"></div>';
  return l?`<div class="lcgrid"><div class="lcl">${l}</div><div>${r}</div></div>`:r}   // left: coordinate input + whole-field image, right: plot
function draw(){
  fovView();const c=$('#lcc');if(!c)return;TIPS=[];const sed=tab==='sed';
  if(C.auto){C.auto=false;apply()}           // --ra/--dec without --lightcurve: match on the first opening
  if(C.msg){c.innerHTML=`<div class="empty${C.bad?' bad':''}">${esc(C.msg)}</div>`;return}
  c.innerHTML=controls(sed)+chips()+(PT.length?(sed?sedPlot():lcPlot()):'<div class="empty">표시할 점이 없습니다.</div>')+notes(sed)}
// coordinate input (D.srcs): C keeps the typed text across re-renders; C.mode 'table' = D.lcdata, 'cat' = catalog matching
const T=D.srcs&&D.srcs.target,C={mode:D.lcdata?'table':'cat',ra:T?String(T[0]):'',dec:T?String(T[1]):'',
  r:String((D.srcs&&D.srcs.radius)||5),bad:false,tgt:false,
  msg:D.lcdata?'':D.srcs&&D.srcs.error?'좌표 입력에 쓸 검출 카탈로그가 없습니다.':'RA·Dec를 넣고 좌표 적용(또는 Enter)을 누르세요.'+(D.fov&&!D.fov.error?' 시야 이미지에서 광원을 클릭해도 됩니다.':''),auto:!D.lcdata&&!!T&&!!(D.srcs&&D.srcs.file)};
let SRC=null;                                         // fetch of D.srcs.file, once (again after a failure)
function coordBar(){if(!D.srcs)return '';
  if(D.srcs.error)return `<div class="card"><p class="cap bad">좌표 입력 없음 — 검출 카탈로그 파일을 만들지 못했습니다: ${esc(D.srcs.error)}</p></div>`;
  const t=C.mode==='table',ct=`${t?'지금은 target_photometry 표의 광도곡선·SED입니다. ':''}좌표를 넣으면 프레임마다 검출 카탈로그(SNR > 3)에서 반경 안 가장 가까운 광원으로 그립니다. `+
    `RA·Dec는 도(십진수) 또는 hh:mm:ss / ±dd:mm:ss, RA 칸에 "RA Dec"를 한꺼번에 붙여넣어도 됩니다.`;
  return `<div class="card"><div class="bar" style="margin-bottom:0"><label class="meta">RA <input data-crd="ra" value="${esc(C.ra)}" placeholder="도 또는 hh:mm:ss"></label>`+
    `<label class="meta">Dec <input data-crd="dec" value="${esc(C.dec)}" placeholder="도 또는 ±dd:mm:ss"></label>`+
    `<label class="meta">반경 <input type="number" min="0" step="any" data-crd="r" value="${esc(C.r)}"> ″</label>`+
    `<button data-act="crd">${t?'카탈로그 매칭으로 보기':'좌표 적용'}</button>${D.fov&&!D.fov.error?'':tgtBtn()}`+
    `<span class="bad" id="cmsg" style="font-size:12px"></span></div>`+
    `<p class="cap" title="${esc(ct)}">${esc(ct)}</p></div>`}
function tgtBtn(){return T||D.lcdata?`<button data-act="tgt"${C.mode==='table'||C.tgt?' class="on"':''} title="${esc(D.title)}">${esc(D.title)}</button>`:''}   // in the image card's bar when there is one: no new row in the coordinate bar
function deg(s,h){ // decimal degrees, or h:m:s (RA, h) / ±d:m:s (Dec), ':' or blanks between the fields
  s=s.trim();if(/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(s))return +s;
  const m=s.match(/^([+-]?)(\d+)[:\s]+(\d+)(?:[:\s]+(\d+\.?\d*))?$/);if(!m||+m[3]>=60||+(m[4]||0)>=60)return NaN;
  const v=(+m[2]+m[3]/60+(m[4]||0)/3600)*(h?15:1);return m[1]==='-'?-v:v}
function radec(){ // [ra, dec] in degrees or null; an RA field not readable as one RA but in 2 or 6 fields is "RA Dec": both fields are rewritten
  const t=C.ra.trim().split(/[\s,]+/),one=deg(C.ra,1);
  if(!(one>=0&&one<360)&&(t.length===2||t.length===6)){C.ra=t.slice(0,t.length/2).join(' ');C.dec=t.slice(t.length/2).join(' ');
    for(const k of ['ra','dec']){const el=$(`[data-crd=${k}]`);if(el)el.value=C[k]}}
  const ra=deg(C.ra,1),dec=deg(C.dec,0);return ra>=0&&ra<360&&Math.abs(dec)<=90?[ra,dec]:null}
const sep=(a,b,c,d)=>{const r=Math.PI/180,x=Math.sin((d-b)*r/2),y=Math.sin((c-a)*r/2);   // arcsec, haversine
  return 2*Math.asin(Math.min(1,Math.sqrt(x*x+Math.cos(b*r)*Math.cos(d*r)*y*y)))/r*3600};
function match(F,ra,dec,r){ // per frame the nearest detection within r arcsec -> L in the D.lcdata form (+ sep, SATDIST)
  const pts=[];let o=0,n2=0;
  for(const [i,t,f,n] of F.frames){let b=-1,bs=Infinity,k=0;
    for(let j=o;j<o+n;j++){if(Math.abs(F.dec[j]-dec)*3600>=r)continue;const s=sep(F.ra[j],F.dec[j],ra,dec);if(s<r){k++;if(s<bs){bs=s;b=j}}}
    o+=n;if(k>1)n2++;
    if(b>=0){const im=D.images[i];pts.push([t,f,F.mag[b],F.err[b],im.id,im.row.telname,1,bs,F.sat[b]])}}
  const used=[...new Set(pts.map(p=>p[1]))].sort((a,c)=>a-c),ix=new Map(used.map((f,j)=>[f,j]));   // filters with points only
  for(const p of pts)p[1]=ix.get(p[1]);
  return {key:F.key,t0:F.t0,n_bad:0,filters:used.map(f=>F.filters[f]),pts,cat:{ra,dec,r,snr:F.snr,nf:F.frames.length,nm:pts.length,n2}}}
function apply(){
  const p=radec(),r=parseFloat(C.r),m=$('#cmsg');
  if(!p||!(r>0)){if(m)m.textContent=m.title='좌표를 읽을 수 없습니다: RA는 도 또는 hh:mm:ss, Dec는 도 또는 ±dd:mm:ss, 반경 > 0″';hh();return}   // a wrapped message: left column refitted
  const re=C.mode!=='cat';C.mode='cat';C.msg='검출 카탈로그를 읽는 중…';C.bad=false;
  C.tgt=!!T&&sep(p[0],p[1],T[0],T[1])<0.1&&r===(D.srcs.radius||5);if(re){setL({});render()}else draw();   // C.tgt: matching at the target, default radius
  const tb=$('[data-act=tgt]');if(tb)tb.classList.toggle('on',C.tgt);
  if(!SRC)SRC=fetch(D.srcs.file,{cache:'no-cache'}).then(x=>{if(!x.ok)throw new Error(`HTTP ${x.status}`);return x.json()});
  SRC.then(F=>{if(C.mode!=='cat')return;
    if(F.token!==D.srcs.token)throw new Error('이 페이지와 다른 visualize 실행에서 만든 파일입니다 — 갤러리를 다시 만든 뒤 페이지를 새로고침하세요');
    setL(match(F,p[0],p[1],r));C.msg='';draw()})
  .catch(e=>{SRC=null;if(C.mode!=='cat')return;setL({});C.bad=true;   // fetch, stale file or matching failed: shown, not stuck at loading
    C.msg=`검출 카탈로그(${D.srcs.file})를 쓸 수 없습니다: ${e.message}`+(location.protocol==='file:'?' — file://로 연 페이지는 데이터 파일을 읽을 수 없습니다. http 서버로 여세요.':'');draw()})}
document.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.crd;if(k)C[k]=e.target.value});
document.addEventListener('keydown',e=>{if(e.key==='Enter'&&e.target.dataset&&e.target.dataset.crd)apply()});
document.addEventListener('click',e=>{const a=e.target.closest('[data-act]');if(!a)return;const k=a.dataset.act;
  if(k==='crd')return apply();
  if(k==='tgt'){if(T){C.ra=String(T[0]);C.dec=String(T[1]);C.r=String(D.srcs.radius||5);V.pick=-1}   // fields back to the target
    if(!D.lcdata){render();return apply()}C.mode='table';C.msg='';setL(D.lcdata);return render()}
  if(k==='mode')S.mode=a.dataset.v;else if(k==='unzoom')S.zoom=null;else if(k==='all')S.off.clear();
  else if(k==='none')S.off=new Set(FL.map((_,i)=>i));else if(k==='flt'){const i=+a.dataset.v;S.off.has(i)?S.off.delete(i):S.off.add(i)}
  draw()});
document.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.ctl;if(!k)return;
  if(k==='bin'){const v=parseFloat(e.target.value);S.bin=v>0?v:1;S.sed=-1}else if(k==='sed')S.sed=+e.target.value;else S[k]=e.target.checked;
  draw()});
document.addEventListener('mousemove',e=>{
  if(ZD){const r=ZD.r,k=ZD.W/r.width,z=ZD.s.querySelector('.zr');
    z.setAttribute('x',((Math.min(ZD.a,e.clientX)-r.left)*k).toFixed(1));z.setAttribute('width',(Math.abs(e.clientX-ZD.a)*k).toFixed(1));return}
  const g=e.target.closest&&e.target.closest('[data-tip]'),tp=$('#tip');if(!g){tp.style.display='none';return}
  tp.textContent=TIPS[+g.dataset.tip];tp.style.display='block';
  tp.style.left=Math.min(e.clientX+14,innerWidth-tp.offsetWidth-8)+'px';tp.style.top=Math.min(e.clientY+14,innerHeight-tp.offsetHeight-8)+'px'});
document.addEventListener('mousedown',e=>{const s=e.target.closest&&e.target.closest('svg[data-zoom]');if(!s||e.button)return;
  ZD={s,r:s.getBoundingClientRect(),a:e.clientX,W:+s.dataset.zoom.split(',')[4]};$('#tip').style.display='none';e.preventDefault()});
document.addEventListener('mouseup',e=>{if(!ZD)return;const {s,r,a}=ZD,[x0,x1,l,iw,W]=s.dataset.zoom.split(',').map(Number),k=W/r.width;ZD=null;
  const px=c=>x0+((c-r.left)*k-l)/iw*(x1-x0);
  if(Math.abs(e.clientX-a)>5){S.zoom=[px(Math.min(a,e.clientX)),px(Math.max(a,e.clientX))];draw()}else s.querySelector('.zr').setAttribute('width',0)});
// whole field of one frame (D.fov): wheel zooms, drag pans, a click (moved ≤ 5 px) takes the nearest detection within
// 20 screen px into the coordinate fields and applies it; V.box = viewBox in image px, kept across re-renders
const V={on:true,box:null,pick:-1};let VD=null;
function fovCard(){const F=D.fov;if(!F)return '';const nm=`${F.frame+1}. ${esc(D.images[F.frame].short)}`;
  if(F.error)return `<div class="card"><p class="cap bad">시야 이미지 없음 (대표 프레임 ${nm}): ${esc(F.error)}</p></div>`;
  const b=V.box||[0,0,F.w,F.h],p=V.pick,ft=`대표 프레임 ${F.frame+1}. ${D.images[F.frame].short} — SNR > 3 검출이 가장 많은 프레임 (${F.u.length}개, 초록 원) · `+
    `휠: 확대·축소 · 끌기: 이동 · 클릭: 화면에서 20 px 안의 가장 가까운 검출의 RA·Dec로 좌표 적용 (빨간 원) · 다른 유닛·필터의 프레임과 시야가 다를 수 있습니다`;
  return `<div class="card"><div class="bar"><label class="meta"><input type="checkbox" data-fov="on"${V.on?' checked':''}> 검출 원</label>`+
    `<button data-fov="reset">전체 시야</button>${tgtBtn()}<span class="bad" id="fmsg" style="font-size:12px"></span></div>`+
    `<svg class="fov" data-fov="img" viewBox="${b.join(' ')}" style="width:min(100%,calc(var(--av,60vh)*${(F.w/F.h).toFixed(4)}))"><image href="${esc(F.file)}?${encodeURIComponent(D.srcs.token)}" width="${F.w}" height="${F.h}"/>`+
    `<g class="fd"${V.on?'':' style="display:none"'}>${F.u.map((u,j)=>`<circle cx="${u}" cy="${F.v[j]}" r="1"/>`).join('')}</g>`+
    `<circle class="pk" cx="${p<0?0:F.u[p]}" cy="${p<0?0:F.v[p]}" r="1"${p<0?' style="display:none"':''}/></svg>`+
    `<p class="cap" title="${esc(ft)}">대표 프레임 ${F.frame+1}. ${esc(D.images[F.frame].short.split(': ')[0])} · 검출 ${F.u.length}개 · 휠 확대 · 끌기 이동 · 클릭 선택</p></div>`}
function fovView(){const s=$('svg[data-fov]');if(!s)return;const F=D.fov,b=V.box||[0,0,F.w,F.h],k=b[2]/(s.getBoundingClientRect().width||F.w);
  s.setAttribute('viewBox',b.join(' '));for(const c of s.querySelectorAll('circle'))c.setAttribute('r',((c.classList.contains('pk')?9:4)*k).toFixed(2))}  // radius in screen px
function fovSet(x,y,w){const F=D.fov,h=w*F.h/F.w;V.box=[Math.min(Math.max(x,0),F.w-w),Math.min(Math.max(y,0),F.h-h),w,h];fovView()}
document.addEventListener('wheel',e=>{const s=e.target.closest&&e.target.closest('svg[data-fov]');if(!s)return;if(!e.deltaY)return;e.preventDefault();
  const F=D.fov,r=s.getBoundingClientRect(),b=V.box||[0,0,F.w,F.h],w=Math.min(Math.max(b[2]*(e.deltaY>0?1.1:1/1.1),F.w/40),F.w),
    ox=e.clientX-r.left,oy=e.clientY-r.top;
  fovSet(b[0]+ox*(b[2]-w)/r.width,b[1]+oy*(b[2]-w)/r.width,w)},{passive:false});   // the image point under the cursor stays
document.addEventListener('mousedown',e=>{const s=e.target.closest&&e.target.closest('svg[data-fov]');if(!s||e.button)return;
  VD={r:s.getBoundingClientRect(),x:e.clientX,y:e.clientY,b:V.box||[0,0,D.fov.w,D.fov.h],moved:false};e.preventDefault()});
document.addEventListener('mousemove',e=>{if(!VD)return;const dx=e.clientX-VD.x,dy=e.clientY-VD.y;
  if(!VD.moved&&Math.hypot(dx,dy)<=5)return;VD.moved=true;const k=VD.b[2]/VD.r.width;fovSet(VD.b[0]-dx*k,VD.b[1]-dy*k,VD.b[2])});
document.addEventListener('mouseup',e=>{if(!VD)return;const {r,b,x,y,moved}=VD;VD=null;if(moved||Math.hypot(e.clientX-x,e.clientY-y)>5)return;
  const F=D.fov,k=b[2]/r.width,ix=b[0]+(e.clientX-r.left)*k,iy=b[1]+(e.clientY-r.top)*k,m=$('#fmsg');let j=-1,d=Infinity;
  for(let i=0;i<F.u.length;i++){const q=(F.u[i]-ix)**2+(F.v[i]-iy)**2;if(q<d){d=q;j=i}}
  if(j<0||Math.sqrt(d)/k>20){if(m)m.textContent='근처에 검출 없음';hh();return}
  if(m)m.textContent='';hh();V.pick=j;C.ra=String(F.ra[j]);C.dec=String(F.dec[j]);
  for(const c of ['ra','dec']){const el=$(`[data-crd=${c}]`);if(el)el.value=C[c]}
  const pk=$('svg[data-fov] .pk');if(pk){pk.setAttribute('cx',F.u[j]);pk.setAttribute('cy',F.v[j]);pk.style.display=''}
  apply()});   // table mode: apply() switches to catalog matching, as the "카탈로그 매칭으로 보기" button
document.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.crd;if((k==='ra'||k==='dec')&&V.pick>=0){   // typed: no picked detection
  V.pick=-1;const pk=$('svg[data-fov] .pk');if(pk)pk.style.display='none'}});
document.addEventListener('change',e=>{if(!(e.target.dataset&&e.target.dataset.fov==='on'))return;V.on=e.target.checked;
  const g=$('svg[data-fov] .fd');if(g)g.style.display=V.on?'':'none'});
document.addEventListener('click',e=>{if(e.target.closest('[data-fov=reset]')){V.box=null;fovView()}});
addEventListener('resize',fovView);
$('#t').textContent=D.title;$('#s').textContent=D.sub;
const m=location.hash.match(/^#(\w+)\/(\d+)/);if(m&&D.stages.includes(m[1])){tab=m[1];idx=Math.min(+m[2]-1,D.images.length-1)}
if((D.lcdata||D.srcs)&&/^#(lc|sed)$/.test(location.hash))tab=location.hash.slice(1);
setL(D.lcdata||{});render();
</script></body></html>'''


def _objname(path: Path, info: dict = None) -> str:
    """OBJNAME from .info (info: already read), else the scidata layout .../<objname>/<telname>/<filter>/<file>."""
    return (_info(path) if info is None else info).get('OBJNAME') or path.parents[2].name


def _stage_inputs(stage: str, path: Path, info: dict) -> dict:
    """[mtime_ns, size] (None: missing) of the files a stage figure is drawn from, by role - no paths, DONE_FILE is
    served with the gallery. The sky-catalog archive (platesolve) is not an input."""
    files = dict(
        preprocess=dict(image=path, info=_product(path, '.info'),
                        **{k[:-4].lower(): info.get(k) for k in ('BIASPATH', 'DARKPATH', 'FLATPATH')}),
        platesolve=dict(image=path, cat=_product(path, '.cat'), info=_product(path, '.info')),
        mask=dict(image=path, srcmask=_product(path, '.srcmask'), invalidmask=_product(path, '.invalidmask')),
        background={s: _product(path, s) for s in ('.bkgmap.png', '.bkgrms.png')},
        photometry={s: _product(path, s) for s in ('.refcat.png', '.zp_mag.png', '.zp_color.png', '.zp_2d.png')})[stage]
    out = {}
    for role, f in files.items():
        try:
            st = os.stat(f) if f else None
            out[role] = [st.st_mtime_ns, st.st_size] if st else None
        except FileNotFoundError:
            out[role] = None
    return out


def _expected(stage: str, path: Path, info: dict) -> List[str]:
    """Figure names a stage draws when its inputs exist - for --pages-only in a frame folder without a stage record."""
    if stage == 'preprocess':
        ok = all(p and Path(p).exists() for p in (info.get(k) for k in ('BIASPATH', 'DARKPATH', 'FLATPATH')))
        return ['preprocess.png'] if ok else []
    if stage == 'platesolve':
        return ['wcs_offsets.png', 'wcs_cutouts.png'] if _product(path, '.cat').exists() else []
    if stage == 'mask':
        return ['mask.png'] if any(_product(path, s).exists() for s in ('.srcmask', '.invalidmask')) else []
    sfx = ('.bkgmap.png', '.bkgrms.png') if stage == 'background' else ('.refcat.png', '.zp_mag.png', '.zp_color.png', '.zp_2d.png')
    return [_product(path, s).name for s in sfx if _product(path, s).exists()]


def _read_done(imgdir: Path) -> dict:
    try:
        return json.loads((imgdir / DONE_FILE).read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _write_done(imgdir: Path, done: dict):
    tmp = imgdir / (DONE_FILE + '.tmp')
    tmp.write_text(json.dumps(done, indent=1))
    os.replace(tmp, imgdir / DONE_FILE)            # atomic: a reader sees the old or the new record


def _frame(args):
    """Stage figures of one summary row under out/<stem>/; return its gallery entry, manifest item
    ({} when the file is missing), detections (_sources, None without them) and the stages not drawn again.
    mode 'all': draw every stage; 'skip': keep a stage whose DONE_FILE record (this FIG_REV) has the same inputs
    (_stage_inputs) and whose figures are all there; 'pages': draw nothing, list the figures on disk (a stage with one
    missing is left empty and noted in the status). 'all' and 'skip' rewrite DONE_FILE."""
    global _FRAME_CATS
    row, out, stages, info, mode = args
    path = Path(row['path'])
    status = 'ok' if bool(row['success']) else f"FAILED at {row['step']}: {row['error']}"
    raw = row.get('raw_path')                      # raw input file name (summary raw_path), else the summary path
    raw = path.name if raw is None or pd.isna(raw) or str(raw).strip() in ('', 'nan') else Path(str(raw)).name
    short = '|'.join(_fmt(row.get(c), c) for c in ('filter', 'telname', 'obsdate')) + f': {raw}'
    entry = dict(id=path.stem, name=path.name, short=short, ok=bool(row['success']), status=status, stages={},
                 row={c: _fmt(row.get(c), c) for c, _ in SUMMARY_COLS})
    if not path.exists():
        entry['status'] += ' (file missing)'
        entry['ok'] = False
        return entry, {}, None, []
    imgdir = out / path.stem
    imgdir.mkdir(exist_ok=True)
    data, header = fits.getdata(path, header=True)  # memory map: pixels are read only by the figures drawn (not by pages)
    data = _Cutouts(data)                           # cutouts converted on access; whole image in fig_preprocess, fig_mask
    info = _info(path) if info is None else info
    done = _read_done(imgdir)
    fresh = done.get('rev') == FIG_REV              # records of this figure revision only
    rec = dict(done.get('stages', {})) if fresh else {}
    kept = []
    _FRAME_CATS = {}
    try:
        for stage in (stages or _auto_stages(path, header, info)):
            old = rec.get(stage)
            sig = _stage_inputs(stage, path, info)
            same = bool(old) and old.get('inputs') == sig   # recorded inputs = current: its thumbnails may be kept
            if mode == 'pages':
                want = old['names'] if old else _expected(stage, path, info)
                names = want if all((imgdir / n).exists() for n in want) else []
                if len(names) < len(want):
                    entry['status'] += f' | {stage} figure missing'
                    entry['ok'] = False
                kept.append(stage)
            else:
                if (mode == 'skip' and same
                        and all((imgdir / n).exists() for n in old['names'])):
                    names = old['names']
                    kept.append(stage)
                else:
                    t0, same = time.time(), False         # redrawn: new thumbnails
                    try:
                        names = globals()[f'fig_{stage}'](path, data, header, info, imgdir)
                        rec[stage] = dict(t0=round(t0, 3), inputs=sig, names=names)
                    except Exception as e:
                        names = []
                        rec.pop(stage, None)
                        entry['status'] += f' | {stage} figure failed: {type(e).__name__}: {e}'
                        entry['ok'] = False
            entry['stages'][stage] = [_figure(imgdir, n, f'{path.stem}/', reuse=mode != 'all' and same) for n in names]
        if mode != 'pages':
            _write_done(imgdir, dict(rev=FIG_REV, stages=rec))
        try:
            src = _sources(path, row, info)
        except Exception as e:
            src = None
            entry['status'] += f' | detections for coordinate input failed: {type(e).__name__}: {e}'
            entry['ok'] = False
    finally:
        _FRAME_CATS = None
    return entry, {str(path): {k: [f['src'] for f in v] for k, v in entry['stages'].items()}}, src, kept


def _frames(tasks: list, n_proc: int):
    """_frame results in task order. In parallel the first frame runs here alone, so that its platesolve figure
    fills the sky-catalog cache that the workers then read (frames of other fields may still query at once) and the
    process caches (_CACHE: catalog tiles, masters) that the forked workers start with;
    frames sharing an output directory (same file stem) are drawn one after the other."""
    stems = [Path(t[0]['path']).stem for t in tasks]
    if n_proc > 1 and len(tasks) > 1 and len(set(stems)) == len(stems):
        yield _frame(tasks[0])
        with ProcessPoolExecutor(max_workers=min(n_proc, len(tasks) - 1)) as ex:
            yield from ex.map(_frame, tasks[1:])
    else:
        yield from map(_frame, tasks)


def _frame_order(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows in gallery order: medium-band filters (m###, m###w) first, then the others (broad band), each by filter
    wavelength (<filter>.csv in SYNPHOT_FILTERDIR as in _lc_payload, else the m### name; unknown wavelengths last,
    by name), then telname, then obsdate. Stable: ties keep the summary order."""
    from ezphot.configuration import Configuration
    filterdir = Path(Configuration(_silent=True).config['SYNPHOT_FILTERDIR'])
    filt = rows['filter'].fillna('').astype(str)
    wls = {}
    for f in filt.unique():
        try:
            wl = _filter_wavelength(filterdir, '', f)[0]
        except Exception as e:
            wl = None
            print(f"[visualize] transmission curve of {f} unreadable: {type(e).__name__}: {e}")
        m = re.fullmatch(r'm(\d{3})w?', f)
        wls[f] = wl if wl is not None else float(m.group(1)) if m else np.inf
    medium = filt.str.fullmatch(r'm\d{3}w?')
    key = pd.DataFrame(dict(b=(~medium).to_numpy(), w=filt.map(wls).to_numpy(float), f=filt.to_numpy(),
                            u=rows['telname'].to_numpy(), t=pd.to_datetime(rows['obsdate'], errors='coerce').to_numpy()))
    return rows.iloc[key.sort_values(list(key), kind='stable', na_position='last').index]


def _gallery(rows: pd.DataFrame, lc_df: Optional[pd.DataFrame], out: Path, title: str, source: str,
             ra=None, dec=None, stages=None, verbose=True, n_proc=8, infos=None, mode='all') -> Path:
    out.mkdir(parents=True, exist_ok=True)
    rows = _frame_order(rows)
    images, manifest, seen, srcs = [], {}, [], []
    tasks = [(row, out, stages, (infos or {}).get(str(row['path'])), mode) for _, row in rows.iterrows()]
    for entry, item, src, kept in _frames(tasks, n_proc):
        images.append(entry)
        srcs.append(src)
        seen += [s for s in entry['stages'] if s not in seen]
        if verbose and item:
            print(f"[visualize] {entry['name']}: " + ', '.join(f'{k}({len(v)})' for k, v in entry['stages'].items())
                  + (f"; not redrawn: {', '.join(kept)}" if kept else ''))
        manifest.update(item)

    run = [_figure(out, n) for n in fig_run(rows, out)]
    lc = [_figure(out, n) for n in fig_lightcurve(lc_df, out, rows['path'])] if lc_df is not None else []
    lcdata = None
    if lc_df is not None:
        try:
            lcdata = _lc_payload(lc_df, rows['path'], verbose)
        except Exception as e:                     # shown in the light-curve / SED tabs
            lcdata = dict(error=re.sub(r'(?<![\w.~])/[^\s\'"]+', '<path>', f'{type(e).__name__}: {e}'))   # no paths in the page
            print(f"[visualize] interactive light curve failed: {lcdata['error']}")
    filters = ', '.join(sorted(rows['filter'].dropna().astype(str).unique()))
    tels = ', '.join(sorted(rows['telname'].dropna().astype(str).unique()))
    n_ok = int(rows['success'].astype(bool).sum())
    data = dict(title=title, sub=f'{source} · {n_ok}/{len(rows)} 성공 · 필터 {filters} · {tels}',
                stages=[s for s in STAGES if s in seen], labels=STAGE_LABELS, cols=SUMMARY_COLS,
                images=images, run=run, lc=lc)
    if lcdata is not None:
        data['lcdata'] = lcdata
    if any(s is not None for s in srcs):           # coordinate input in the light-curve / SED tabs
        try:
            token = f'{len(images)} {pd.Timestamp.now().isoformat()}'   # in both files: the page detects a stale sources.json
            (out / SOURCES_FILE).write_text(json.dumps(dict(_sources_payload(srcs, verbose), token=token),
                                                       separators=(',', ':'), allow_nan=False))
            data['srcs'] = dict(file=SOURCES_FILE, token=token, bin=LC_BIN_DAYS, radius=SOURCES_RADIUS_ARCSEC,
                                target=[float(ra), float(dec)] if ra is not None and dec is not None else None)
        except Exception as e:                     # shown in the light-curve / SED tabs
            data['srcs'] = dict(error=re.sub(r'(?<![\w.~])/[^\s\'"]+', '<path>', f'{type(e).__name__}: {e}'))
            print(f"[visualize] detections for coordinate input failed: {data['srcs']['error']}")
        if 'file' in data['srcs']:                 # click input: whole field of the frame with the most detections
            i = int(np.argmax([-1 if s is None else len(s['ra']) for s in srcs]))   # ties: first in gallery order
            try:
                data['fov'] = dict(_fov(Path(rows.iloc[i]['path']), srcs[i], out), frame=i)
                if verbose:
                    print(f"[visualize] click input: {FOV_FILE} {data['fov']['w']} x {data['fov']['h']} px of "
                          f"{images[i]['name']} ({len(srcs[i]['ra'])} detections)")
            except Exception as e:                 # shown in the light-curve / SED tabs
                data['fov'] = dict(error=re.sub(r'(?<![\w.~])/[^\s\'"]+', '<path>', f'{type(e).__name__}: {e}'), frame=i)
                print(f"[visualize] whole-field image for the click input failed: {data['fov']['error']}")
    (out / 'manifest.json').write_text(json.dumps(dict(run=[f['src'] for f in run], lightcurve=[f['src'] for f in lc],
                                                       images=manifest), indent=1))
    index = out / 'index.html'
    index.write_text(_html(title, data))
    if verbose:
        print(f"[visualize] gallery -> {index}")
    return index


def visualize(summary: pd.DataFrame | str,
              ra: float = None,
              dec: float = None,
              lightcurve: pd.DataFrame | str = None,
              stages: List[str] = None,
              max_images: int = None,
              root: str | Path = None,
              verbose: bool = True,
              n_proc: int = 8,
              skip_unchanged: bool = False,
              pages_only: bool = False) -> List[Path]:
    """Write per-stage figures and index.html for a process_images summary, one gallery per objname
    at <root>/ezphot/<objname>/viz/ (a later run of the same objname replaces the page); return the index paths.
    The images of a gallery are drawn in n_proc processes (1 = one after the other).
    skip_unchanged: a stage figure of a frame is not drawn again when its inputs (file mtime and size: image, .info,
    .cat, masks, masters, linked ezphot figures; not the sky-catalog archive) are those recorded in the frame folder's
    DONE_FILE by an earlier run of the same FIG_REV and its figures are all there. pages_only: no stage figure is drawn;
    the pages are rebuilt from the figures on disk. Every run that draws (re)writes DONE_FILE; pages_only does not."""
    if skip_unchanged and pages_only:
        raise ValueError('skip_unchanged and pages_only exclude each other')
    mode = 'pages' if pages_only else 'skip' if skip_unchanged else 'all'
    _CACHE.clear()                                  # nothing from an earlier call (also cleared on return)
    source = 'summary (DataFrame)'
    if isinstance(summary, (str, Path)):
        source = Path(summary).expanduser()
        summary = pd.read_csv(source)
        source = source.name
    root = Path(root).expanduser() if root else VIZ_ROOT
    rows = summary.head(max_images) if max_images else summary
    lc_all = None
    if lightcurve is not None:
        lc_all = pd.read_csv(lightcurve) if isinstance(lightcurve, (str, Path)) else lightcurve

    infos = {str(p): _info(Path(p)) for p in rows['path']}   # read once: objname here, the figures in _frame
    objnames = rows['path'].map(lambda p: _objname(Path(p), infos[str(p)]))
    indices = []
    try:
        for obj, g in rows.groupby(objnames):
            lc_df = lc_all
            if lc_all is not None and 'objname' in lc_all:
                lc_df = lc_all[lc_all['objname'] == obj]
            indices.append(_gallery(g, lc_df, root / 'ezphot' / obj / 'viz', obj, source,
                                    ra=ra, dec=dec, stages=stages, verbose=verbose, n_proc=n_proc, infos=infos, mode=mode))
    finally:
        _CACHE.clear()                              # no master set or tile left in the calling process
    return indices


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('summary')
    ap.add_argument('--ra', type=float); ap.add_argument('--dec', type=float)
    ap.add_argument('--lightcurve', help='target_photometry table (CSV)')
    ap.add_argument('--stages', nargs='+', choices=STAGES, help='override the stages found on disk')
    ap.add_argument('--max-images', type=int)
    ap.add_argument('--root', help=f'gallery root served over http (default {VIZ_ROOT})')
    ap.add_argument('--n-proc', type=int, default=8)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--skip-unchanged', action='store_true',
                   help=f'do not redraw a stage figure whose inputs did not change since it was drawn ({DONE_FILE})')
    g.add_argument('--pages-only', action='store_true', help='draw no stage figure; rebuild the pages from the figures on disk')
    a = ap.parse_args(argv)
    visualize(a.summary, ra=a.ra, dec=a.dec, lightcurve=a.lightcurve, stages=a.stages,
              max_images=a.max_images, root=a.root, n_proc=a.n_proc,
              skip_unchanged=a.skip_unchanged, pages_only=a.pages_only)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
