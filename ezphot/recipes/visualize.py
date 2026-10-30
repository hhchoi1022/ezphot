#!/usr/bin/env python
"""
visualize: per-stage figures of a process_images run, collected into a static HTML gallery
for looking at the products by eye. Read-only on the products; writes only under --out.

A stage is shown when its products are on disk (or when it is named in --stages):
    preprocess  header BIASCOR/DARKCOR/FLATCOR + master paths in .info -> raw (rebuilt) vs corrected, masters
    platesolve  SCAMP PV1_* in header -> detected-vs-reference offsets over the field, 3x3 cutouts
    mask        .invalidmask / .srcmask
    background  .bkgmap / .bkgrms
    photometry  .cat + ZP_APER_3 -> ZP figures written by ezphot, target cutout with the 10" aperture
    run         summary-level ZP / seeing / depth vs time (always, >1 image)
    lightcurve  only with --lightcurve (target_photometry table): lightcurve.png when >1 epoch, and
                interactive light-curve / SED tabs of MAGSKY_APER_3 (10") from 1 row on (a table with no usable
                row gives the tabs with a note): frames are combined in the page per time bin and filter
                (default 1 day, width changeable); filter wavelength = response-weighted mean of the transmission
                curve in SYNPHOT_FILTERDIR (<observatory>_<filter>.csv, then <filter>.csv)

    python -m ezphot.recipes.visualize summary.csv [--ra deg --dec deg] [--lightcurve lc.csv] \
        [--stages platesolve photometry] [--max-images N] [--root DIR]

Gallery: <root>/ezphot/<objname>/viz/index.html (root default ~/viz); a later run of the same
objname replaces the page. Serve with
    python -m http.server 8765 --bind 127.0.0.1 --directory ~/viz
    -> http://127.0.0.1:8765/ezphot/<objname>/viz/
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
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
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval, ZScaleInterval
from astropy.wcs import WCS

from ezphot.recipes import defaults as D

STAGES = ('preprocess', 'platesolve', 'mask', 'background', 'photometry')
VIZ_ROOT = Path.home() / 'viz'
APER10_DIAMETER_ARCSEC = 10
ZP_KEY = f'ZP_{D.APER10_KEY}'
MAG_KEY, MAGERR_KEY, MAGSKY_KEY = f'MAG_{D.APER10_KEY}', f'MAGERR_{D.APER10_KEY}', f'MAGSKY_{D.APER10_KEY}'
LC_BIN_DAYS = 1                                    # default time bin of the interactive light-curve / SED tabs


# ---------------------------------------------------------------- helpers
def _bin(data: np.ndarray, f: int, func=np.nanmean) -> np.ndarray:
    ny, nx = (data.shape[0] // f) * f, (data.shape[1] // f) * f
    return func(data[:ny, :nx].reshape(ny // f, f, nx // f, f), axis=(1, 3))


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
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)
    return dst.name


def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return path.name


def _read_cat(path: Path) -> Table:
    return Table.read(str(path), format='ascii.basic')


# ---------------------------------------------------------------- stage figures
def fig_preprocess(path: Path, data, header, info, outdir: Path) -> List[str]:
    paths = [info.get(k) for k in ('BIASPATH', 'DARKPATH', 'FLATPATH')]
    if not all(p and Path(p).exists() for p in paths):
        return []
    bias, dark, flat = (fits.getdata(p).astype(np.float32) for p in paths)
    dark_scale = header['EXPTIME'] / fits.getheader(paths[1])['EXPTIME']
    flat_norm = flat / np.nanmean(flat)            # ccdproc.flat_correct divides by flat / mean(flat)
    raw = data * flat_norm + dark * dark_scale + bias
    f = 8
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    _show(axes[0, 0], _bin(raw, f), 'raw (rebuilt = corr*flat + dark + bias)')
    _show(axes[0, 1], _bin(data, f), 'corrected')
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
    from ezphot.skycatalog import SkyCatalogUtility
    cat_path = _product(path, '.cat')
    if not cat_path.exists():
        return []
    w = _wcs(header)
    ny, nx = data.shape
    centre = w.pixel_to_world(nx / 2, ny / 2)
    util = SkyCatalogUtility()
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

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    ax = axes[0]
    q = ax.quiver(x, y, dra, ddec, np.hypot(dra, ddec), cmap='viridis', angles='xy',
                  scale_units='xy', scale=1 / 2000, width=0.003)
    ax.quiverkey(q, 0.05, -0.12, 0.5, '0.5"', labelpos='E')
    ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_aspect('equal')
    ax.set_title(f'detected -> reference offset (x2000), N={ok.sum()}', fontsize=9)
    fig.colorbar(q, ax=ax, label='|offset| [arcsec]')
    ax = axes[1]
    ax.scatter(r, np.hypot(dra, ddec), s=4, alpha=0.6)
    ax.set_xlabel('distance from centre [arcmin]'); ax.set_ylabel('|offset| [arcsec]')
    ax.set_title(f'median {np.median(np.hypot(dra, ddec)):.2f}"', fontsize=9)
    ax = axes[2]
    ax.hist(dra, bins=40, histtype='step', label=f'dRA* med {np.median(dra):+.2f}"')
    ax.hist(ddec, bins=40, histtype='step', label=f'dDec med {np.median(ddec):+.2f}"')
    ax.set_xlabel('arcsec'); ax.legend(fontsize=8)
    fig.suptitle(f'{path.name}  CTYPE={header["CTYPE1"]}  PV1 terms={sum(k.startswith("PV1_") for k in header)}', fontsize=10)
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
    names = []
    ivp = _product(path, '.invalidmask')
    if ivp.exists():
        m = fits.getdata(ivp).astype(np.float32)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.imshow(_bin(m, 8, np.max), origin='lower', cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(f'invalid-pixel mask, 8x8 blocks with any masked pixel  ({100 * np.mean(m > 0):.3f}% masked)', fontsize=9)
        names.append(_save(fig, outdir / 'invalidmask.png'))
    names += [n for n in [_link(_product(path, '.srcmask.png'), outdir)] if n]
    return names


def fig_background(path: Path, data, header, info, outdir: Path) -> List[str]:
    return [n for n in (_link(_product(path, s), outdir) for s in ('.bkgmap.png', '.bkgrms.png')) if n]


def fig_photometry(path: Path, data, header, info, outdir: Path, ra=None, dec=None) -> List[str]:
    names = [n for n in (_link(_product(path, s), outdir)
                         for s in ('.refcat.png', '.zp_mag.png', '.zp_color.png', '.zp_2d.png')) if n]
    if ra is None or dec is None:
        return names
    w = _wcs(header)
    tx, ty = w.world_to_pixel(SkyCoord(ra, dec, unit='deg'))
    scale = info.get('PIXELSCALE', 0.505)
    half = int(30 / scale)
    x0, y0 = int(tx) - half, int(ty) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half > data.shape[1] or y0 + 2 * half > data.shape[0]:
        return names
    cat = _read_cat(_product(path, '.cat'))
    sep = SkyCoord(cat['X_WORLD'], cat['Y_WORLD'], unit='deg').separation(SkyCoord(ra, dec, unit='deg'))
    k = int(np.argmin(sep))
    fig, ax = plt.subplots(figsize=(6, 6))
    cut = data[y0:y0 + 2 * half, x0:x0 + 2 * half]
    ax.imshow(cut, origin='lower', cmap='gray',
              norm=ImageNormalize(cut, interval=PercentileInterval(99.9), stretch=AsinhStretch(0.02)))
    rad =APER10_DIAMETER_ARCSEC / 2 / scale
    ax.add_patch(plt.Circle((cat['X_IMAGE'][k] - 1 - x0, cat['Y_IMAGE'][k] - 1 - y0), rad, fill=False, color='lime', lw=1))
    ax.plot(tx - x0, ty - y0, '+', color='r', ms=14)
    msky = cat[MAGSKY_KEY][k] if MAGSKY_KEY in cat.colnames else np.nan
    ax.set_title(f'target (+) / nearest source sep {sep[k].arcsec:.1f}"  10" aperture (green)\n'
                 f'{MAGSKY_KEY}={msky:.3f}  {ZP_KEY}={header.get(ZP_KEY, np.nan):.3f}  SEEING={header.get("SEEING", np.nan):.2f}"',
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    names.insert(0, _save(fig, outdir / 'target_cutout.png'))
    return names


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


STAGE_LABELS = dict(preprocess='전처리', platesolve='WCS', mask='마스크', background='배경', photometry='측광')
CAPTIONS = {
    'preprocess.png': '보정 전(master로 재구성) / 보정 후, 픽셀 히스토그램, 사용된 master bias·dark·flat',
    'wcs_offsets.png': '검출 위치 → 기준성 위치 오프셋: 시야 분포, 중심 거리 의존성, 히스토그램',
    'wcs_cutouts.png': '헤더 WCS로 투영한 기준성 위치(빨강), 시야 9곳 4′×4′',
    'invalidmask.png': 'invalid pixel mask (8×8 블록 중 하나라도 마스크된 곳)',
    '.srcmask.png': 'source mask',
    '.bkgmap.png': '2D 배경 맵',
    '.bkgrms.png': '배경 rms 맵',
    '.refcat.png': 'ZP 계산에 쓴 기준성 선택',
    '.zp_mag.png': 'ZP vs 기준성 등급',
    '.zp_color.png': 'ZP vs 색',
    '.zp_2d.png': 'ZP 공간 분포',
    'target_cutout.png': '타깃 주변 60″, 10″ aperture(초록)',
    'run_trends.png': f'{ZP_KEY}, SEEING, UL5SKY (10″) 시간 추이',
    'lightcurve.png': f'{MAGSKY_KEY} 광도곡선 — 채운 원: 이번 summary의 이미지, 빈 원: 디스크의 다른 카탈로그',
}
SUMMARY_COLS = [('filter', '필터'), ('telname', '망원경'), ('obsdate', '관측 시각 (UT)'), ('exptime', '노출'),
                ('SEEING', 'seeing'), (ZP_KEY, 'ZP'), (f'ZPERR_{D.APER10_KEY}', 'ZP err'),
                (f'UL5SKY_{D.APER10_KEY}', 'UL5 (10″)'), ('n_ref', 'n_ref')]


def _figure(outdir: Path, name: str, prefix: str = '') -> dict:
    """Gallery entry with a JPEG thumbnail; the full PNG opens on click."""
    from PIL import Image
    thumb = f'thumb_{Path(name).stem}.jpg'
    with Image.open(outdir / name) as im:
        im.thumbnail((1400, 1400))
        bg = Image.new('RGB', im.size, 'white')
        bg.paste(im, mask=im.getchannel('A') if im.mode == 'RGBA' else None)
        bg.save(outdir / thumb, quality=85)
        wide = im.size[0] > 1.8 * im.size[1]              # multi-panel strips span the full row
    caption = next((v for k, v in CAPTIONS.items() if name.endswith(k)), '')
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
.meta{color:var(--mute);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,560px),1fr));gap:14px}
figure{margin:0}figure.wide{grid-column:1/-1;max-width:1400px}figure img{width:100%;height:auto;display:block;border-radius:4px;cursor:zoom-in;background:#fff}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;align-items:center;justify-content:center;z-index:10;cursor:zoom-out}
#lb img{max-width:96vw;max-height:94vh;background:#fff}
.empty{color:var(--mute);padding:24px;text-align:center}
.pl{width:100%;height:auto;display:block;user-select:none}.pl text{fill:var(--mute);font-size:11px}.pl .gr{stroke:var(--line)}.pl .ax{fill:none;stroke:var(--line)}.pl .zr{fill:var(--acc);opacity:.15;pointer-events:none}
.bar button.on{border-color:var(--acc);color:var(--acc);font-weight:600}
.bar input[type=number]{width:80px;padding:6px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin:0 0 10px}
.chip{display:inline-flex;align-items:center;gap:5px;padding:2px 8px;font:inherit;font-size:12px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:12px;cursor:pointer}
.chip.off{opacity:.35}.chip i{width:10px;height:10px;border-radius:50%}.chip i.sq{border-radius:1px}
.panels{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}.panels h3{font-size:12px;font-weight:500;margin:0}
.ramp{display:inline-block;width:120px;height:8px;border-radius:4px;vertical-align:middle}
#tip{position:fixed;display:none;z-index:9;pointer-events:none;white-space:pre;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.25)}
</style></head><body>
<header><h1 id="t"></h1><div class="sub" id="s"></div><nav id="nav"></nav></header>
<main id="m"></main><div id="lb"><img alt=""></div><div id="tip"></div>
<script>
const D=__DATA__;
const TABS=[['overview','개요'],...(D.lcdata?[['lc','광도곡선'],['sed','SED']]:[]),...D.stages.map(s=>[s,D.labels[s]])];
let tab='overview',idx=0;
const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function fig(f){return `<figure${f.wide?' class="wide"':''}><img loading="lazy" src="${f.thumb}" data-full="${f.src}" alt="">${f.caption?`<p class="cap">${esc(f.caption)}</p>`:''}</figure>`}
function card(t,figs){return `<div class="card"><h2>${esc(t)}</h2><div class="grid">${figs.map(fig).join('')}</div></div>`}
function overview(){
  const head=D.cols.map(c=>`<th>${esc(c[1])}</th>`).join('');
  const rows=D.images.map((im,i)=>`<tr data-i="${i}"><td>${esc(im.short)}</td>${D.cols.map(c=>`<td>${esc(im.row[c[0]])}</td>`).join('')}<td class="${im.ok?'':'bad'}" style="text-align:left">${esc(im.status)}</td></tr>`).join('');
  let h=`<div class="card"><h2>이미지 ${D.images.length}장</h2><div class="wrap"><table><thead><tr><th>프레임</th>${head}<th style="text-align:left">상태</th></tr></thead><tbody>${rows}</tbody></table></div><p class="cap">행을 누르면 그 이미지의 단계별 그림으로 이동합니다.</p></div>`;
  if(D.run.length)h+=card('실행 추이',D.run);
  if(D.lc.length)h+=card('광도곡선',D.lc);
  return h}
function stage(){
  const im=D.images[idx],figs=(im.stages[tab]||[]);
  const opts=D.images.map((x,i)=>`<option value="${i}"${i===idx?' selected':''}>${i+1}. ${esc(x.short)} · ${esc(x.row.filter)} · ${esc(x.row.obsdate)}</option>`).join('');
  return `<div class="bar"><button id="pv">←</button><select id="sel">${opts}</select><button id="nx">→</button><span class="meta">${idx+1} / ${D.images.length}</span></div>
  <div class="meta" style="margin-bottom:10px">${esc(im.name)} — seeing ${esc(im.row.SEEING)}″, ZP ${esc(im.row[D.cols[5][0]])} ± ${esc(im.row[D.cols[6][0]])}, UL5 ${esc(im.row[D.cols[7][0]])} <span class="${im.ok?'':'bad'}">· ${esc(im.status)}</span></div>
  ${figs.length?`<div class="grid">${figs.map(fig).join('')}</div>`:'<div class="empty">이 이미지에는 이 단계의 그림이 없습니다.</div>'}`}
function render(){
  $('#nav').innerHTML=TABS.map(([k,l])=>`<button data-k="${k}" class="${k===tab?'on':''}">${esc(l)}</button>`).join('');
  const lcv=tab==='lc'||tab==='sed';
  $('#m').innerHTML=tab==='overview'?overview():lcv?lcView():stage();
  if(lcv)draw();
  history.replaceState(null,'',tab==='overview'?'#':lcv?`#${tab}`:`#${tab}/${idx+1}`)}
function go(i){idx=(i+D.images.length)%D.images.length;render()}
document.addEventListener('click',e=>{
  const b=e.target.closest('nav button');if(b){tab=b.dataset.k;render();return}
  const r=e.target.closest('tbody tr');if(r){idx=+r.dataset.i;tab=D.stages.includes('photometry')?'photometry':D.stages[0];render();scrollTo(0,0);return}
  if(e.target.id==='pv')return go(idx-1);if(e.target.id==='nx')return go(idx+1);
  const img=e.target.closest('figure img');if(img){$('#lb img').src=img.dataset.full;$('#lb').style.display='flex';return}
  if(e.target.closest('#lb'))$('#lb').style.display='none'});
document.addEventListener('change',e=>{if(e.target.id==='sel')go(+e.target.value)});
document.addEventListener('keydown',e=>{if(e.key==='Escape')$('#lb').style.display='none';
  if(tab==='overview'||tab==='lc'||tab==='sed'||e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft')go(idx-1);if(e.key==='ArrowRight')go(idx+1)});
// interactive light curve / SED (D.lcdata, only with --lightcurve); time binning and combining happen here
const L=D.lcdata||{},FL=L.filters||[],PT=(L.pts||[]).map(p=>({t:p[0],f:p[1],m:p[2],e:p[3],fr:p[4],tel:p[5],run:p[6]}));
const S={bin:L.bin||1,comb:true,ind:false,mode:'abs',off:new Set(),zoom:null,sed:-1},DARK=matchMedia('(prefers-color-scheme:dark)').matches;
const WLS=FL.map(f=>f.wl).filter(w=>w!=null),WL0=Math.min(...WLS),WL1=Math.max(...WLS);
let TIPS=[],ZD=null;
const utc=t=>new Date((t-40587)*864e5).toISOString().slice(0,19).replace('T',' ');
const fwl=f=>f.wl==null?'파장 없음':f.src==='name'?`≈${f.wl} nm (이름 추정)`:`${f.wl} nm`;
const wt=p=>1/Math.max(p.e,1e-4)**2;                 // inverse variance; floor keeps MAGERR=0 finite
function fcol(i){const w=FL[i].wl;if(w==null)return 'var(--mute)';const k=WL1>WL0?(w-WL0)/(WL1-WL0):.5;
  return `hsl(${Math.round(270*(1-k))},${DARK?80:70}%,${DARK?62:42}%)`}   // violet (blue end) -> red
const FM=FL.map((_,i)=>{let sw=0,sm=0;for(const p of PT)if(p.f===i){sw+=wt(p);sm+=wt(p)*p.m}return sm/sw});  // per-filter mean for dmag
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
function ptip(p,d){const f=FL[p.f];
  return `${f.n} · ${fwl(f)}\n${p.m.toFixed(3)} ± ${p.e.toFixed(3)}${dtxt(d)}\n${p.fr}\n${p.tel} · ${utc(p.t)} UTC${p.run?'':'\n(다른 카탈로그)'}`}
function ctip(c,d){const f=FL[c.f],w=S.bin;
  return `${f.n} · ${fwl(f)}\n${c.m.toFixed(3)} ± ${c.e.toFixed(3)}${dtxt(d)} · n=${c.n}\n`+
    (c.n>1?`√(1/Σw) ${c.ew.toFixed(3)} · std/√n ${c.es.toFixed(3)}`:`${c.a[0].fr} (프레임 오차)`)+
    `\n${[...new Set(c.a.map(p=>p.tel))].join(', ')}\nbin ${c.b}: ${utc(L.t0+c.b*w)} – ${utc(L.t0+(c.b+1)*w)} UTC`}
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
  if(est.length)h+=`<p class="cap">투과곡선이 없어 이름에서 파장을 추정: ${esc(est.join(', '))}</p>`;
  if(no.length)h+=`<p class="cap bad">파장 없음${sed?' (SED에서 제외)':''}: ${esc(no.join(', '))}</p>`;
  if(bad.length)h+=`<p class="cap bad">투과곡선 읽기 실패: ${esc(bad.join('; '))}</p>`;
  return h}
function lcView(){return L.error?`<div class="card"><div class="empty bad">광도곡선·SED 데이터를 만들지 못했습니다: ${esc(L.error)}</div></div>`:'<div class="card" id="lcc"></div>'}
function draw(){
  const c=$('#lcc');if(!c)return;TIPS=[];const sed=tab==='sed';
  c.innerHTML=controls(sed)+chips()+(PT.length?(sed?sedPlot():lcPlot()):'<div class="empty">표시할 점이 없습니다.</div>')+notes(sed)}
document.addEventListener('click',e=>{const a=e.target.closest('[data-act]');if(!a)return;const k=a.dataset.act;
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
$('#t').textContent=D.title;$('#s').textContent=D.sub;
const m=location.hash.match(/^#(\w+)\/(\d+)/);if(m&&D.stages.includes(m[1])){tab=m[1];idx=Math.min(+m[2]-1,D.images.length-1)}
if(D.lcdata&&/^#(lc|sed)$/.test(location.hash))tab=location.hash.slice(1);
render();
</script></body></html>'''


def _objname(path: Path) -> str:
    """OBJNAME from .info, else the scidata layout .../<objname>/<telname>/<filter>/<file>."""
    return _info(path).get('OBJNAME') or path.parents[2].name


def _gallery(rows: pd.DataFrame, lc_df: Optional[pd.DataFrame], out: Path, title: str, source: str,
             ra=None, dec=None, stages=None, verbose=True) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    images, manifest, seen = [], {}, []
    for _, row in rows.iterrows():
        path = Path(row['path'])
        status = 'ok' if bool(row['success']) else f"FAILED at {row['step']}: {row['error']}"
        short = path.name.split('_')[2] + '_' + path.stem.split('_')[-1] if path.name.count('_') > 3 else path.stem
        entry = dict(id=path.stem, name=path.name, short=short, ok=bool(row['success']), status=status, stages={},
                     row={c: _fmt(row.get(c), c) for c, _ in SUMMARY_COLS})
        images.append(entry)
        if not path.exists():
            entry['status'] += ' (file missing)'
            entry['ok'] = False
            continue
        imgdir = out / path.stem
        imgdir.mkdir(exist_ok=True)
        data, header = fits.getdata(path, header=True)
        data = data.astype(np.float32)
        info = _info(path)
        for stage in (stages or _auto_stages(path, header, info)):
            try:
                if stage == 'photometry':
                    names = fig_photometry(path, data, header, info, imgdir, ra=ra, dec=dec)
                else:
                    names = globals()[f'fig_{stage}'](path, data, header, info, imgdir)
            except Exception as e:
                names = []
                entry['status'] += f' | {stage} figure failed: {type(e).__name__}: {e}'
                entry['ok'] = False
            entry['stages'][stage] = [_figure(imgdir, n, f'{path.stem}/') for n in names]
            if stage not in seen:
                seen.append(stage)
        if verbose:
            print(f"[visualize] {path.name}: " + ', '.join(f'{k}({len(v)})' for k, v in entry['stages'].items()))
        manifest[str(path)] = {k: [f['src'] for f in v] for k, v in entry['stages'].items()}

    run = [_figure(out, n) for n in fig_run(rows, out)]
    lc = [_figure(out, n) for n in fig_lightcurve(lc_df, out, rows['path'])] if lc_df is not None else []
    lcdata = None
    if lc_df is not None:
        try:
            lcdata = _lc_payload(lc_df, rows['path'], verbose)
        except Exception as e:                     # shown in the light-curve / SED tabs
            lcdata = dict(error=re.sub(r'(?<![\w.~])/[^\s\'"]+', '<path>', f'{type(e).__name__}: {e}'))   # no paths in the page
            print(f"[visualize] interactive light curve failed: {lcdata['error']}")
    filters = ', '.join(sorted(rows['filter'].astype(str).unique()))
    tels = ', '.join(sorted(rows['telname'].astype(str).unique()))
    n_ok = int(rows['success'].astype(bool).sum())
    data = dict(title=title, sub=f'{source} · {n_ok}/{len(rows)} 성공 · 필터 {filters} · {tels}',
                stages=[s for s in STAGES if s in seen], labels=STAGE_LABELS, cols=SUMMARY_COLS,
                images=images, run=run, lc=lc)
    if lcdata is not None:
        data['lcdata'] = lcdata
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
              verbose: bool = True) -> List[Path]:
    """Write per-stage figures and index.html for a process_images summary, one gallery per objname
    at <root>/ezphot/<objname>/viz/ (a later run of the same objname replaces the page); return the index paths."""
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

    objnames = rows['path'].map(lambda p: _objname(Path(p)))
    indices = []
    for obj, g in rows.groupby(objnames):
        lc_df = lc_all
        if lc_all is not None and 'objname' in lc_all:
            lc_df = lc_all[lc_all['objname'] == obj]
        indices.append(_gallery(g, lc_df, root / 'ezphot' / obj / 'viz', obj, source,
                                ra=ra, dec=dec, stages=stages, verbose=verbose))
    return indices


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('summary')
    ap.add_argument('--ra', type=float); ap.add_argument('--dec', type=float)
    ap.add_argument('--lightcurve', help='target_photometry table (CSV)')
    ap.add_argument('--stages', nargs='+', choices=STAGES, help='override the stages found on disk')
    ap.add_argument('--max-images', type=int)
    ap.add_argument('--root', help=f'gallery root served over http (default {VIZ_ROOT})')
    a = ap.parse_args(argv)
    visualize(a.summary, ra=a.ra, dec=a.dec, lightcurve=a.lightcurve, stages=a.stages,
              max_images=a.max_images, root=a.root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
