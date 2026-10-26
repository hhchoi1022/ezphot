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
    lightcurve  only with --lightcurve (target_photometry table) and >1 epoch

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
</style></head><body>
<header><h1 id="t"></h1><div class="sub" id="s"></div><nav id="nav"></nav></header>
<main id="m"></main><div id="lb"><img alt=""></div>
<script>
const D=__DATA__;
const TABS=[['overview','개요'],...D.stages.map(s=>[s,D.labels[s]])];
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
  $('#m').innerHTML=tab==='overview'?overview():stage();
  history.replaceState(null,'',tab==='overview'?'#':`#${tab}/${idx+1}`)}
function go(i){idx=(i+D.images.length)%D.images.length;render()}
document.addEventListener('click',e=>{
  const b=e.target.closest('nav button');if(b){tab=b.dataset.k;render();return}
  const r=e.target.closest('tbody tr');if(r){idx=+r.dataset.i;tab=D.stages.includes('photometry')?'photometry':D.stages[0];render();scrollTo(0,0);return}
  if(e.target.id==='pv')return go(idx-1);if(e.target.id==='nx')return go(idx+1);
  const img=e.target.closest('figure img');if(img){$('#lb img').src=img.dataset.full;$('#lb').style.display='flex';return}
  if(e.target.closest('#lb'))$('#lb').style.display='none'});
document.addEventListener('change',e=>{if(e.target.id==='sel')go(+e.target.value)});
document.addEventListener('keydown',e=>{if(e.key==='Escape')$('#lb').style.display='none';
  if(tab==='overview'||e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft')go(idx-1);if(e.key==='ArrowRight')go(idx+1)});
$('#t').textContent=D.title;$('#s').textContent=D.sub;
const m=location.hash.match(/^#(\w+)\/(\d+)/);if(m&&D.stages.includes(m[1])){tab=m[1];idx=Math.min(+m[2]-1,D.images.length-1)}
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
    filters = ', '.join(sorted(rows['filter'].astype(str).unique()))
    tels = ', '.join(sorted(rows['telname'].astype(str).unique()))
    n_ok = int(rows['success'].astype(bool).sum())
    data = dict(title=title, sub=f'{source} · {n_ok}/{len(rows)} 성공 · 필터 {filters} · {tels}',
                stages=[s for s in STAGES if s in seen], labels=STAGE_LABELS, cols=SUMMARY_COLS,
                images=images, run=run, lc=lc)
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
