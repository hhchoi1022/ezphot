#!/usr/bin/env python
"""
dia: difference imaging of calibrated images (single frames or stacks) against a reference frame.

    reference (get_referenceframe, or --reference) -> DIA (hotpants, bridge DIA settings) -> normalisation QA
    -> apply_zp on the difference catalog -> forced photometry at (ra, dec) on the difference image
       (recipes.forced_photometry.forced_image, calibrate='header')

Hotpants picks its stamps itself, away from (ra, dec) and config['dia']['exclude_coords'] (radius
exclude_radius_arcsec, default 10"); each difference image gets sub_<name>.fits.hotpants.json (hotpants command, used
stamps, noise options) next to it.

Input is one or more summary CSVs of process_images or stack (success rows). One row per image:
reference path, difference image, number of candidates/transients and, if ra/dec are given, the
target's forced MAGSKY/MAGERR/ZPERR/UL5SKY in the 10" aperture (*_APER_3) on the difference image (local annulus
background, empirical noise ratio and host Poisson term as forced_photometry does on sub_ images; a difference image
without a finite ZP_APER_3 fails at step apply_zp), then ra, dec, noise_ratio, host_source and the normalisation QA
columns (_norm_qa; they change no value and no success). An image that cannot be opened fails at step load.
While it runs, <out>.progress.json next to the summary counts the finished images (read by recipes.status).

Python:
    from ezphot.recipes import dia
    summary = dia('~/ezphot/log/recipes/EG21/stack_....csv', ra=..., dec=..., n_proc=4)

CLI:
    python -m ezphot.recipes.dia <summary.csv> [more.csv ...] [--ra 76.3 --dec -68.0] \
        [--reference ref.fits] [--n-proc 4] [--out dia.csv]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from photutils.aperture import CircularAperture, aperture_photometry

from ezphot.imageobjects import ReferenceImage, ScienceImage
from ezphot.recipes import defaults as D
from ezphot.recipes._common import NothingToDo
from ezphot.recipes._progress import run_tasks
from ezphot.recipes.forced_photometry import NOISE_KEYS, forced_image
from ezphot.recipes.stack import load_summary

DIA_CONFIG_KEYS = ('dia', 'forced_photometry', 'forced_photometry_sub')
FORCED_KEYS = [f'{p}_{D.APER10_KEY}' for p in ('MAGSKY', 'MAGERR', 'ZPERR', 'UL5SKY')]
NORM_KEYS = ('norm_resid', 'norm_resid_n', 'norm_resid_mad', 'ksum', 'ksum_exp', 'ksum_ratio', 'norm_flag')
# Normalisation QA (_norm_qa) as the 2026-11 T08635 DIA QA that set defaults.DIA_NORM_RESID_MAX: 10" aperture, local
# background = 3-sigma-clipped mean in 15-25" (radius); norm_resid needs at least NORM_MIN_STARS stars.
NORM_APER_ARCSEC, NORM_ANNULUS_ARCSEC, NORM_MIN_STARS = 10.0, (15.0, 25.0), 10


def _config(overrides: dict = None) -> dict:
    cfg = {k: dict(getattr(D, k.upper())) for k in DIA_CONFIG_KEYS}
    for k, v in (overrides or {}).items():
        if k not in cfg:
            raise KeyError(f"unknown config section {k!r}; expected one of {DIA_CONFIG_KEYS}")
        cfg[k].update(v)
    return cfg


def _row(path, ra=None, dec=None) -> dict:
    """Summary row of one image before processing: every column, in summary order."""
    row = dict(path=str(path), telname=None, filter=None, obsdate=None, exptime=None,
               success=False, step=None, error=None, reference=None, subtracted=None,
               n_candidates=None, n_transients=None)
    row.update(dict.fromkeys(FORCED_KEYS), ra=ra, dec=dec)
    row.update(dict.fromkeys(NOISE_KEYS), **dict.fromkeys(NORM_KEYS))
    return row


def _one_target(ra, dec):
    """(ra, dec) of the one dia target as floats, each None when not given (forced photometry needs both): a number or
    a one-element list/tuple/array; anything else is a ValueError, raised by dia() before any processing."""
    out = []
    for name, v in (('ra', ra), ('dec', dec)):
        if v is not None:
            try:
                a = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                a = None
            if a is None or a.size != 1:
                raise ValueError(f"dia measures one target: {name} must be a number or a one-element sequence, "
                                 f"got {v!r}")
            v = float(a.reshape(-1)[0])
        out.append(v)
    return tuple(out)


def _star_ratios(target_img: ScienceImage, subtracted_img: ScienceImage) -> np.ndarray:
    """10" flux ratio difference / science of the field stars: the science image's ZP stars (X_WORLD/Y_WORLD of
    <img>.fits.refcat next to it, written by process_images/stack; none without it) on sub_<name> and sci_<name> (the
    reprojected science frame of the DIA, same grid), each with its own 3-sigma-clipped mean in 15-25" subtracted; only
    stars whose box (half-width 25" + 2 px) lies inside both images with finite, valid pixels (|value| >= 1e-20, ADU
    scale assumed: the empty areas of sci_ (SWarp weight 0 of the reprojection) are 0 before 0.9.0 and NaN from
    0.9.0, sub_ has the hotpants fill 1e-30 where invalid, e.g. above the DIA iu/tu) and with a positive science
    flux.
    Reads the files (read-only memory maps), writes nothing."""
    refcat = Path(f'{target_img.path}.refcat')
    sub = Path(subtracted_img.path)
    sci = sub.with_name('sci_' + sub.name[len('sub_'):])
    if not refcat.exists() or not sci.exists():
        return np.array([])
    stars = Table.read(str(refcat), format='ascii')
    x, y = subtracted_img.wcs.all_world2pix(np.asarray(stars['X_WORLD'], float), np.asarray(stars['Y_WORLD'], float), 0)
    scale = float(np.mean(subtracted_img.pixelscale))   # arcsec / px
    r_ap, r_in, r_out = NORM_APER_ARCSEC / 2 / scale, NORM_ANNULUS_ARCSEC[0] / scale, NORM_ANNULUS_ARCSEC[1] / scale
    b = int(np.ceil(r_out)) + 2
    yy, xx = np.mgrid[-b:b + 1, -b:b + 1]
    ratios = []
    with fits.open(sub, memmap=True) as hdul_sub, fits.open(sci, memmap=True) as hdul_sci:
        ny, nx = hdul_sub[0].shape
        for xi, yi in zip(x, y):
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            xa, ya = int(round(xi)), int(round(yi))
            if not (b <= xa < nx - b and b <= ya < ny - b):
                continue
            box = (slice(ya - b, ya + b + 1), slice(xa - b, xa + b + 1))
            cuts = [np.array(h[0].data[box], dtype=float) for h in (hdul_sub, hdul_sci)]
            if not all(np.all(np.isfinite(c) & ~((c > -1e-20) & (c < 1e-20))) for c in cuts):   # NaN (0 before 0.9.0) sci_, 1e-30 sub_
                continue
            rr = np.hypot(xx - (xi - xa), yy - (yi - ya))
            annulus = (rr >= r_in) & (rr < r_out)
            aperture = CircularAperture((xi - xa + b, yi - ya + b), r=r_ap)
            f_sub, f_sci = (float(aperture_photometry(c - sigma_clipped_stats(c[annulus], sigma=3.0)[0],
                                                      aperture)['aperture_sum'][0]) for c in cuts)
            if f_sci > 0:
                ratios.append(f_sub / f_sci)
    return np.array(ratios)


def _norm_qa(target_img: ScienceImage, reference_img: ReferenceImage, subtracted_img: ScienceImage) -> dict:
    """Normalisation QA of one DIA (QA columns only: values and success do not depend on it; reads files, writes none).
    norm_resid = median of _star_ratios (0 when the hotpants flux scale is right), norm_resid_n = number of stars,
    norm_resid_mad = 1.4826 MAD; norm_resid, norm_resid_mad empty and norm_flag empty (not raised) with fewer than
    NORM_MIN_STARS stars (or no .refcat / sci_). ksum = KSUM00 of the difference header (hotpants kernel sum); only with
    CONVOL00 TEMPLATE (reference convolved to the science) ksum_exp = 10**(-0.4 (ZP_ref - ZP_sci)) of ZP_APER_3
    (reference header, difference = science header) and ksum_ratio = ksum / ksum_exp. norm_flag = |norm_resid| >
    defaults.DIA_NORM_RESID_MAX."""
    out = dict.fromkeys(NORM_KEYS)
    try:
        header, zp_key = subtracted_img.header, f'ZP_{D.APER10_KEY}'
        if header.get('KSUM00') is not None:
            out['ksum'] = float(header['KSUM00'])
        if (str(header.get('CONVOL00', '')).strip().upper() == 'TEMPLATE' and zp_key in header
                and zp_key in reference_img.header):
            out['ksum_exp'] = float(10 ** (-0.4 * (float(reference_img.header[zp_key]) - float(header[zp_key]))))
            if out['ksum'] is not None:
                out['ksum_ratio'] = out['ksum'] / out['ksum_exp']
        ratios = _star_ratios(target_img, subtracted_img)
        out['norm_resid_n'] = len(ratios)
        if len(ratios) >= NORM_MIN_STARS:
            med = float(np.median(ratios))
            out.update(norm_resid=med, norm_resid_mad=float(1.4826 * np.median(np.abs(ratios - med))),
                       norm_flag=bool(abs(med) > D.DIA_NORM_RESID_MAX))
    except Exception as e:   # QA columns only: report and leave them empty, the row goes on
        print(f"[dia] normalisation QA of {Path(str(subtracted_img.path)).name}: {type(e).__name__}: {e}", flush=True)
    return out


def dia_image(target_img: ScienceImage,
              ra: float = None,
              dec: float = None,
              reference: str = None,
              config: dict = None,
              clear: bool = True) -> dict:
    """Subtract the reference from one calibrated image. Never raises; returns a summary row."""
    cfg = _config(config)
    row = _row(target_img.path, ra, dec)
    reference_img = subtracted_img = subtracted_catalog = None
    candidate_catalog = transient_catalog = None
    try:
        row['step'] = 'load'   # header values that do not parse (DATE-OBS, EXPTIME) fail this image only
        row.update(telname=target_img.telname, filter=target_img.filter, obsdate=target_img.obsdate,
                   exptime=target_img.exptime)
        row['step'] = 'reference'
        if reference is not None:
            reference_img = ReferenceImage(path=reference, load=True)
        else:
            result = target_img.get_referenceframe(verbose=False)
            if result is None:
                raise RuntimeError("no reference frame found")
            reference_img = result[0]
        row['reference'] = str(reference_img.path)
        row['step'] = 'DIA'
        dia_kwargs = dict(cfg['dia'])   # the target (ra, dec) and config['dia'] exclude_coords stay out of the stamps
        if ra is not None and dec is not None:
            dia_kwargs['exclude_coords'] = [(ra, dec)] + list(dia_kwargs.get('exclude_coords') or [])
        all_catalogs, candidate_catalogs, transient_catalogs, _, _, subtract_imglist = target_img.DIA(
            reference_img=reference_img, target_bkg=None, **dia_kwargs)
        subtracted_img = subtract_imglist[0]
        candidate_catalog, transient_catalog = candidate_catalogs[0], transient_catalogs[0]
        row['subtracted'] = str(subtracted_img.path)
        row['n_candidates'] = candidate_catalog.nsources if candidate_catalog is not None else 0
        row['n_transients'] = transient_catalog.nsources if transient_catalog is not None else 0
        row.update(_norm_qa(target_img, reference_img, subtracted_img))
        if all_catalogs[0] is not None:
            row['step'] = 'apply_zp'
            subtracted_catalog = subtracted_img.apply_zp(all_catalogs[0], verbose=False, save=True)
        if ra is not None and dec is not None:
            # forced_image keeps the summary fields of the science image (the sub_ header MJD-OBS is not the science
            # one); it sets subtracted_img.savedir to its folder (where DIA wrote it) and clears its data when clear.
            row['step'] = 'forced_photometry'
            forced = forced_image(subtracted_img, ra, dec, 'header',
                                  {k: cfg[k] for k in ('forced_photometry', 'forced_photometry_sub')}, clear)[0]
            row.update({k: forced[k] for k in FORCED_KEYS + list(NOISE_KEYS)})
            if not forced['success']:   # its error; step apply_zp: no finite ZP_APER_3 in the difference header
                row['step'] = 'apply_zp' if forced['step'] == 'apply_zp' else 'forced_photometry'
                row['error'] = forced['error']
        if row['error'] is None:
            row['success'] = True
            row['step'] = 'done'
    except Exception as e:
        row['error'] = f"{type(e).__name__}: {e}"
    finally:
        if clear:
            for obj in (target_img, reference_img, subtracted_img, subtracted_catalog,
                        candidate_catalog, transient_catalog):
                if obj is not None:
                    try:
                        obj.clear(verbose=False)
                    except Exception:
                        pass
    return row


def _worker(args):
    path, ra, dec, reference, config = args
    try:
        target_img = ScienceImage(path=path, load=True)
    except Exception as e:   # missing file, unreadable header, unknown telescope: a failed row, the run goes on
        return dict(_row(path, ra, dec), step='load', error=f"{type(e).__name__}: {e}")
    return dia_image(target_img, ra, dec, reference, config, True)


def dia(summaries: Union[str, Path, List],
        ra: float = None,
        dec: float = None,
        reference: str = None,
        config: dict = None,
        n_proc: int = 4,
        out: Union[str, Path] = None,
        verbose: bool = True) -> pd.DataFrame:
    """Run dia_image over the successful images of process_images/stack summaries; return (and save) the table.
    ra, dec: one target (a number or a one-element sequence each); anything else is a ValueError before processing."""
    paths = load_summary(summaries)
    if len(paths) == 0:
        raise NothingToDo("no successful images in the summary")
    ra, dec = _one_target(ra, dec)
    if out is None:   # decided at the start: <out>.progress.json is written next to it while running
        objname = ScienceImage(path=paths[0], load=True).objname or 'unknown'
        out = Path.home() / 'ezphot' / 'log' / 'recipes' / objname / f"dia_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tasks = [(p, ra, dec, reference, config) for p in paths]
    rows = run_tasks(_worker, tasks, n_proc, out)
    summary = pd.DataFrame(rows)
    summary['norm_resid_n'] = summary['norm_resid_n'].astype('Int64')   # integer also with failed rows (empty)
    summary.to_csv(out, index=False)
    if verbose:
        n_ok = int(summary['success'].sum())
        print(f"[dia] {n_ok}/{len(summary)} images subtracted in {time.time() - t0:.0f}s")
        if n_ok < len(summary):
            print(summary.loc[~summary['success'], ['path', 'step', 'error']].to_string(index=False))
        print(f"[dia] summary -> {out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('summaries', nargs='+', help='process_images or stack summary CSV(s)')
    ap.add_argument('--ra', type=float); ap.add_argument('--dec', type=float)
    ap.add_argument('--reference', help='reference FITS for every image (default: get_referenceframe)')
    ap.add_argument('--n-proc', type=int, default=4)
    ap.add_argument('--out', help='summary CSV path')
    a = ap.parse_args(argv)
    summary = dia(a.summaries, ra=a.ra, dec=a.dec, reference=a.reference, n_proc=a.n_proc, out=a.out)
    return 0 if summary['success'].all() else 1


if __name__ == '__main__':
    raise SystemExit(main())
