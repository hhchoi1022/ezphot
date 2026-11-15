#!/usr/bin/env python
"""
forced_photometry: forced circular-aperture photometry at given sky positions on calibrated images
(single frames, stacks or difference images), all targets of an image in one call.

    calibrate='header' (default): photometry_forced_circular at (ra, dec) -> apply_zp with the image header
        ZP_APER_k / ZPERR_APER_k (UL3SKY/UL5SKY = the catalog's UL3/UL5_APER_k + ZP), as dia does on difference images
    calibrate='reference_stars': photometry_forced_circular at (ra, dec) and at the reference stars of the image
        (X_WORLD/Y_WORLD of <img>.fits.refcat, written by process_images/stack) -> photometric_calibration on that
        catalog (process_images photcal settings; reference stars picked on MAG_APER_3, no region cut: the refcat
        stars already passed the region cut of process_images). The forced catalog has no FWHM/CLASS_STAR/
        ELONGATION/FLAGS columns, so those star cuts and SATDIST are skipped, and the isolation cut only sees the
        measured positions.
        The recipe fixes update_header=False, save_image=False, save=True, mag_lower/upper (defaults.MAG_RANGE) and
        the region (ra_deg=dec_deg=None, inner_fraction=1); config cannot change these.

Background: difference images (sub_*) none, as dia (circular_photometry computes the bkgrms from the image). Other
images use their .bkgmap/.bkgrms sidecars, i.e. the background of the process_images/stack photometry that set the
header ZP (stacks: the propagated bkgrms). A missing sidecar is computed from the image in memory, without a source
mask (sources only sigma-clipped, so it may differ from process_images); bkg_source / bkgrms_source say which.

Products next to each image (its own folder, not the config archive layout): <img>.fits.circ.cat (+ .info), an ASCII
table with X_WORLD/Y_WORLD and the per-aperture MAGSKY/MAGERR/ZP/ZPERR/UL5SKY_APER[_k] columns; with
'reference_stars' also <img>.fits.circ.refcat. A second run on the same image overwrites them (a 'header' run leaves
an older .circ.refcat as it is). No figures. The input FITS is not rewritten.
One summary row per image x target (target = index in ra/dec): success/step/error (the same for all targets of an
image), n_ref (reference stars of 'reference_stars'), bkg_source, bkgrms_source, catalog and the target's
MAGSKY/MAGERR/ZP/ZPERR/UL5SKY in the 10" aperture (*_APER_3). An image without a finite ZP_APER_3 fails (header: step
apply_zp, no ZP_APER_3 in the header; reference_stars: step photometric_calibration, also when n_ref is 0); an image
that cannot be opened fails at step load; an empty UL5SKY_APER_3 means no depth was available.
While it runs, <out>.progress.json next to the summary counts the finished images (read by recipes.status).

Python:
    from ezphot.recipes import forced_photometry
    summary = forced_photometry(['coadd_....fits', 'stack_....csv'], ra=[76.3, 76.4], dec=[-68.0, -68.1])

CLI (images: FITS paths and/or process_images/stack summary CSVs, whose successful rows are used):
    python -m ezphot.recipes.forced_photometry <image.fits | summary.csv> [...] --ra 76.3 [76.4 ...] \
        --dec -68.0 [-68.1 ...] [--calibrate header|reference_stars] [--n-proc 4] [--out forced.csv]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from ezphot.dataobjects import Catalog
from ezphot.imageobjects import ScienceImage
from ezphot.recipes import defaults as D
from ezphot.recipes._progress import run_tasks
from ezphot.recipes.stack import load_summary

FORCED_CONFIG_KEYS = ('forced_photometry', 'photcal')
CALIBRATE = ('header', 'reference_stars')
FORCED_KEYS = [f'{p}_{D.APER10_KEY}' for p in ('MAGSKY', 'MAGERR', 'ZP', 'ZPERR', 'UL5SKY')]
ROW_KEYS = ('path', 'target', 'ra', 'dec', 'telname', 'filter', 'obsdate', 'exptime', 'calibrate', 'success', 'step',
            'error', 'n_ref', 'bkg_source', 'bkgrms_source', 'catalog')   # summary columns before FORCED_KEYS
# Processing recipes make no figures. Forced catalogs have no MAG_AUTO (the photcal section's key): reference
# stars are picked on the 10" aperture; photcal still fits the ZP of every aperture.
RECIPE_DEFAULTS = {'forced_photometry': dict(visualize=False, save_fig=False),
                   'photcal': dict(visualize=False, save_fig=False, magnitude_key=f'MAG_{D.APER10_KEY}',
                                   magnitudeerr_key=f'MAGERR_{D.APER10_KEY}')}


def _config(overrides: dict = None) -> dict:
    cfg = {k: dict(getattr(D, k.upper()), **RECIPE_DEFAULTS[k]) for k in FORCED_CONFIG_KEYS}
    for k, v in (overrides or {}).items():
        if k not in cfg:
            raise KeyError(f"unknown config section {k!r}; expected one of {FORCED_CONFIG_KEYS}")
        cfg[k].update(v)
    return cfg


def _targets(ra, dec):
    ra, dec = np.atleast_1d(np.asarray(ra, dtype=float)), np.atleast_1d(np.asarray(dec, dtype=float))
    if ra.ndim != 1 or ra.shape != dec.shape or len(ra) == 0:
        raise ValueError(f"ra and dec need the same number (>= 1) of values, got shapes {ra.shape} and {dec.shape}")
    return ra, dec


def _background(target_img: ScienceImage):
    """(bkg, bkgrms, bkg_source, bkgrms_source) for the forced photometry; target_img.savedir is its own folder."""
    if target_img.path.name.startswith('sub_'):   # as dia
        return None, None, 'none', 'image'
    bkg, bkgrms = target_img.bkgmap, target_img.bkgrms
    sources = ('sidecar' if bkg is not None else 'image', 'sidecar' if bkgrms is not None else 'image')
    if bkg is None or bkgrms is None:
        # In memory and without a source mask: mask_sources/mask_circle delete an existing <img>.srcmask here.
        quiet = dict(save=False, verbose=False, visualize=False, save_fig=False)
        ivpmask = target_img.calculate_invalidmask(**dict(D.INVALIDMASK, **quiet))
        if bkg is None:
            bkg = target_img.calculate_bkg(target_ivpmask=ivpmask, **dict(D.BKGMAP, **quiet))
        if bkgrms is None:
            bkgrms = target_img.calculate_bkgrms(target_ivpmask=ivpmask, **dict(D.BKGRMS, **quiet))
    return (bkg, bkgrms) + sources


def forced_image(target_img: ScienceImage,
                 ra,
                 dec,
                 calibrate: str = 'header',
                 config: dict = None,
                 clear: bool = True) -> List[dict]:
    """Forced photometry at every (ra, dec) on one image, in one call; one summary row per target.
    Processing errors do not raise (rows with success=False); bad arguments (config section, calibrate, ra/dec) do.
    Sets target_img.savedir to the image's folder, where the .circ.cat is written."""
    cfg = _config(config)
    if calibrate not in CALIBRATE:
        raise ValueError(f"calibrate must be one of {CALIBRATE}, got {calibrate!r}")
    ra, dec = _targets(ra, dec)
    row = dict(dict.fromkeys(ROW_KEYS), path=str(target_img.path), calibrate=calibrate, success=False)
    values = [dict.fromkeys(FORCED_KEYS) for _ in ra]
    bkg = bkgrms = refcat = catalog = used = None
    try:
        row['step'] = 'load'   # header values that do not parse (DATE-OBS, EXPTIME) fail this image only
        row.update(telname=target_img.telname, filter=target_img.filter, obsdate=target_img.obsdate,
                   exptime=target_img.exptime)
        target_img.savedir = target_img.path.parent   # products next to the image, not the config archive layout
        row['step'] = 'background'
        bkg, bkgrms, row['bkg_source'], row['bkgrms_source'] = _background(target_img)
        x, y = ra, dec
        if calibrate == 'reference_stars':
            row['step'] = 'refcat'
            refcat = Catalog(target_img.savepath.refcatalogpath, catalog_type='reference', load=False, target_img=target_img)
            if not refcat.is_exists:
                raise FileNotFoundError(f"no reference-star catalog {refcat.path}")
            x = np.concatenate([ra, np.asarray(refcat.data['X_WORLD'], dtype=float)])
            y = np.concatenate([dec, np.asarray(refcat.data['Y_WORLD'], dtype=float)])
        row['step'] = 'forced_photometry'
        catalog = target_img.photometry_forced_circular(
            x_arr=x, y_arr=y, unit='coord', target_bkg=bkg, target_bkgrms=bkgrms,
            **dict(cfg['forced_photometry'], save=True))   # save=False would remove the .circ.cat
        row['catalog'] = str(catalog.path)
        if calibrate == 'header':
            row['step'] = 'apply_zp'
            catalog = target_img.apply_zp(catalog, verbose=False, save=True)
        else:
            row['step'] = 'photometric_calibration'
            # The input FITS stays as it is; no region cut, the refcat stars already passed the process_images one.
            photcal = dict(cfg['photcal'], update_header=False, save_image=False, save=True,
                           ra_deg=None, dec_deg=None, inner_fraction=1.0)
            photcal['mag_lower'], photcal['mag_upper'] = D.MAG_RANGE.get(target_img.filter, D.MAG_RANGE_DEFAULT)
            _, catalog, used, _ = target_img.photometric_calibration(target_catalog=catalog, **photcal)
            row['n_ref'] = used.nsources
        data = catalog.data   # the targets are its first len(ra) rows
        zp_key = FORCED_KEYS[2]   # photcal writes a rough MAGSKY before selecting stars: judge on the ZP column
        if zp_key not in data.colnames or not np.isfinite(float(data[zp_key][0])) or row['n_ref'] == 0:
            why = ('missing or not finite in the image header' if calibrate == 'header'
                   else f"{row['n_ref']} reference stars")
            raise RuntimeError(f"no finite {zp_key}: {why}")
        for i, v in enumerate(values):
            for k in FORCED_KEYS:
                if k in data.colnames:
                    v[k] = float(data[k][i])
        row['success'] = True
        row['step'] = 'done'
    except Exception as e:
        row['error'] = f"{type(e).__name__}: {e}"
    finally:
        if clear:
            for obj in (target_img, bkg, bkgrms, refcat, catalog, used):
                if obj is not None:
                    try:
                        obj.clear(verbose=False)
                    except Exception:
                        pass
    return _target_rows(row, ra, dec, values)


def _target_rows(row: dict, ra, dec, values: List[dict] = None) -> List[dict]:
    values = values or [dict.fromkeys(FORCED_KEYS) for _ in ra]
    return [dict(row, target=i, ra=float(r), dec=float(d), **v) for i, (r, d, v) in enumerate(zip(ra, dec, values))]


def _image_paths(images: Union[str, Path, List]) -> List[str]:
    """Resolved image paths, duplicates dropped: FITS paths, and for *.csv (process_images/stack summaries) the paths
    of their successful rows."""
    paths = []
    for f in ([images] if isinstance(images, (str, Path)) else images):
        paths += [str(Path(p).expanduser().resolve()) for p in (load_summary(f) if str(f).endswith('.csv') else [f])]
    return list(dict.fromkeys(paths))


def _worker(args):
    path, ra, dec, calibrate, config = args
    try:
        target_img = ScienceImage(path=path, load=True)
    except Exception as e:   # missing file, unreadable header, unknown telescope: failed rows, the run goes on
        row = dict(dict.fromkeys(ROW_KEYS), path=path, calibrate=calibrate, success=False, step='load',
                   error=f"{type(e).__name__}: {e}")
        rows = _target_rows(row, ra, dec)
    else:
        rows = forced_image(target_img, ra, dec, calibrate, config, True)
    return dict(success=rows[0]['success'], rows=rows)   # one progress step per image


def forced_photometry(images: Union[str, Path, List],
                      ra,
                      dec,
                      calibrate: str = 'header',
                      config: dict = None,
                      n_proc: int = 4,
                      out: Union[str, Path] = None,
                      verbose: bool = True) -> pd.DataFrame:
    """Run forced_image over FITS paths and/or the successful images of process_images/stack summary CSVs;
    return (and save) one row per image x target."""
    paths = _image_paths(images)
    if len(paths) == 0:
        raise ValueError("no images given")
    ra, dec = _targets(ra, dec)
    _config(config)   # unknown sections fail here, not once per image
    if calibrate not in CALIBRATE:
        raise ValueError(f"calibrate must be one of {CALIBRATE}, got {calibrate!r}")
    if out is None:   # decided at the start: <out>.progress.json is written next to it while running
        objname = ScienceImage(path=paths[0], load=True).objname or 'unknown'
        out = Path.home() / 'ezphot' / 'log' / 'recipes' / objname / f"forced_photometry_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tasks = [(p, ra, dec, calibrate, config) for p in paths]
    rows = [row for result in run_tasks(_worker, tasks, n_proc, out) for row in result['rows']]
    summary = pd.DataFrame(rows)
    summary['n_ref'] = summary['n_ref'].astype('Int64')   # integer also when failed rows (empty) are mixed in
    summary.to_csv(out, index=False)
    if verbose:
        n_ok = int(summary['success'].sum())
        print(f"[forced_photometry] {n_ok}/{len(summary)} rows ({len(paths)} images x {len(ra)} targets) "
              f"succeeded in {time.time() - t0:.0f}s")
        if n_ok < len(summary):
            failed = summary.loc[~summary['success']].drop_duplicates('path')
            print(failed[['path', 'step', 'error']].to_string(index=False))
        print(f"[forced_photometry] summary -> {out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('images', nargs='+', help='FITS image(s) and/or process_images or stack summary CSV(s)')
    ap.add_argument('--ra', type=float, nargs='+', required=True)
    ap.add_argument('--dec', type=float, nargs='+', required=True)
    ap.add_argument('--calibrate', choices=CALIBRATE, default='header')
    ap.add_argument('--n-proc', type=int, default=4)
    ap.add_argument('--out', help='summary CSV path')
    a = ap.parse_args(argv)
    summary = forced_photometry(a.images, a.ra, a.dec, calibrate=a.calibrate, n_proc=a.n_proc, out=a.out)
    return 0 if summary['success'].all() else 1


if __name__ == '__main__':
    raise SystemExit(main())
