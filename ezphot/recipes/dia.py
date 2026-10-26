#!/usr/bin/env python
"""
dia: difference imaging of calibrated images (single frames or stacks) against a reference frame.

    reference (get_referenceframe, or --reference) -> DIA (hotpants, bridge DIA settings)
    -> apply_zp on the difference catalog -> forced photometry at (ra, dec) on the difference image

Input is one or more summary CSVs of process_images or stack (success rows). One row per image:
reference path, difference image, number of candidates/transients and, if ra/dec are given, the
target's forced MAGSKY/MAGERR/ZPERR/UL5SKY in the 10" aperture (*_APER_3) on the difference image.

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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Union

import pandas as pd

from ezphot.imageobjects import ReferenceImage, ScienceImage
from ezphot.recipes import defaults as D
from ezphot.recipes.stack import load_summary

DIA_CONFIG_KEYS = ('dia', 'forced_photometry')
FORCED_KEYS = [f'{p}_{D.APER10_KEY}' for p in ('MAGSKY', 'MAGERR', 'ZPERR', 'UL5SKY')]


def _config(overrides: dict = None) -> dict:
    cfg = {k: dict(getattr(D, k.upper())) for k in DIA_CONFIG_KEYS}
    for k, v in (overrides or {}).items():
        if k not in cfg:
            raise KeyError(f"unknown config section {k!r}; expected one of {DIA_CONFIG_KEYS}")
        cfg[k].update(v)
    return cfg


def dia_image(target_img: ScienceImage,
              ra: float = None,
              dec: float = None,
              reference: str = None,
              config: dict = None,
              clear: bool = True) -> dict:
    """Subtract the reference from one calibrated image. Never raises; returns a summary row."""
    cfg = _config(config)
    row = dict(path=str(target_img.path), telname=target_img.telname, filter=target_img.filter,
               obsdate=target_img.obsdate, exptime=target_img.exptime,
               success=False, step=None, error=None, reference=None, subtracted=None,
               n_candidates=None, n_transients=None)
    for k in FORCED_KEYS:
        row[k] = None
    reference_img = subtracted_img = subtracted_catalog = forced_catalog = None
    candidate_catalog = transient_catalog = None
    try:
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
        all_catalogs, candidate_catalogs, transient_catalogs, _, _, subtract_imglist = target_img.DIA(
            reference_img=reference_img, target_bkg=None, **cfg['dia'])
        subtracted_img = subtract_imglist[0]
        candidate_catalog, transient_catalog = candidate_catalogs[0], transient_catalogs[0]
        row['subtracted'] = str(subtracted_img.path)
        row['n_candidates'] = candidate_catalog.nsources if candidate_catalog is not None else 0
        row['n_transients'] = transient_catalog.nsources if transient_catalog is not None else 0
        if all_catalogs[0] is not None:
            row['step'] = 'apply_zp'
            subtracted_catalog = subtracted_img.apply_zp(all_catalogs[0], verbose=False, save=True)
        if ra is not None and dec is not None:
            row['step'] = 'forced_photometry'
            forced_catalog = subtracted_img.photometry_forced_circular(
                x_arr=ra, y_arr=dec, unit='coord', target_bkg=None, target_bkgrms=None,
                **cfg['forced_photometry'])
            forced_catalog = subtracted_img.apply_zp(forced_catalog, verbose=False, save=True)
            data = forced_catalog.data
            for k in FORCED_KEYS:
                if k in data.colnames:
                    row[k] = float(data[k][0])
        row['success'] = True
        row['step'] = 'done'
    except Exception as e:
        row['error'] = f"{type(e).__name__}: {e}"
    finally:
        if clear:
            for obj in (target_img, reference_img, subtracted_img, subtracted_catalog, forced_catalog,
                        candidate_catalog, transient_catalog):
                if obj is not None:
                    try:
                        obj.clear(verbose=False)
                    except Exception:
                        pass
    return row


def _worker(args):
    path, ra, dec, reference, config = args
    return dia_image(ScienceImage(path=path, load=True), ra, dec, reference, config, True)


def dia(summaries: Union[str, Path, List],
        ra: float = None,
        dec: float = None,
        reference: str = None,
        config: dict = None,
        n_proc: int = 4,
        out: Union[str, Path] = None,
        verbose: bool = True) -> pd.DataFrame:
    """Run dia_image over the successful images of process_images/stack summaries; return (and save) the table."""
    paths = load_summary(summaries)
    if len(paths) == 0:
        raise ValueError("no successful images in the summary")
    t0 = time.time()
    tasks = [(p, ra, dec, reference, config) for p in paths]
    if n_proc > 1 and len(paths) > 1:
        with ProcessPoolExecutor(max_workers=min(n_proc, len(paths))) as ex:
            rows = list(ex.map(_worker, tasks))
    else:
        rows = [_worker(t) for t in tasks]
    summary = pd.DataFrame(rows)

    if out is None:
        objname = ScienceImage(path=paths[0], load=True).objname or 'unknown'
        out = Path.home() / 'ezphot' / 'log' / 'recipes' / objname / f"dia_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
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
