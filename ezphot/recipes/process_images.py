#!/usr/bin/env python
"""
process_images: single-image chain on ezphot ScienceImage objects.

    [correct_bdf] -> [platesolve] -> invalidmask -> (circular mask @ target) -> sourcemask
    -> bkgmap -> bkgrms -> photometry_sex -> photometric_calibration
    -> photometry_sex + photometric_calibration again with the seeing of the ZP stars from the first pass

Products are written next to each image by ezphot itself (.cat/.refcat/.bkgmap/...).
This recipe adds one summary row per image (status + QA header values) and saves it as CSV.

Python:
    from ezphot.recipes import process_images
    summary = process_images(imageset_or_list, ra=..., dec=..., n_proc=4)

CLI (images are found with DataBrowser('scidata') unless --paths is given):
    python -m ezphot.recipes.process_images --objname EG21 --observatory 7DT --filter m400 \
        --ra 76.3 --dec -68.0 --n-proc 4 [--no-platesolve] [--preprocess] [--out summary.csv]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from ezphot.imageobjects import Errormap, ImageSet, ScienceImage
from ezphot.recipes import defaults as D

STEP_CONFIG_KEYS = ('circularmask', 'invalidmask', 'sourcemask', 'bkgmap', 'bkgrms', 'photometry', 'photcal')
# ezphot writes modified copies with these prefixes; they are not raw inputs.
DERIVED_PREFIXES = ('align_', 'com_', 'coadd_', 'scale_', 'conv_', 'sub_', 'inv_', 'subbkg_')


def _config(overrides: Optional[dict]) -> dict:
    cfg = {k: dict(getattr(D, k.upper())) for k in STEP_CONFIG_KEYS}
    for k, v in (overrides or {}).items():
        if k not in cfg:
            raise KeyError(f"unknown config section {k!r}; expected one of {STEP_CONFIG_KEYS}")
        cfg[k].update(v)
    return cfg


def _needs_platesolve(img: ScienceImage) -> bool:
    """True when the frame has no WCS, or only an astrometry.net SIP solution: SExtractor ignores SIP,
    so the SCAMP (PV) step must run before photometry."""
    h = img.header
    if 'CTYPE1' not in h or 'SIP' in str(h['CTYPE1']):
        return True
    return not any(k.startswith('PV1_') for k in h)


def process_image(target_img: ScienceImage,
                  ra: float = None,
                  dec: float = None,
                  do_preprocess: bool = False,
                  do_platesolve: bool = True,
                  config: dict = None,
                  mag_faint_shift: float = 0.0,
                  clear: bool = True,
                  bkgrms: Errormap = None) -> dict:
    """Run the chain on one image. Never raises; returns a summary row with success/error.

    mag_faint_shift : added to the per-filter reference-star faint limit (mag_upper), e.g. 1.5
    to include fainter stars in the ZP fit and the ZP figures.
    bkgrms : use this error map instead of calculating one (stacks keep the propagated bkgrms).
    """
    cfg = _config(config)
    row = dict(path=str(target_img.path), telname=target_img.telname, filter=target_img.filter,
               obsdate=target_img.obsdate, exptime=target_img.exptime,
               success=False, step=None, error=None, n_ref=None)
    for k in D.SUMMARY_HEADER_KEYS:
        row[k] = None
    ivpmask = srcmask = bkg = catalog = refcat = None
    try:
        if do_preprocess:
            row['step'] = 'correct_bdf'
            target_img = target_img.correct_bdf(save=True, verbose=False)
        if do_platesolve and _needs_platesolve(target_img):
            row['step'] = 'platesolve'
            target_img = target_img.platesolve(verbose=False)
        row['step'] = 'invalidmask'
        ivpmask = target_img.calculate_invalidmask(**cfg['invalidmask'])
        if ra is not None and dec is not None:
            row['step'] = 'circularmask'
            srcmask = target_img.calculate_circularmask(
                target_srcmask=None, x_position=ra, y_position=dec,
                radius_arcsec=cfg['circularmask']['radius_arcsec'], unit='coord',
                save=False, verbose=False, visualize=False, save_fig=False)
        row['step'] = 'sourcemask'
        srcmask = target_img.calculate_sourcemask(target_srcmask=srcmask, **cfg['sourcemask'])
        row['step'] = 'bkgmap'
        bkg = target_img.calculate_bkg(target_srcmask=srcmask, target_ivpmask=ivpmask, **cfg['bkgmap'])
        row['step'] = 'bkgrms'
        if bkgrms is None:
            bkgrms = target_img.calculate_bkgrms(target_srcmask=srcmask, target_ivpmask=ivpmask, **cfg['bkgrms'])
        row['step'] = 'photometry'
        catalog = target_img.photometry_sex(target_bkg=bkg, target_bkgrms=bkgrms, target_mask=None,
                                            sex_params=None, **cfg['photometry'])
        row['step'] = 'photometric_calibration'
        photcal = dict(cfg['photcal'])
        photcal['mag_lower'], photcal['mag_upper'] = D.MAG_RANGE.get(target_img.filter, D.MAG_RANGE_DEFAULT)
        photcal['mag_upper'] += mag_faint_shift
        photcal['ra_deg'], photcal['dec_deg'] = ra, dec
        target_img, catalog, refcat, photcal_update = target_img.photometric_calibration(target_catalog=catalog, **photcal)
        # photometry_sex never reads the header SEEING (it would depend on earlier runs), so feed it the
        # ZP-star seeing of this run: SEEING_FWHM (-> CLASS_STAR star selection) and seeing-scaled apertures.
        if 'SEEING' in photcal_update:
            row['step'] = 'photometry_pass2'
            catalog = target_img.photometry_sex(target_bkg=bkg, target_bkgrms=bkgrms, target_mask=None, sex_params=None,
                                                **dict(cfg['photometry'], seeing_arcsec=photcal_update['SEEING'][0]))
            row['step'] = 'photometric_calibration_pass2'
            target_img, catalog, refcat, _ = target_img.photometric_calibration(target_catalog=catalog, **photcal)
        row['n_ref'] = refcat.nsources if refcat is not None else None
        header = target_img.header
        for k in D.SUMMARY_HEADER_KEYS:
            row[k] = header.get(k)
        row['success'] = True
        row['step'] = 'done'
    except Exception as e:
        row['error'] = f"{type(e).__name__}: {e}"
    finally:
        if clear:
            for obj in (target_img, ivpmask, srcmask, bkg, bkgrms, catalog, refcat):
                if obj is not None:
                    try:
                        obj.clear(verbose=False)
                    except Exception:
                        pass
    return row


def _worker(args):
    return process_image(*args)


def process_images(images: Union[ImageSet, List[ScienceImage]],
                   ra: float = None,
                   dec: float = None,
                   do_preprocess: bool = False,
                   do_platesolve: bool = True,
                   config: dict = None,
                   mag_faint_shift: float = 0.0,
                   n_proc: int = 4,
                   out: Union[str, Path] = None,
                   verbose: bool = True) -> pd.DataFrame:
    """Run process_image over an ImageSet / list and return (and save) the summary table."""
    imgs = list(images.target_images) if isinstance(images, ImageSet) else list(images)
    if len(imgs) == 0:
        raise ValueError("no images to process")
    t0 = time.time()
    tasks = [(img, ra, dec, do_preprocess, do_platesolve, config, mag_faint_shift, True) for img in imgs]
    if n_proc > 1 and len(imgs) > 1:
        with ProcessPoolExecutor(max_workers=min(n_proc, len(imgs))) as ex:
            rows = list(ex.map(_worker, tasks))
    else:
        rows = [_worker(t) for t in tasks]
    summary = pd.DataFrame(rows)

    if out is None:
        objname = imgs[0].objname or 'unknown'
        out = Path.home() / 'ezphot' / 'log' / 'recipes' / objname / f"process_images_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    if verbose:
        n_ok = int(summary['success'].sum())
        print(f"[process_images] {n_ok}/{len(summary)} images succeeded in {time.time() - t0:.0f}s")
        if n_ok < len(summary):
            print(summary.loc[~summary['success'], ['path', 'step', 'error']].to_string(index=False))
        print(f"[process_images] summary -> {out}")
    return summary


def load_images(observatory: str = None, telkey: str = None, telname: str = None,
                objname: str = None, filter: str = None, obsdate: str = None,
                pattern: str = '*.fits', paths: List[str] = None) -> ImageSet:
    """Build an ImageSet of raw science images, from explicit paths or DataBrowser('scidata')."""
    if paths:
        return ImageSet([ScienceImage(path=p, load=True) for p in paths])
    from ezphot.utils import DataBrowser
    db = DataBrowser('scidata')
    for key, val in dict(observatory=observatory, telkey=telkey, telname=telname,
                         objname=objname, filter=filter, obsdate=obsdate).items():
        if val is not None:
            setattr(db, key, val)
    imgset = db.search(pattern=pattern, return_type='science')
    raw = [img for img in imgset.target_images if not img.path.name.startswith(DERIVED_PREFIXES)]
    return ImageSet(raw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths', nargs='+', help='explicit FITS paths (skips DataBrowser search)')
    ap.add_argument('--observatory'); ap.add_argument('--telkey'); ap.add_argument('--telname')
    ap.add_argument('--objname'); ap.add_argument('--filter'); ap.add_argument('--obsdate')
    ap.add_argument('--pattern', default='*.fits')
    ap.add_argument('--ra', type=float); ap.add_argument('--dec', type=float)
    ap.add_argument('--preprocess', action='store_true', help='run correct_bdf first')
    ap.add_argument('--no-platesolve', action='store_true')
    ap.add_argument('--faint-shift', type=float, default=0.0,
                    help='add this to every filter\'s reference-star faint limit (mag_upper)')
    ap.add_argument('--n-proc', type=int, default=4)
    ap.add_argument('--max-images', type=int, help='process only the first N images (testing)')
    ap.add_argument('--out', help='summary CSV path')
    a = ap.parse_args(argv)

    imgset = load_images(a.observatory, a.telkey, a.telname, a.objname, a.filter, a.obsdate, a.pattern, a.paths)
    imgs = imgset.target_images[:a.max_images] if a.max_images else imgset.target_images
    print(f"[process_images] {len(imgs)} images")
    summary = process_images(imgs, ra=a.ra, dec=a.dec, do_preprocess=a.preprocess,
                             do_platesolve=not a.no_platesolve, mag_faint_shift=a.faint_shift,
                             n_proc=a.n_proc, out=a.out)
    return 0 if summary['success'].all() else 1


if __name__ == '__main__':
    raise SystemExit(main())
