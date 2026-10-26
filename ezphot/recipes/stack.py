#!/usr/bin/env python
"""
stack: per-night stacks of process_images outputs, then the process_images chain on each stack.

    process_images summary (success rows) -> divide_images (filter, observatory, obsdate +-0.5 d)
    -> prepare_stack (ZP_APER_3 scaling, LANCZOS3 reproject) -> stack (median, extrema clip)
    -> invalidmask -> (circular mask @ target) -> sourcemask -> bkgmap -> photometry_sex
    -> photometric_calibration (2-pass, see process_images). No platesolve; the bkgrms propagated
    by the stacker is kept (as bridge stacked_process does). Groups of one image are not stacked.

Python:
    from ezphot.recipes import stack
    summary = stack('~/ezphot/log/recipes/EG21/process_images_....csv', ra=..., dec=..., n_proc=4)

CLI:
    python -m ezphot.recipes.stack <process_images summary.csv> [more.csv ...] \
        [--ra 76.3 --dec -68.0] [--n-proc 4] [--out stack.csv]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Union

import pandas as pd

from ezphot.imageobjects import ImageSet, ScienceImage
from ezphot.recipes import defaults as D
from ezphot.recipes.process_images import process_image

STACK_CONFIG_KEYS = ('stack_group', 'stack_prepare', 'stack')


def _split_config(overrides: dict = None):
    """Stack sections (with defaults) and the rest, which goes to process_image."""
    overrides = dict(overrides or {})
    cfg = {k: dict(getattr(D, k.upper()), **overrides.pop(k, {})) for k in STACK_CONFIG_KEYS}
    return cfg, overrides


def load_summary(summaries: Union[str, Path, List]) -> List[str]:
    """Paths of the successfully processed images in one or more process_images summary CSVs."""
    paths = []
    for f in ([summaries] if isinstance(summaries, (str, Path)) else summaries):
        df = pd.read_csv(Path(f).expanduser())
        paths += df.loc[df['success'].astype(bool), 'path'].tolist()
    return list(dict.fromkeys(paths))


def stack_group(paths: List[str],
                ra: float = None,
                dec: float = None,
                config: dict = None,
                mag_faint_shift: float = 0.0) -> dict:
    """Stack one group and calibrate the stack. Never raises; returns a summary row."""
    cfg, process_config = _split_config(config)
    imgset = None
    try:
        imgset = ImageSet([ScienceImage(path=p, load=True) for p in paths])
        imgset.prepare_stack(**cfg['stack_prepare'])
        stacked_img, stacked_bkgrms = imgset.stack(**cfg['stack'])
    except Exception as e:
        img = imgset.images[0] if imgset is not None else None
        row = dict(path=None, telname=getattr(img, 'telname', None), filter=getattr(img, 'filter', None),
                   obsdate=getattr(img, 'obsdate', None), exptime=getattr(img, 'exptime', None),
                   success=False, step='stack', error=f"{type(e).__name__}: {e}", n_ref=None)
    else:
        row = process_image(stacked_img, ra=ra, dec=dec, do_platesolve=False, config=process_config,
                            mag_faint_shift=mag_faint_shift, bkgrms=stacked_bkgrms)
    finally:
        if imgset is not None:
            imgset.clear()
    row['n_combined'] = len(paths)
    row['inputs'] = ';'.join(paths)
    return row


def _worker(args):
    return stack_group(*args)


def stack(summaries: Union[str, Path, List],
          ra: float = None,
          dec: float = None,
          config: dict = None,
          mag_faint_shift: float = 0.0,
          n_proc: int = 4,
          out: Union[str, Path] = None,
          verbose: bool = True) -> pd.DataFrame:
    """Group, stack and calibrate the images of process_images summaries; return (and save) one row per stack."""
    paths = load_summary(summaries)
    if len(paths) == 0:
        raise ValueError("no successfully processed images in the summary")
    cfg, _ = _split_config(config)
    groups = ImageSet([ScienceImage(path=p, load=True) for p in paths]).divide_images(**cfg['stack_group'])
    groups = [[str(img.path) for img in g.target_images] for g in groups]
    n_single = sum(len(g) == 1 for g in groups)
    groups = [g for g in groups if len(g) > 1]
    if verbose:
        print(f"[stack] {len(paths)} images -> {len(groups)} stacks ({n_single} single-image groups not stacked)")
    if len(groups) == 0:
        raise ValueError("no group has more than one image")
    t0 = time.time()
    tasks = [(g, ra, dec, config, mag_faint_shift) for g in groups]
    if n_proc > 1 and len(groups) > 1:
        with ProcessPoolExecutor(max_workers=min(n_proc, len(groups))) as ex:
            rows = list(ex.map(_worker, tasks))
    else:
        rows = [_worker(t) for t in tasks]
    summary = pd.DataFrame(rows)

    if out is None:
        objname = ScienceImage(path=paths[0], load=True).objname or 'unknown'
        out = Path.home() / 'ezphot' / 'log' / 'recipes' / objname / f"stack_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    if verbose:
        n_ok = int(summary['success'].sum())
        print(f"[stack] {n_ok}/{len(summary)} stacks succeeded in {time.time() - t0:.0f}s")
        if n_ok < len(summary):
            print(summary.loc[~summary['success'], ['filter', 'telname', 'step', 'error']].to_string(index=False))
        print(f"[stack] summary -> {out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('summaries', nargs='+', help='process_images summary CSV(s)')
    ap.add_argument('--ra', type=float); ap.add_argument('--dec', type=float)
    ap.add_argument('--faint-shift', type=float, default=0.0,
                    help='add this to every filter\'s reference-star faint limit (mag_upper)')
    ap.add_argument('--n-proc', type=int, default=4)
    ap.add_argument('--out', help='summary CSV path')
    a = ap.parse_args(argv)
    summary = stack(a.summaries, ra=a.ra, dec=a.dec, mag_faint_shift=a.faint_shift, n_proc=a.n_proc, out=a.out)
    return 0 if summary['success'].all() else 1


if __name__ == '__main__':
    raise SystemExit(main())
