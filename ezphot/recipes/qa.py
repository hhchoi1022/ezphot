#!/usr/bin/env python
"""
qa: judge a process_images summary.csv against thresholds. Shared by the photometry skill
and the QA agent so that "pass" means the same thing everywhere.

    python -m ezphot.recipes.qa summary.csv [--zperr-max 0.1] [--nref-min 10] [--seeing-max 5] [--depth-min 17] [--color-max 0.2|none]
Exit code 1 if any image fails.
The colour check needs |K_COLOR_APER_3_g-r| <= color_max (photcal's slope of the 10" ZP residual vs g-r), taken
from the summary column or, when it is not there (older summaries), from the image header of a successful row.
An image without the value passes the colour check, unlike the other checks; --color-max none turns it off.
"""
from __future__ import annotations

import argparse

import pandas as pd
from astropy.io import fits

from ezphot.recipes import defaults as D

ZP_KEY, ZPERR_KEY, DEPTH_KEY = f'ZP_{D.APER10_KEY}', f'ZPERR_{D.APER10_KEY}', f'UL5SKY_{D.APER10_KEY}'
COLOR_KEY = f'K_COLOR_{D.APER10_KEY}_g-r'


def _header_color(path) -> tuple:
    """(COLOR_KEY of the image header, file readable); the value is NaN when the key is missing
    or the file cannot be read (missing, not FITS)."""
    try:
        header = fits.getheader(path)
    except OSError:
        return float('nan'), False
    try:
        return float(header[COLOR_KEY]), True
    except KeyError:
        return float('nan'), True


def qa(summary: pd.DataFrame | str,
       zperr_max: float = 0.1,
       nref_min: int = 10,
       seeing_max: float = 5.0,
       depth_min: float = None,
       color_max: float = 0.2) -> pd.DataFrame:
    """Add boolean check columns and a `qa_pass` column; return the annotated frame."""
    df = pd.read_csv(summary) if isinstance(summary, str) else summary.copy()
    df['chk_success'] = df['success'].astype(bool)
    df['chk_zperr'] = pd.to_numeric(df[ZPERR_KEY], errors='coerce') <= zperr_max
    df['chk_nref'] = pd.to_numeric(df['n_ref'], errors='coerce') >= nref_min
    df['chk_seeing'] = pd.to_numeric(df['SEEING'], errors='coerce') <= seeing_max
    checks = ['chk_success', 'chk_zperr', 'chk_nref', 'chk_seeing']
    if depth_min is not None:
        df['chk_depth'] = pd.to_numeric(df[DEPTH_KEY], errors='coerce') >= depth_min
        checks.append('chk_depth')
    if color_max is not None:
        k = (pd.to_numeric(df[COLOR_KEY], errors='coerce') if COLOR_KEY in df.columns
             else pd.Series(float('nan'), index=df.index))
        need = k.isna() & df['chk_success']          # failed rows are never read
        read = [_header_color(p) for p in df.loc[need, 'path']]
        k.loc[need] = [v for v, _ in read]
        df[COLOR_KEY] = k
        df.attrs['color_unreadable'] = sum(not ok for _, ok in read)   # for the CLI summary line, not a column
        # NaN (no colour fit / no key / unreadable) PASSES here, the opposite of the other checks:
        # a missing colour term is not evidence of a bad colour term.
        df['chk_color'] = k.isna() | (k.abs() <= color_max)
        checks.append('chk_color')
    df['qa_pass'] = df[checks].all(axis=1)
    return df


def _float_or_none(s: str):
    return None if s.lower() == 'none' else float(s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('summary')
    ap.add_argument('--zperr-max', type=float, default=0.1)
    ap.add_argument('--nref-min', type=int, default=10)
    ap.add_argument('--seeing-max', type=float, default=5.0)
    ap.add_argument('--depth-min', type=float)
    ap.add_argument('--color-max', type=_float_or_none, default=0.2, help=f'|{COLOR_KEY}| limit, or none (no colour check)')
    a = ap.parse_args(argv)

    df = qa(a.summary, a.zperr_max, a.nref_min, a.seeing_max, a.depth_min, a.color_max)
    cols = ['path', 'filter', 'SEEING', 'n_ref', ZP_KEY, ZPERR_KEY, DEPTH_KEY]
    cols += [COLOR_KEY] if a.color_max is not None else []
    show = df[cols + ['qa_pass']].copy()
    show['path'] = show['path'].map(lambda p: str(p).split('/')[-1])
    print(show.to_string(index=False))
    n_fail = int((~df['qa_pass']).sum())
    if a.color_max is not None:   # successful rows without a value: no key in summary/header, or FITS unreadable
        n_unread = df.attrs['color_unreadable']
        n_nokey = int((df['chk_success'] & df[COLOR_KEY].isna()).sum()) - n_unread
    print(f"[qa] {len(df) - n_fail}/{len(df)} pass "
          f"(zperr<={a.zperr_max}, n_ref>={a.nref_min}, seeing<={a.seeing_max}"
          + (f", depth>={a.depth_min}" if a.depth_min is not None else '')
          + (f", |{COLOR_KEY}|<={a.color_max}; {n_nokey} without key, {n_unread} unreadable, counted as pass"
             if a.color_max is not None else '') + ")")
    if n_fail:
        bad = df.loc[~df['qa_pass']]
        for _, r in bad.iterrows():
            failed = [c[4:] for c in df.columns if c.startswith('chk_') and not bool(r[c])]
            print(f"  FAIL {str(r['path']).split('/')[-1]}: {','.join(failed)}" + (f" ({r['error']})" if isinstance(r.get('error'), str) else ''))
    return 1 if n_fail else 0


if __name__ == '__main__':
    raise SystemExit(main())
