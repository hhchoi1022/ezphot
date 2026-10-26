#!/usr/bin/env python
"""
qa: judge a process_images summary.csv against thresholds. Shared by the photometry skill
and the QA agent so that "pass" means the same thing everywhere.

    python -m ezphot.recipes.qa summary.csv [--zperr-max 0.1] [--nref-min 10] [--seeing-max 5] [--depth-min 17]
Exit code 1 if any image fails.
"""
from __future__ import annotations

import argparse

import pandas as pd

from ezphot.recipes import defaults as D

ZP_KEY, ZPERR_KEY, DEPTH_KEY = f'ZP_{D.APER10_KEY}', f'ZPERR_{D.APER10_KEY}', f'UL5SKY_{D.APER10_KEY}'


def qa(summary: pd.DataFrame | str,
       zperr_max: float = 0.1,
       nref_min: int = 10,
       seeing_max: float = 5.0,
       depth_min: float = None) -> pd.DataFrame:
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
    df['qa_pass'] = df[checks].all(axis=1)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('summary')
    ap.add_argument('--zperr-max', type=float, default=0.1)
    ap.add_argument('--nref-min', type=int, default=10)
    ap.add_argument('--seeing-max', type=float, default=5.0)
    ap.add_argument('--depth-min', type=float)
    a = ap.parse_args(argv)

    df = qa(a.summary, a.zperr_max, a.nref_min, a.seeing_max, a.depth_min)
    cols = ['path', 'filter', 'SEEING', 'n_ref', ZP_KEY, ZPERR_KEY, DEPTH_KEY, 'qa_pass']
    show = df[cols].copy()
    show['path'] = show['path'].map(lambda p: str(p).split('/')[-1])
    print(show.to_string(index=False))
    n_fail = int((~df['qa_pass']).sum())
    print(f"[qa] {len(df) - n_fail}/{len(df)} pass "
          f"(zperr<={a.zperr_max}, n_ref>={a.nref_min}, seeing<={a.seeing_max}"
          + (f", depth>={a.depth_min}" if a.depth_min is not None else '') + ")")
    if n_fail:
        bad = df.loc[~df['qa_pass']]
        for _, r in bad.iterrows():
            failed = [c[4:] for c in df.columns if c.startswith('chk_') and not bool(r[c])]
            print(f"  FAIL {str(r['path']).split('/')[-1]}: {','.join(failed)}" + (f" ({r['error']})" if isinstance(r.get('error'), str) else ''))
    return 1 if n_fail else 0


if __name__ == '__main__':
    raise SystemExit(main())
