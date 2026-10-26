#!/usr/bin/env python
"""
compare_summary: regression check of a new process_images summary.csv against a baseline
(tests/baseline/<objname>_summary.csv). Used by the ezphot-verifier agent after a code change.

    python tests/compare_summary.py tests/baseline/T10796_summary.csv new_summary.csv \
        [--zp-tol 0.002] [--zperr-tol 0.001] [--nref-tol 5] [--seeing-tol 0.1] [--depth-tol 0.01]
Rows are matched by image file name. Exit code 1 if any row is missing/extra, changes success/step,
or moves by more than a tolerance. Default tolerances cover the run-to-run scatter seen on
T05785 (2026-10-23: |dZP| <= 0.0009, |dn_ref| <= 3, |dSEEING| <= 0.031) and T10796 (2026-10-24:
|dSEEING| up to 0.074 on a frame with ~45 ZP stars, since SEEING is their median FWHM).
"""
from __future__ import annotations

import argparse

import pandas as pd

from ezphot.recipes import defaults as D

ZP_KEY, ZPERR_KEY, DEPTH_KEY = f'ZP_{D.APER10_KEY}', f'ZPERR_{D.APER10_KEY}', f'UL5SKY_{D.APER10_KEY}'


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['name'] = df['path'].map(lambda p: str(p).split('/')[-1])
    return df.set_index('name')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('baseline')
    ap.add_argument('new')
    ap.add_argument('--zp-tol', type=float, default=0.002)
    ap.add_argument('--zperr-tol', type=float, default=0.001)
    ap.add_argument('--nref-tol', type=int, default=5)
    ap.add_argument('--seeing-tol', type=float, default=0.1)
    ap.add_argument('--depth-tol', type=float, default=0.01)
    a = ap.parse_args(argv)

    old, new = _load(a.baseline), _load(a.new)
    problems = [f"missing in new: {n}" for n in old.index.difference(new.index)]
    problems += [f"extra in new: {n}" for n in new.index.difference(old.index)]

    tols = {ZP_KEY: a.zp_tol, ZPERR_KEY: a.zperr_tol, 'n_ref': a.nref_tol,
            'SEEING': a.seeing_tol, DEPTH_KEY: a.depth_tol}
    common = old.index.intersection(new.index)
    for n in common:
        o, w = old.loc[n], new.loc[n]
        for c in ('success', 'step'):
            if str(o[c]) != str(w[c]):
                problems.append(f"{n}: {c} {o[c]} -> {w[c]}")
        for c, tol in tols.items():
            vo, vw = pd.to_numeric(o[c], errors='coerce'), pd.to_numeric(w[c], errors='coerce')
            if pd.isna(vo) and pd.isna(vw):
                continue
            if pd.isna(vo) != pd.isna(vw) or abs(vw - vo) > tol:
                problems.append(f"{n}: {c} {vo} -> {vw} (tol {tol})")

    print(f"[compare] {len(common)} matched rows (baseline {len(old)}, new {len(new)})")
    for c in tols:
        d = (pd.to_numeric(new.loc[common, c], errors='coerce') - pd.to_numeric(old.loc[common, c], errors='coerce')).abs()
        print(f"  max |d{c}| = {d.max():.5g}")
    for p in problems:
        print(f"  FAIL {p}")
    print(f"[compare] {'FAIL' if problems else 'PASS'} ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
