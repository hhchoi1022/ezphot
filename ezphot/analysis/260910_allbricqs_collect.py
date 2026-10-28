#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_allbricqs_collect.py

Collect the products of the newly processed AllBRICQS survey tiles (260910_allbricqs_process.py) into the same tables the
2026-09-08/10 analysis chain uses, so that 260910_allbricqs_photometry.py can run on them:

  harvest : FITS headers + sidecars of the medium-band single frames of the given tiles -> <out>/frames.csv,
            colorterms_header.csv           (260908_zpcolor_collect.harvest_one)
  refit   : re-match the .refcat stars to the GAIAXP archives, recompute ZP and colour terms
            -> refit_frames.csv, refit_stars.csv.gz   (260908_zpcolor_collect.stage_refit)
  extract : XP spectra (flux_cor) of the reference stars from the doabdi tiles -> tilestar_spectra.npz
            (260910_effcurve_fit.extract_tile)

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260910_allbricqs_collect.py --stage all --tiles T05785 T10796 --n-proc 8
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
C = importlib.import_module("260910_effcurve_common")
K = importlib.import_module("260908_zpcolor_collect")
FIT = importlib.import_module("260910_effcurve_fit")
R = C.R
warnings.filterwarnings("ignore")
log = C.log

SCIDATA = Path("/home/hhchoi1022/ezphot/data/scidata/7DT/7DT_C361K_HIGH_1x1")
DEFAULT_OUT = Path("/home/hhchoi1022/ezphot/log/20260910_allbricqs/collect")
FRAME_RE = re.compile(r"^7DT\d\d_\d{8}_\d{6}_T\d{5}_m\d{3}w?_1x1_[\d.]+s_\d{4}\.fits$")


def list_frames(tiles):
    files = []
    for t in tiles:
        for p in sorted((SCIDATA / t).glob("7DT??/m*/*.fits")):
            if FRAME_RE.match(p.name): files.append(p)
    return files


def stage_harvest(out: Path, tiles, n_proc):
    files = list_frames(tiles)
    log(f"harvest: {len(files)} medium-band single frames in {len(tiles)} tiles")
    rows, longs = [], []
    with Pool(max(1, min(n_proc, len(files)))) as pool:
        for row, long in pool.imap_unordered(K.harvest_one, [str(f) for f in files], chunksize=4):
            rows.append(row); longs.extend(long)
    df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    K.write_csv_atomic(df, out / "frames.csv"); K.write_csv_atomic(pd.DataFrame(longs), out / "colorterms_header.csv")
    log(f"harvest: {len(df)} rows, {df['ZP_APER_3'].notna().sum()} with ZP_APER_3, {int(df['has_refcat'].sum())} with .refcat, {int(df['has_cat'].sum())} with .cat")
    print(df.groupby(["OBJECT", "FILTER"]).size().unstack(fill_value=0).to_string())


def stage_extract(out: Path, n_proc):
    rf = pd.read_csv(out / "refit_frames.csv", low_memory=False)
    rf = rf[(rf.catalog == R.CATALOG) & rf.ref_tiles.notna()]
    tiles_of = {o: sorted({t for s in g.ref_tiles.unique() for t in str(s).split(";") if t}) for o, g in rf.groupby("objname")}
    st = pd.read_csv(out / "refit_stars.csv.gz", usecols=["catalog", "objname", "ref_id"], dtype={"ref_id": str})
    st = st[(st.catalog == R.CATALOG) & st.objname.isin(tiles_of)]
    ids_of = {o: set(g.ref_id.unique()) for o, g in st.groupby("objname")}
    wanted_by_tile = {}
    for o, ids in ids_of.items():
        for t in tiles_of[o]: wanted_by_tile.setdefault(t, set()).update(ids)
    jobs = sorted(wanted_by_tile.items())
    log(f"extract: {sum(len(v) for v in ids_of.values())} reference stars from {len(jobs)} raw XP tiles")
    spectra, errors, cautions, tile_of = {}, {}, {}, {}
    with Pool(max(1, min(n_proc, len(jobs)))) as pool:
        for tile, found, status in pool.imap_unordered(FIT.extract_tile, jobs):
            log(f"  {tile}: {status}, {len(found)} stars")
            for sid, (fl, er, cau) in found.items(): spectra[sid] = fl; errors[sid] = er; cautions[sid] = cau; tile_of[sid] = tile
    ids = sorted(spectra); obj_of = {}
    for o, ss in ids_of.items():
        for s in ss: obj_of.setdefault(s, o)
    np.savez_compressed(out / "tilestar_spectra.npz", ids=np.array(ids), flux=np.vstack([spectra[s] for s in ids]), flux_err=np.vstack([errors[s] for s in ids]),
                        caution=np.array([cautions[s] for s in ids]), wave_nm=R.XP_WAVE_NM, objname=np.array([obj_of[s] for s in ids]), tile=np.array([tile_of[s] for s in ids]))
    log(f"extract: {len(ids)} spectra saved ({sum(len(v) for v in ids_of.values()) - len(ids)} reference stars not found)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["harvest", "refit", "extract", "all"]); ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT)); ap.add_argument("--n-proc", type=int, default=8)
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("harvest", "all"): stage_harvest(out, a.tiles, a.n_proc)
    if a.stage in ("refit", "all"): K.stage_refit(out, None, None, 0, a.n_proc)
    if a.stage in ("extract", "all"): stage_extract(out, a.n_proc)


if __name__ == "__main__":
    main()
