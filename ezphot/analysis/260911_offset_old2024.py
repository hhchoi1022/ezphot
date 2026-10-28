#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_offset_old2024.py

delta* (colour-term-zeroing offset on the physical curve) for the OLD standard-field dataset that the global
fitted delta_f was derived from (2024-08 campaign, 12k frames) — WITHOUT re-photometry: per-star inst mags are
reconstructed from refit_stars.csv.gz (inst = ref_mag - zp) and the corrected-XP spectra come from the saved
refstar_spectra.npz.  Same solver as 260911_offset_timeseries (1 nm grid, linear root).  Output has the same
schema so it can be merged into the F28 time series.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_offset_old2024.py
"""
from __future__ import annotations

import importlib
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
OT = importlib.import_module("260911_offset_timeseries")     # DGRID, solve_offset, C(load_new_curve), GRID
C = OT.C

ZPC = Path("/home/hhchoi1022/ezphot/log/20260908_zp_color_check")
FOLDER = OT.FOLDER


def meta_std(fr):
    """unit, night, mjd from an old std-frame filename: 7DT01_20240809_033842_...fits"""
    u = re.search(r"(7DT\d\d)", fr)
    m = re.search(r"_(\d{8})_(\d{6})_", fr)
    if not m:
        return (u.group(1) if u else "?", "?", np.nan)
    s, t = m.group(1), m.group(2)
    night = f"{s[:4]}-{s[4:6]}-{s[6:]}"
    mjd = Time(f"{night}T{t[:2]}:{t[2:4]}:{t[4:]}").mjd
    return (u.group(1) if u else "?", night, float(mjd))


def main():
    z = np.load(ZPC / "shift_recal" / "refstar_spectra.npz", allow_pickle=True)
    flux_of = {str(i): f for i, f in zip(z["ids"], z["flux"])}
    print(f"XP spectra: {len(flux_of)} stars", flush=True)

    s = pd.read_csv(ZPC / "refit_stars.csv.gz")
    s = s[(s.catalog == "GAIAXP_CORR_LAMOST") & s["filter"].isin(OT.GRID)]
    s["ref_id"] = s.ref_id.astype(str)
    s = s[s.ref_id.isin(flux_of) & (s.magerr < 0.1) & np.isfinite(s.bprp)]
    s["inst"] = s.ref_mag - s.zp                                  # reconstruct instrumental mag
    s["sig"] = np.sqrt(s.magerr ** 2 + 0.01 ** 2)
    print(f"star rows: {len(s)}, frames: {s.file.nunique()}", flush=True)

    phys = {f: C.load_new_curve(f) for f in OT.GRID}
    rows = []
    for (filt, fr), g in s.groupby(["filter", "file"]):
        if len(g) < 15:
            continue
        fc = np.vstack([flux_of[i] for i in g.ref_id])
        dstar, derr, a0 = OT.solve_offset(*phys[filt], fc, g.inst.to_numpy(), g.bprp.to_numpy(), g.sig.to_numpy())
        if not np.isfinite(dstar):
            continue
        unit, night, mjd = meta_std(fr)
        rows.append(dict(filter=filt, unit=unit, night=night, mjd=mjd, tile=str(g.objname.iloc[0]), n=len(g),
                         dstar=dstar, dstar_err=derr, a_phys=a0))
        if len(rows) % 2000 == 0:
            print(f"  {len(rows)} frames done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(FOLDER / "offset_timeseries_std2024.csv", index=False)
    print(f"\n{len(res)} frames; filters {res['filter'].nunique()}; MJD {res.mjd.min():.0f}..{res.mjd.max():.0f}")
    med = res.groupby("filter").agg(dstar_med=("dstar", "median"),
                                    mad=("dstar", lambda v: float(np.median(np.abs(v - np.median(v))))),
                                    n=("dstar", "size"))
    med = med.reindex([f for f in OT.GRID if f in med.index])
    print(med.round(2).to_string())
    print(f"\nsaved -> offset_timeseries_std2024.csv in {FOLDER}")


if __name__ == "__main__":
    main()
