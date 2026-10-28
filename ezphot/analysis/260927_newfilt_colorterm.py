#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260927_newfilt_colorterm.py — per-image corrected-XP field colour term for the new-generation (in-between) filters,
2026 survey frames re-photometered by 260927_newfilt_rephot.py (new background).

Stars: Step 1 selection (260926_delta_star_v2.select_stars_v2: FLAGS 0, SNR_10" >= 100, FLUX_MAX <= 50k, isolated 10",
no shape cuts), corrected Gaia XP match < 2".  Colour term = weighted line of ZP vs BP-RP after a 3-sigma ZP clip (same as
260926_colorterm_time.colour_term).
Curves (x QE x optics x El Sauce atmosphere, vacuum; atmosphere on its lam_air column):
  manufacturer_{nobeam,beam}, hyp_{nobeam,beam} (version_hyp; not m832w, defective), ky_{nobeam,beam} (m438, m483, m512)
Output (~/ezphot/log/20260927_newfilt/): colorterm_newfilt_frames.csv
"""
import argparse
import importlib
import sys
import time
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.io import fits                                      # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
DS = importlib.import_module("260926_delta_star_v2")
CT = importlib.import_module("260926_colorterm_time")
C, K = DS.C, DS.K
OUT = Path("/home/hhchoi1022/ezphot/log/20260927_newfilt")
CATD = OUT / "catalogs"
BAD_HYP = {"m832w"}
_CURVES = None


def build_curves(filters):
    atm = pd.read_csv(DS.TS / "atmosphere_elsauce.csv")
    qe = pd.read_csv(DS.SRC / "QE.csv")
    opt = pd.read_csv(DS.SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    out = defaultdict(dict)
    for f in filters:
        for ver, key in (("manufacturer", "manufacturer"), ("version_hyp", "hyp"), ("version_ky", "ky")):
            p = DS.VBASE / ver / f"{f}.csv"
            if not p.exists() or (key == "hyp" and f in BAD_HYP):
                continue
            d = pd.read_csv(p).sort_values("lam_nm")
            lam = d.lam_nm.to_numpy(float); tr = np.clip(d.trans.to_numpy(float), 0, None)
            env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
                   * np.interp(lam, atm["lam_air"], atm["trans"]))
            te = DS.dr500_transmission(lam, tr, n_eff=1.5, f_number=3.03, obscuration=0.59)
            lv = DS.air_to_vac_nm(lam)
            out[f"{key}_nobeam"][f] = (lv, tr * env)
            out[f"{key}_beam"][f] = (lv, te * env)
    return dict(out)


def _init(filters):
    global _CURVES
    warnings.filterwarnings("ignore")
    K._refit_init()
    _CURVES = build_curves(filters)


def process_tile(args):
    tile, frames = args
    try:
        matched, need = [], set()
        for fr, filt, unit, night in frames:
            cp = CATD / (Path(fr).name[:-5] + "_cat.fits")
            d = DS.select_stars_v2(cp)
            if len(d) < 15:
                continue
            mjd = float(fits.getheader(cp).get("MJD", np.nan))
            got = K.get_reference("GAIAXP_CORR_LAMOST", tile, float(np.median(d.ALPHA_J2000)), float(np.median(d.DELTA_J2000)), 1.34, 0.90)
            if got is None or got[1] is None or len(got[1]) == 0:
                continue
            tbl, coords, _ = got
            idx, d2d, _ = match_coordinates_sky(SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg"), coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            if ok.sum() < 15:
                continue
            mr, di = tbl[idx[ok]], d[ok]
            for rid, bp, m, e in zip(np.asarray(mr["id"]).astype(str), np.asarray(mr["bp-rp"], float), di.MAG_APER_3, di.MAGERR_APER_3):
                matched.append((filt, fr, unit, night, mjd, rid, bp, float(m), float(e)))
                need.add(rid)
        if not matched:
            return []
        xp = C.tile_xp(tile, need)
        M = pd.DataFrame(matched, columns=["filter", "file", "unit", "night", "mjd", "ref_id", "bprp", "inst", "ierr"])
        M = M[M.ref_id.isin(xp)]
        M["sig"] = np.sqrt(M.ierr.to_numpy() ** 2 + 0.01 ** 2)
        out = []
        for (filt, fr), g in M.groupby(["filter", "file"]):
            if len(g) < 15:
                continue
            fc = np.vstack([xp[i][1] for i in g.ref_id])
            base = dict(filter=filt, file=Path(fr).name, unit=g.unit.iloc[0], night=g.night.iloc[0], mjd=g.mjd.iloc[0], tile=tile)
            for v, curves in _CURVES.items():
                if filt not in curves:
                    continue
                a, ae, n = CT.colour_term(*curves[filt], fc, g.inst.to_numpy(), g.bprp.to_numpy(), g.sig.to_numpy())
                if np.isfinite(a):
                    out.append(dict(base, variant=v, n=n, a0=round(float(a), 5), a0_err=round(float(ae), 5)))
        return out
    except Exception:                                                 # noqa: BLE001
        return []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=32)
    a = ap.parse_args()
    S = pd.read_csv(OUT / "frames.csv")
    M = pd.read_csv(OUT / "rephot_meta.csv")
    ok = set(M[M.status.isin(["ok", "cached"])].file)
    S = S[S.file.isin(ok)]
    filters = sorted(S["filter"].unique(), key=lambda f: int(f[1:4]))
    by_tile = defaultdict(list)
    for r in S.itertuples():
        by_tile[r.tile].append((r.file, r.filter, r.unit, r.night))
    items = sorted(by_tile.items(), key=lambda kv: -len(kv[1]))
    print(time.strftime("[%H:%M:%S]"), f"{len(S)} frames, {len(items)} tiles, filters {filters}", flush=True)
    rows = []
    with Pool(a.n_proc, initializer=_init, initargs=(filters,)) as pool:
        for r in pool.imap_unordered(process_tile, items, chunksize=1):
            rows += r
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "colorterm_newfilt_frames.csv", index=False)
    print(time.strftime("[%H:%M:%S]"), f"saved {len(R)} rows, {R.file.nunique()} frames; variants {sorted(R.variant.unique())}", flush=True)


if __name__ == "__main__":
    main()
