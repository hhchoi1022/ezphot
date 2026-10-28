#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_tilestd_analyze.py

Combine the survey-tile standard frames (260911_tilestd_harvest.py) with the 2024-08 standard-field data and
re-run the corrected-XP closed loop (field colour term + CALSPEC residual) for the nominal vs physical curve,
split by dataset (standard-field only / survey-tile only / combined).  Corrected (LAMOST) XP only.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python \\
      ezphot/analysis/260911_tilestd_analyze.py
"""
from __future__ import annotations

import glob
import importlib
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
from astropy.table import Table     # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
R = importlib.import_module("260908_zpcolor_shift_recal")

LOG = Path("/home/hhchoi1022/ezphot/log")
ZPC = LOG / "20260908_zp_color_check"
TILE = LOG / "20260911_tilestd"
TS = LOG / "20260910_transmission_sources"
PHYS = TS / "total_curves_physical"
CALSPEC_DIR = Path("/qso/data3/doabdi/intern/synthetic/calspec_data")
TILESRC = Path("/lyman/data1/factory_doabdi/GAIA_BEST/final_result")
GRID = R.GRID
XP_WAVE = R.XP_WAVE_NM
STD_TILE = {"HZ4": "T22272", "LDS749B": "T19136", "WD2039-682": "T01485", "BD-11_3759": "T14891"}


def load_new_curve(f):
    d = pd.read_csv(PHYS / f"{f}.csv"); return d.lam.to_numpy(), d.trans.to_numpy()


def extract_xp_cor(ref_ids, centers):
    """flux_cor (343,) for the given Gaia source_ids, scanning final_result tiles near the standards."""
    tiles = pd.read_csv("/lyman/data1/factory_hhchoi/code/ezphot/ezphot/utils/tileinfo/7-DT/final_tiles.txt", sep=r"\s+")
    want = set(str(i) for i in ref_ids)
    cand = set()
    for ra, dec in centers:
        dd = np.abs(tiles.dec.values - dec)
        dr = np.abs(((tiles.ra.values - ra + 180) % 360) - 180) * np.cos(np.radians(dec))
        for t in tiles.id.values[(dd < 1.2) & (dr < 1.2)]:
            if (TILESRC / f"{t}.csv").exists():
                cand.add(t)
    out = {}
    NW = 343
    for t in sorted(cand):
        with open(TILESRC / f"{t}.csv") as fh:
            fh.readline()
            for line in fh:
                c = line.rstrip("\n").split(",")
                sid = c[0].lstrip("﻿")
                if sid not in want or sid in out:
                    continue
                v = np.fromstring(c[13], sep=";")
                if v.size == NW:
                    out[sid] = v
    return out


def main():
    # ---- tile harvest ----
    st_tile = pd.concat([pd.read_csv(f, dtype={"ref_id": str}) for f in glob.glob(str(TILE / "refit_stars_tile_*.csv.gz"))],
                        ignore_index=True) if glob.glob(str(TILE / "refit_stars_tile_*.csv.gz")) else pd.DataFrame()
    cf_tile = pd.concat([pd.read_csv(f) for f in glob.glob(str(TILE / "calspec_frames_tile_*.csv"))],
                        ignore_index=True) if glob.glob(str(TILE / "calspec_frames_tile_*.csv")) else pd.DataFrame()
    st_tile = st_tile[st_tile["filter"].isin(GRID)].copy()
    # exclude cloud-affected tile frames: high per-frame ZP scatter (matching the standard-field ZPERR<0.08 cut)
    # OR a strong within-frame colour gradient in the ZP (thin-cirrus colour-dependent extinction), e.g. 7DT02
    # m450/m475 on T01485/T19136 flagged as cloud-suspect in 260908_zpcolor_report.md section 6.
    def frame_grad(g):
        c = g["bprp"].to_numpy(); z = g["zp"].to_numpy(); ok = np.isfinite(c) & np.isfinite(z)
        return np.polyfit(c[ok], z[ok], 1)[0] if ok.sum() > 20 and np.ptp(c[ok]) > 0.5 else 0.0
    zpstd = st_tile.groupby("file")["zp"].std()
    grad = st_tile.groupby("file", group_keys=False).apply(frame_grad)
    bad_tile = set(zpstd[zpstd > 0.08].index) | set(grad[grad.abs() > 0.05].index)
    st_tile = st_tile[~st_tile["file"].isin(bad_tile)].copy()
    if len(cf_tile):
        cf_tile = cf_tile[~cf_tile["file"].isin(bad_tile)].copy()
    print(f"tile: {len(st_tile)} ref-star rows ({st_tile.file.nunique()} frames), {len(cf_tile)} calspec rows; "
          f"excluded {len(bad_tile)} cloud/bad frames (zp_std>0.08)")

    # ---- XP cor flux for tile ref stars ----
    centers = []
    meta = pd.read_csv("/lyman/data1/Calibration/latest_CALSPEC/merged_meta.csv").set_index("Name")
    syn = pd.read_csv(ZPC / "calspec_synphot.csv")
    for o in STD_TILE:
        sp = syn[syn.object == o].spec.iloc[0]; nm = re.split(r"_(mod|stis|stisnic|stiswfc|nic)", sp)[0]
        centers.append((float(meta.loc[nm, "radeg"]), float(meta.loc[nm, "decdeg"])))
    xp_tile = extract_xp_cor(st_tile.ref_id.unique(), centers)
    st_tile = st_tile[st_tile.ref_id.isin(xp_tile)].copy()
    print(f"tile XP cor extracted: {len(xp_tile)} stars; ref-star rows with XP: {len(st_tile)}")

    # ---- standard-field data ----
    zf = np.load(TS / "refstar_flux_origcor.npz")
    ids_std = zf["ids"]; flux_std = zf["flux_cor"]; idx_std = {s: i for i, s in enumerate(ids_std)}
    frames = pd.read_csv(ZPC / "frames.csv"); bad = set(frames[frames["ZPERR_APER_3"] > 0.08]["file"])
    st_std = pd.read_csv(ZPC / "refit_stars.csv.gz", dtype={"ref_id": str})
    st_std = st_std[(st_std.catalog == "GAIAXP_CORR_LAMOST") & ~st_std.file.isin(bad)
                    & st_std["filter"].isin(GRID) & st_std.ref_id.isin(idx_std)].copy()

    # unified flux table (std first, then tile)
    tile_ids = list(xp_tile)
    all_ids = list(ids_std) + tile_ids
    flux_all = np.vstack([flux_std, np.array([xp_tile[s] for s in tile_ids])]) if tile_ids else flux_std
    idx_all = {s: i for i, s in enumerate(all_ids)}
    for st in (st_std, st_tile):
        st["row"] = st.ref_id.map(idx_all)
        st["sig"] = np.sqrt(st.magerr.to_numpy() ** 2 + 0.01 ** 2)
    st_std["set"] = "std"; st_tile["set"] = "tile"
    st_all = pd.concat([st_std, st_tile[st_std.columns.intersection(st_tile.columns)]], ignore_index=True).dropna(subset=["row"])
    st_all["row"] = st_all.row.astype(int)

    # ---- CALSPEC frames (std + tile) ----
    cf_std = R.calspec_clean(ZPC, bad); cf_std["set"] = "std"
    keep = ["file", "object", "filter", "telescop", "mjd", "spectral_type", "mag_inst_APER_3", "flags", "flux_max",
            "class_star", "sep_arcsec", "magerr_APER_3", "status"]
    if len(cf_tile):
        cft = cf_tile[cf_tile["filter"].isin(GRID)].copy()
        cft = cft[(cft["status"] == "ok") & (cft["flags"] <= 1) & (cft["flux_max"] < 45000)
                  & (cft["sep_arcsec"] < 1.5) & (cft["magerr_APER_3"] < 0.05)]
        cft["set"] = "tile"
        cf_all = pd.concat([cf_std, cft], ignore_index=True)
    else:
        cf_all = cf_std
    is_da = {o: str(t).startswith("DA") for o, t in cf_all.groupby("object")["spectral_type"].first().items()}
    cal_spec = {}
    for o, sp in zip(syn.drop_duplicates("object").object, syn.drop_duplicates("object").spec):
        t = Table.read(CALSPEC_DIR / sp); wl = np.asarray(t["WAVELENGTH"], float) / 10.0
        fl = np.asarray(t["FLUX"], float) * 1e-2; m = (wl > 300) & (wl < 1000) & np.isfinite(fl)
        cal_spec[o] = (wl[m], fl[m])

    # ---- closed loop: nominal vs physical, per dataset split ----
    rows = []
    for f in GRID:
        curves = {"nominal": R.load_curve(f), "physical": load_new_curve(f)}
        for dset, stf, cff in [("std", st_all[st_all.set == "std"], cf_all[cf_all.set == "std"]),
                               ("tile", st_all[st_all.set == "tile"], cf_all[cf_all.set == "tile"]),
                               ("combined", st_all, cf_all)]:
            rf = stf[stf["filter"] == f]; cfff = cff[cff["filter"] == f]
            if len(rf) < 20:
                continue
            row = rf.row.to_numpy(); m_arch = rf.ref_mag.to_numpy()
            for cn, cv in curves.items():
                m_ref = R.syn_mag(XP_WAVE, flux_all[row], cv[0], cv[1], 0.0)
                zp_i = rf.zp.to_numpy() + (m_ref - m_arch)
                med, sd, n, keep = R.clipped_zp(rf.file.to_numpy(), zp_i)
                resid = zp_i - rf.file.map(med).to_numpy()
                a, a_err, b, nfit = R.weighted_linfit(rf.bprp.to_numpy()[keep], resid[keep], rf.sig.to_numpy()[keep])
                out = dict(filter=f, pivot=float(f[1:4]), dataset=dset, curve=cn, n_stars=len(rf),
                           n_frames=int(rf.file.nunique()), a_bprp=round(a, 4),
                           zp_std=round(float(np.nanmedian(sd.values)), 4))
                if len(cfff):
                    mt = {o: float(R.syn_mag(cal_spec[o][0], cal_spec[o][1], cv[0], cv[1], 0.0)) for o in cal_spec}
                    mobs = cfff["mag_inst_APER_3"].to_numpy() + cfff["file"].map(med).to_numpy()
                    delta = mobs - cfff["object"].map(mt).to_numpy()
                    obj = cfff["object"].to_numpy()
                    fin = np.isfinite(delta)
                    dvals = {}
                    for o in np.unique(obj[fin]):
                        dd = delta[fin][obj[fin] == o]
                        if len(dd) >= 5:
                            dvals[o] = float(np.median(dd))
                    if dvals:
                        da_v = [v for o, v in dvals.items() if is_da.get(o, False)]
                        ot_v = [v for o, v in dvals.items() if not is_da.get(o, False)]
                        out["n_cal_stars"] = len(dvals)
                        out["median_all"] = round(float(np.median(list(dvals.values()))), 4)
                        out["da_other"] = round(float(np.median(da_v) - np.median(ot_v)), 4) if da_v and ot_v else np.nan
                rows.append(out)
    res = pd.DataFrame(rows)
    res.to_csv(TILE / "tilestd_closedloop.csv", index=False)
    print("\n=== colour term a(BP-RP), physical curve, by dataset ===")
    print(res[res.curve == "physical"].pivot_table(index="filter", columns="dataset", values="a_bprp").round(4).to_string())
    print("\n=== CALSPEC median (physical) & n_cal_stars by dataset ===")
    print(res[res.curve == "physical"].pivot_table(index="filter", columns="dataset", values=["median_all", "n_cal_stars"]).round(4).to_string())

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    col = {"std": "#1f77b4", "tile": "#d62728", "combined": "#2ca02c"}
    for ax, (cn, metric, ylab) in zip(axes, [("physical", "a_bprp", "colour term a(BP-RP) [physical]"),
                                             ("physical", "median_all", "CALSPEC median [physical]"),
                                             ("nominal", "median_all", "CALSPEC median [nominal]")]):
        for dset in ["std", "tile", "combined"]:
            s = res[(res.curve == cn) & (res.dataset == dset)].dropna(subset=[metric]).sort_values("pivot")
            ax.plot(s["pivot"], s[metric], "o-", ms=4, color=col[dset], label=dset)
        ax.axhline(0, color="k", lw=0.5); ax.set(xlabel="pivot [nm]", ylabel=ylab, title=ylab)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("F17  standard-field vs survey-tile vs combined (corrected XP)")
    fig.tight_layout(); fig.savefig(TS / "figures" / "F17_tilestd.png", dpi=150)
    print("\nsaved -> tilestd_closedloop.csv, figures/F17_tilestd.png")


if __name__ == "__main__":
    main()
