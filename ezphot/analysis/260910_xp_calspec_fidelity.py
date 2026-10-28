#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_xp_calspec_fidelity.py

How accurate are Gaia XP synthetic magnitudes in the 7DT medium bands, as a function of stellar colour?
Every CALSPEC star (latest_CALSPEC/merged_meta.csv) is looked up in the doabdi XP tiles (raw XP flux and the
LAMOST-corrected flux_cor used by the reference catalogue); synthetic magnitudes through the NOMINAL curves are compared
with the CALSPEC spectrum (STIS/NICMOS observed file preferred, model otherwise).  d = m_XP - m_CALSPEC per band.
The M3.5V dwarf BD-11 3759 (BP-RP 2.9) is the only M-type calibrator; it is also a 7DT standard, so its 7DT residual
(validation_calspec_stars.csv) can be compared with the XP-based residual of the M-star test.

  --stage extract : find the CALSPEC stars in the tiles -> calspec_xp_spectra.npz
  --stage compute : synthetic magnitudes, table calspec_xp_fidelity.csv, figure figures/F8_xp_fidelity_vs_colour.png
"""
from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
from astropy.io import ascii        # noqa: E402
from astropy.table import Table     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
C = importlib.import_module("260910_effcurve_common")
R = C.R
warnings.filterwarnings("ignore")
log = C.log

OUT = Path("/home/hhchoi1022/ezphot/log/20260910_xp_mdwarfs")
CALSPEC = Path("/lyman/data1/Calibration/latest_CALSPEC")
TILES_TXT = Path(__file__).resolve().parents[1] / "utils" / "tileinfo" / "7-DT" / "final_tiles.txt"
MATCH_ARCSEC = 3.0


def find_in_tile(args):
    name, tile, ra1, dec1 = args
    p = R.SPECTRA_DIR / f"{tile}.csv"
    if not p.exists(): return name, None
    with open(p, "r", errors="ignore") as fh:
        header = fh.readline().lstrip("﻿").rstrip("\n").split(","); col = {c: k for k, c in enumerate(header)}
        pre = "gaiaxp_photinfo_replenish_"
        try: i_ra, i_dec = col[pre + "ra"], col[pre + "dec"]
        except KeyError: return name, None
        for line in fh:
            parts = line.rstrip("\n").split(",")
            try: ra, dec = float(parts[i_ra]), float(parts[i_dec])
            except (ValueError, IndexError): continue
            if abs(dec - dec1) > 0.01: continue
            sep = np.hypot((ra - ra1) * np.cos(np.radians(dec1)), dec - dec1) * 3600
            if sep <= MATCH_ARCSEC:
                try:
                    raw = np.asarray(parts[col["correctedxpspecv1_flux"]].split(";"), float); cor = np.asarray(parts[col["correctedxpspecv1_flux_cor"]].split(";"), float)
                    err = np.asarray(parts[col["correctedxpspecv1_flux_error"]].split(";"), float)
                except (ValueError, KeyError): continue
                if len(raw) != len(R.XP_WAVE_NM): continue
                def fl(c):
                    try: return float(parts[col[pre + c]])
                    except (ValueError, KeyError, IndexError): return np.nan
                return name, dict(source_id=parts[0], tile=tile, sep=sep, bp_rp=fl("bp_mag") - fl("rp_mag"), g=fl("g_mag"), raw=raw, cor=cor, err=err,
                                  caution=parts[col["correctedxpspecv1_caution"]] if "correctedxpspecv1_caution" in col else "")
    return name, None


def load_meta():
    m = pd.read_csv(CALSPEC / "merged_meta.csv")
    m["name"] = m["Name"].astype(str).str.split().str[0]
    m = m[m["radeg"].notna()].copy()
    dt = 16.0
    pmra = m["PM_R.A."].fillna(0.0).to_numpy(float); pmdec = m["PM_Decl."].fillna(0.0).to_numpy(float)
    m["ra2016"] = m["radeg"] + pmra / 3.6e6 * dt / np.cos(np.radians(m["decdeg"])); m["dec2016"] = m["decdeg"] + pmdec / 3.6e6 * dt
    return m


def stage_extract(n_proc):
    m = load_meta(); T = ascii.read(TILES_TXT)
    tra, tdec = np.asarray(T["ra"], float), np.asarray(T["dec"], float)
    jobs = []
    for r in m.itertuples():
        d = np.degrees(np.arccos(np.clip(np.sin(np.radians(r.dec2016)) * np.sin(np.radians(tdec)) + np.cos(np.radians(r.dec2016)) * np.cos(np.radians(tdec)) * np.cos(np.radians(tra - r.ra2016)), -1, 1)))
        for i in np.argsort(d)[:2]: jobs.append((r.name, str(T["id"][i]), float(r.ra2016), float(r.dec2016)))
    log(f"extract: {len(m)} CALSPEC stars, {len(jobs)} tile scans")
    found = {}
    with Pool(n_proc) as pool:
        for name, res in pool.imap_unordered(find_in_tile, jobs):
            if res is not None and (name not in found or res["sep"] < found[name]["sep"]): found[name] = res; log(f"  {name}: {res['tile']} {res['source_id']} sep {res['sep']:.1f}\" BP-RP {res['bp_rp']:.2f} G {res['g']:.2f}")
    names = sorted(found)
    np.savez_compressed(OUT / "calspec_xp_spectra.npz", names=np.array(names), source_id=np.array([found[n]["source_id"] for n in names]), tile=np.array([found[n]["tile"] for n in names]),
                        bp_rp=np.array([found[n]["bp_rp"] for n in names]), g=np.array([found[n]["g"] for n in names]), raw=np.vstack([found[n]["raw"] for n in names]),
                        cor=np.vstack([found[n]["cor"] for n in names]), err=np.vstack([found[n]["err"] for n in names]), caution=np.array([found[n]["caution"] for n in names]), wave_nm=R.XP_WAVE_NM)
    log(f"extract: {len(names)} of {len(m)} CALSPEC stars found in the tiles")


def calspec_spectrum(name):
    files = sorted((CALSPEC / "spectra").glob(f"{name}_*.fits"))
    obs = [f for f in files if "_mod_" not in f.name]; mod = [f for f in files if "_mod_" in f.name]
    for f in obs[::-1] + mod[::-1]:
        try: t = Table.read(f)
        except Exception: continue
        wl = np.asarray(t["WAVELENGTH"], float) / 10.0; fl = np.asarray(t["FLUX"], float) * 1e-2
        ok = np.isfinite(fl) & (fl > 0)
        if wl[ok].min() <= 385 and wl[ok].max() >= 1000:
            mm = ok & (wl > 300) & (wl < 1050); return wl[mm], fl[mm], f.name
    return None


def stage_compute():
    z = np.load(OUT / "calspec_xp_spectra.npz", allow_pickle=False); wave = z["wave_nm"]
    meta = load_meta().set_index("name")
    rows = []
    for k, name in enumerate(z["names"]):
        cs = calspec_spectrum(name)
        if cs is None: log(f"  {name}: no CALSPEC spectrum covering 385-1000 nm"); continue
        wl, fl, fname = cs
        for f in C.MEDIUM:
            curve = C.load_nominal(f)
            m_cal = float(R.syn_mag(wl, fl, C.FINE, curve, 0.0)); m_raw = float(R.syn_mag(wave, z["raw"][k], C.FINE, curve, 0.0)); m_cor = float(R.syn_mag(wave, z["cor"][k], C.FINE, curve, 0.0))
            rows.append(dict(name=name, star=meta.loc[name, "Star name"] if name in meta.index else name, sptype=meta.loc[name, "Sp. T."] if name in meta.index else "", spec_file=fname,
                             bp_rp=float(z["bp_rp"][k]), g=float(z["g"][k]), caution=str(z["caution"][k]), filter=f, pivot=C.pivot_nm(C.FINE, curve), m_cal=m_cal, m_xp_raw=m_raw, m_xp_cor=m_cor,
                             d_raw=m_raw - m_cal, d_cor=m_cor - m_cal))
    D = pd.DataFrame(rows); D.to_csv(OUT / "calspec_xp_fidelity.csv", index=False)
    # grey term per star (median over 450-875 nm grid bands) and colour-dependent part
    grid = D["filter"].str.match(r"^m\d{3}$") & (D["pivot"] >= 440)
    grey = D[grid].groupby("name")["d_cor"].median(); D["d_cor_rel"] = D["d_cor"] - D["name"].map(grey)
    grey_raw = D[grid].groupby("name")["d_raw"].median(); D["d_raw_rel"] = D["d_raw"] - D["name"].map(grey_raw)
    stars = D.groupby("name").agg(star=("star", "first"), sptype=("sptype", "first"), bp_rp=("bp_rp", "first"), g=("g", "first"), grey_cor=("d_cor", lambda s: np.nan), n=("filter", "size"))
    stars["grey_cor"] = grey; stars["grey_raw"] = grey_raw
    stars["rms_rel_cor"] = D[grid].groupby("name")["d_cor_rel"].apply(lambda s: float(np.sqrt(np.mean(s ** 2)))); stars["rms_rel_raw"] = D[grid].groupby("name")["d_raw_rel"].apply(lambda s: float(np.sqrt(np.mean(s ** 2))))
    stars = stars.sort_values("bp_rp"); stars.to_csv(OUT / "calspec_xp_fidelity_stars.csv")
    pd.set_option("display.width", 250); print(stars.round(3).to_string())
    # per band vs colour group
    groups = [("hot (BP−RP < 0.3)", -9, 0.3), ("0.3–0.8", 0.3, 0.8), ("0.8–1.3", 0.8, 1.3), ("1.3–1.8", 1.3, 1.8), ("M (≥ 1.8)", 1.8, 9)]
    summ = []
    for lab, lo, hi in groups:
        sub = D[(D.bp_rp >= lo) & (D.bp_rp < hi)]
        for f, g in sub.groupby("filter"):
            summ.append(dict(group=lab, lo=lo, hi=hi, filter=f, pivot=g["pivot"].iloc[0], n=g.name.nunique(), med_cor=float(g.d_cor_rel.median()), mad_cor=R.mad(g.d_cor_rel), med_raw=float(g.d_raw_rel.median()), mad_raw=R.mad(g.d_raw_rel)))
    SU = pd.DataFrame(summ); SU.to_csv(OUT / "calspec_xp_fidelity_groups.csv", index=False)
    # ---- figure F8
    fig, axes = plt.subplots(3, 1, figsize=(13, 13))
    ax = axes[0]
    for lab, lo, hi in groups:
        s = SU[SU.group == lab]
        if not len(s): continue
        ax.errorbar(s["pivot"], s.med_cor, yerr=s.mad_cor, fmt="o-", ms=4, capsize=2, lw=1, label=f"{lab}: n = {int(s.n.max())}")
    ax.axhline(0, color="k", lw=.6); ax.axhspan(-0.01, 0.01, color="grey", alpha=.12); ax.set_ylim(-0.08, 0.08); ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=3)
    ax.set_ylabel("m_XP(cor) − m_CALSPEC, grey term removed [mag]"); ax.set_title("XP (LAMOST-corrected) synthetic magnitude minus CALSPEC, per band and colour group (median ± MAD over stars; grey term per star removed)", fontsize=10)
    ax = axes[1]
    for name in stars.index[stars.bp_rp >= 1.3]:
        g = D[D.name == name].sort_values("pivot"); ax.plot(g["pivot"], g.d_cor, "o-", ms=4, lw=1, label=f"{g.star.iloc[0]} ({g.sptype.iloc[0]}, BP−RP {g.bp_rp.iloc[0]:.2f}) cor")
        if g.bp_rp.iloc[0] >= 1.8: ax.plot(g["pivot"], g.d_raw, "s--", ms=4, lw=.8, color="0.4", label=f"{g.star.iloc[0]} raw XP")
    ax.axhline(0, color="k", lw=.6); ax.axhspan(-0.01, 0.01, color="grey", alpha=.12); ax.set_ylim(-0.12, 0.12); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_ylabel("m_XP − m_CALSPEC [mag]"); ax.set_title("the red CALSPEC stars individually (BP−RP ≥ 1.3): absolute band-by-band error of the XP synthetic magnitude", fontsize=10)
    ax = axes[2]
    for f in ["m450", "m500", "m550", "m600", "m625", "m700", "m800"]:
        g = D[D["filter"] == f].sort_values("bp_rp"); ax.plot(g.bp_rp, g.d_cor_rel, "o", ms=4, label=f, alpha=.8)
    ax.axhline(0, color="k", lw=.6); ax.axvspan(0.6, 2.0, color="tab:blue", alpha=.06); ax.axvspan(1.8, 4.0, color="tab:red", alpha=.05); ax.set_ylim(-0.08, 0.08); ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=4)
    ax.set_xlabel("BP−RP"); ax.set_ylabel("m_XP(cor) − m_CALSPEC, grey removed [mag]"); ax.set_title("colour dependence of the XP error in selected bands (blue band = reference-star colours, red band = M-star test)", fontsize=10)
    fig.tight_layout(); (OUT / "figures").mkdir(exist_ok=True); fig.savefig(OUT / "figures" / "F8_xp_fidelity_vs_colour.png", dpi=130); plt.close(fig)
    log("F8 written")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["extract", "compute", "all"]); ap.add_argument("--n-proc", type=int, default=24)
    a = ap.parse_args()
    if a.stage in ("extract", "all"): stage_extract(a.n_proc)
    if a.stage in ("compute", "all"): stage_compute()


if __name__ == "__main__":
    main()
