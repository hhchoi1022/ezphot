#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_allbricqs_photometry.py

AllBRICQS bright quasars (Onken+2023 south, Choi+2025 north) as transmission-curve test targets.
Reuses the M-star machinery (260910_xp_mdwarfs_photometry.py): the quasars are matched in the .cat detection lists of all
medium-band single frames, observed magnitudes (MAG_APER_3 + header ZP) are compared with XP-synthetic magnitudes through
the nominal and the effective curves (frame ZP re-derived from the frame's reference stars for the effective curves).
Quasar variability between the XP epoch (2014-17) and the 7DT epoch is absorbed by a per-quasar linear (in wavelength)
correction fitted to the bands without strong emission lines; the residuals of the line bands are the test.

Inputs: scratchpad allbricqs_all.csv (218 quasars), allbricqs_xp_spectra.npz (143 XP spectra from the doabdi tiles),
        allbricqs_edge_sensitivity.csv (template-based dm/dnm per band).
Outputs (log dir /home/hhchoi1022/ezphot/log/20260910_allbricqs/): qso_phot.csv.gz, qso_resid.csv, figures/.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
C = importlib.import_module("260910_effcurve_common")
P = importlib.import_module("260910_xp_mdwarfs_photometry")
R = C.R
warnings.filterwarnings("ignore")
log = C.log

OUT = Path("/home/hhchoi1022/ezphot/log/20260910_allbricqs")
SC = Path("/tmp/claude-10000/-lyman-data1-factory-hhchoi-code-ezphot/e9ba5c7a-dab7-4633-a4a2-61a197d96fce/scratchpad")
XP_NPZ = Path("/home/hhchoi1022/ezphot/log/20260910_xp_mdwarfs/allbricqs_xp_spectra.npz")
LINES = [("Lyα", 121.6), ("Si IV", 139.7), ("C IV", 154.9), ("C III]", 190.9), ("Mg II", 279.8), ("Hγ", 434.0), ("Hβ", 486.1), ("[O III]", 500.7), ("Hα", 656.3)]


def frame_zp_change(kind, filters, frames_dir=None):
    """ZP' - ZP per frame/coadd from its own reference stars re-synthesised with the effective curves (single: refit_stars.csv.gz; coadd: coadd/refit_stars.csv.gz;
    frames_dir: a 260910_allbricqs_collect.py directory with its own refit_stars.csv.gz and tilestar_spectra.npz)."""
    path = (frames_dir / "refit_stars.csv.gz") if frames_dir else C.PREV_OUT / ("refit_stars.csv.gz" if kind == "single" else "coadd/refit_stars.csv.gz")
    st = pd.read_csv(path, dtype={"ref_id": str}, usecols=["file", "catalog", "filter", "ref_id"]); st = st[(st.catalog == R.CATALOG) & st["filter"].isin(filters)]
    spectra, ids = [], []
    for q in ([frames_dir / "tilestar_spectra.npz"] if frames_dir else []) + [C.PREV_OUT / "shift_recal" / "refstar_spectra.npz", C.DEFAULT_OUT / "tilestar_spectra.npz"]:
        if q.exists(): zz = np.load(q, allow_pickle=False); spectra.append(zz["flux"]); ids.append(zz["ids"].astype(str)); wave = zz["wave_nm"]
    flux = np.vstack(spectra); ids = np.concatenate(ids); _, first = np.unique(ids, return_index=True); flux, ids = flux[first], ids[first]; idx = {x: i for i, x in enumerate(ids)}
    st = st[st.ref_id.isin(idx)].copy(); st["row"] = st.ref_id.map(idx); out = []
    for f in filters:
        nom = C.load_nominal(f); eff = P.load_effective(f); g = st[st["filter"] == f]
        if eff is None or not len(g): continue
        u_rows, inv = np.unique(g.row.to_numpy(), return_inverse=True)
        dm = (R.syn_mag(wave, flux[u_rows], C.FINE, eff, 0.0) - R.syn_mag(wave, flux[u_rows], C.FINE, nom, 0.0))[inv]
        t = pd.DataFrame({"file": g.file.to_numpy(), "dm": dm}).groupby("file")["dm"].agg(["median", "size"]); t = t[t["size"] >= 5]
        out.append(pd.DataFrame({"file": t.index, "filter": f, "dzp": t["median"].to_numpy(), "n_ref": t["size"].to_numpy()}))
    return pd.concat(out) if out else pd.DataFrame(columns=["file", "filter", "dzp", "n_ref"])


def stage_match(n_proc, kind, frames_dir=None):
    Q = pd.read_csv(SC / "allbricqs_all.csv", dtype={"source_id": str})
    P._G["stars"] = dict(ids=Q.name.to_numpy(str), ra=Q.ra.to_numpy(float), dec=Q.dec.to_numpy(float), pmra=np.zeros(len(Q)), pmdec=np.zeros(len(Q)))
    F = C.load_frames("single", prev_out=frames_dir) if frames_dir else C.load_frames(kind)
    F = F[F["FILTER"].isin(C.MEDIUM) & F["ZP_APER_3"].notna() & ~F["mislabeled"] & ~F["bad_zperr"]]
    jobs = [(r.file, r.FILTER, r.TELESCOP, float(r.MJD), float(r.AIRMASS), float(r.CRVAL1), float(r.CRVAL2), float(r.ZP_APER_3), float(r.ZPERR_APER_3)) for r in F.itertuples()]
    rows = []
    with Pool(n_proc) as pool:
        for res in pool.imap_unordered(P.match_frame, jobs, chunksize=8): rows += res
    D = pd.DataFrame(rows); D.to_csv(OUT / f"qso_phot_{kind}.csv.gz", index=False, compression="gzip")
    log(f"match: {len(D)} detections of {D.sid.nunique() if len(D) else 0} quasars in {D.file.nunique() if len(D) else 0} frames")
    if len(D): print(D.groupby(["sid", "filter"]).size().unstack(fill_value=0).to_string())


def stage_compute(fig_dir, kind, frames_dir=None):
    D = pd.read_csv(OUT / f"qso_phot_{kind}.csv.gz", dtype={"sid": str})
    D = D[(D["flags"] <= 3) & (D["flux_max"] < 45000) & (D["magerr"] < 0.2)]
    z = np.load(XP_NPZ, allow_pickle=False); names = z["names"].astype(str); idx = {n: i for i, n in enumerate(names)}; wave = z["wave_nm"]
    Q = pd.read_csv(SC / "allbricqs_all.csv", dtype={"source_id": str}).set_index("name")
    E = pd.read_csv(SC / "allbricqs_edge_sensitivity.csv")
    filters = [f for f in C.MEDIUM if f in set(D["filter"])]
    dzp = frame_zp_change(kind, filters, frames_dir); D = D.merge(dzp, on=["file", "filter"], how="left")
    rows = []
    for q, g in D.groupby("sid"):
        if q not in idx: log(f"{q}: no XP spectrum"); continue
        fl = z["cor"][idx[q]]; fl_raw = z["raw"][idx[q]]; zq = float(Q.loc[q, "z"])
        for f, gf in g.groupby("filter"):
            nom = C.load_nominal(f); eff = P.load_effective(f)
            m_nom = float(R.syn_mag(wave, fl, C.FINE, nom, 0.0)); m_eff = float(R.syn_mag(wave, fl, C.FINE, eff, 0.0)) if eff is not None else m_nom
            m_raw = float(R.syn_mag(wave, fl_raw, C.FINE, nom, 0.0))
            obs = (gf.mag_inst + gf.zp).to_numpy(); obs_eff = (gf.mag_inst + gf.zp + gf.dzp).to_numpy()
            e = E[(E.name == q) & (E["filter"] == f)]
            rows.append(dict(name=q, z=zq, filter=f, pivot=C.pivot_nm(C.FINE, nom), n=len(gf), obs=float(np.nanmedian(obs)), obs_err=R.mad(obs) / np.sqrt(len(gf)) + 0.003, m_xp_nom=m_nom, m_xp_eff=m_eff, m_xp_raw_nom=m_raw,
                             r_nom=float(np.nanmedian(obs)) - m_nom, r_eff=float(np.nanmedian(obs_eff)) - m_eff, dm_per_nm=float(e.dm_per_nm.iloc[0]) if len(e) else np.nan, lines=str(e.lines.iloc[0]) if len(e) else ""))
    S = pd.DataFrame(rows)
    if not len(S): log("no quasar with both 7DT detections and an XP spectrum"); return
    # per-quasar linear correction (variability: grey + colour) fitted on bands with |dm/dnm| < 0.005 (no line at an edge)
    out = []
    for q, g in S.groupby("name"):
        g = g.sort_values("pivot").copy(); cont = g[g.dm_per_nm.abs() < 0.005]
        for col in ("r_nom", "r_eff"):
            if len(cont) >= 4:
                a, ae, b, n = R.weighted_linfit(cont["pivot"].to_numpy() - 600.0, cont[col].to_numpy(), np.sqrt(cont.obs_err.to_numpy() ** 2 + 0.01 ** 2))
                g[col + "_corr"] = g[col] - (b + a * (g["pivot"] - 600.0))
            else: g[col + "_corr"] = np.nan
        out.append(g)
    S = pd.concat(out); S.to_csv(OUT / f"qso_resid_{kind}.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[["name", "z", "filter", "n", "obs", "obs_err", "m_xp_nom", "m_xp_raw_nom", "r_nom", "r_eff", "r_nom_corr", "r_eff_corr", "dm_per_nm", "lines"]].round(4).to_string(index=False))
    # figure per quasar: XP spectrum with bands and lines; residuals nominal vs effective
    for q, g in S.groupby("name"):
        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        ax = axes[0]; fl = z["cor"][idx[q]]; fr = z["raw"][idx[q]]; zq = g.z.iloc[0]
        ax.plot(wave, fr * 1e15, color="0.6", lw=.8, label="XP raw"); ax.plot(wave, fl * 1e15, color="k", lw=1.2, label="XP LAMOST-corrected (flux_cor)")
        for j, f in enumerate(C.MEDIUM):
            cv = C.load_nominal(f); m = cv > 0.1 * cv.max()
            if m.any(): ax.axvspan(C.FINE[m].min(), C.FINE[m].max(), color="grey" if j % 2 else "lightgrey", alpha=.18)
        for name, l0 in LINES:
            lc = l0 * (1 + zq)
            if 380 < lc < 1000: ax.axvline(lc, color="tab:red", lw=.8); ax.text(lc, ax.get_ylim()[1] * 0.95 if False else np.nanmax(fl * 1e15) * 1.02, name, color="tab:red", fontsize=8, ha="center")
        ax.set_ylabel("F_λ [10⁻¹⁵ W m⁻² nm⁻¹]"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper right"); ax.set_title(f"{q}: z = {zq:.3f}, BP = {Q.loc[q, 'BP']:.2f}, RP = {Q.loc[q, 'RP']:.2f}; Gaia XP spectrum with the 7DT medium bands and the redshifted emission lines", fontsize=10)
        ax = axes[1]
        ax.errorbar(g["pivot"] - 2, g.r_nom_corr, yerr=g.obs_err, fmt="o", mfc="none", color="0.4", ms=6, capsize=2, label="nominal curves (after linear variability fit on continuum bands)")
        ax.errorbar(g["pivot"] + 2, g.r_eff_corr, yerr=g.obs_err, fmt="s", color="tab:orange", ms=6, capsize=2, label="effective curves (same)")
        ax2 = ax.twinx(); ax2.bar(g["pivot"], g.dm_per_nm, width=8, color="tab:blue", alpha=.25, label="template edge sensitivity dm/dnm (right axis)"); ax2.set_ylabel("dm per nm of curve shift [mag/nm]"); ax2.set_ylim(-0.15, 0.15)
        ax.axhline(0, color="k", lw=.6); ax.axhspan(-0.01, 0.01, color="grey", alpha=.15); ax.set_ylim(-0.15, 0.15); ax.grid(alpha=.3); ax.set_xlabel("wavelength [nm]"); ax.set_ylabel("observed − XP synthetic [mag]")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        ax.set_title("7DT observed magnitude minus XP-synthetic magnitude per band (median over frames)", fontsize=10)
        fig.tight_layout(); fig.savefig(fig_dir / f"Q_{q}_{kind}.png", dpi=130); plt.close(fig)
    log("figures written")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["match", "compute", "all"]); ap.add_argument("--n-proc", type=int, default=32); ap.add_argument("--kind", default="single", choices=["single", "coadd", "new"])
    ap.add_argument("--frames-dir", default=None, help="260910_allbricqs_collect.py output directory (kind 'new': the newly processed tiles)")
    a = ap.parse_args(); fig_dir = OUT / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fd = Path(a.frames_dir) if a.frames_dir else (Path("/home/hhchoi1022/ezphot/log/20260910_allbricqs/collect") if a.kind == "new" else None)
    if a.stage in ("match", "all"): stage_match(a.n_proc, a.kind, fd)
    if a.stage in ("compute", "all"): stage_compute(fig_dir, a.kind, fd)


if __name__ == "__main__":
    main()
