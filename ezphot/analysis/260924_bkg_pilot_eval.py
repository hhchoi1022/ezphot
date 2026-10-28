#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""260924_bkg_pilot_eval.py — evaluate the background pilot (see 260924_bkg_pilot.py). Output: pilot_eval.csv,
pilot_bins.csv, A8_bkg_pilot.png."""
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
TR = importlib.import_module("260924_stepA_triage")
TC, OT, AN, R = TR.TC, TR.OT, TR.AN, TR.R
LOG = Path("/home/hhchoi1022/ezphot/log/20260924_stepA")
PILOT = LOG / "pilot"
ORIG = {"SPSS": Path("/home/hhchoi1022/ezphot/log/20260918_spss/catalogs"),
        "CALSPEC": Path("/home/hhchoi1022/ezphot/log/20260914_calspec_ezphot/catalogs")}
VARS = ["V0", "V1", "V2"]
SNRB = np.array([20, 25, 30, 40, 60, 80, 110, 150])
AN._init()
curves = TC.build_totals()[("manufacturer", "nobeam")]


def load(cp):
    with fits.open(cp) as h:
        d = h["LDAC_OBJECTS"].data
        names = set(d.columns.names)
        rk = "X_WORLD" if "X_WORLD" in names else "ALPHA_J2000"
        dk = "Y_WORLD" if "Y_WORLD" in names else "DELTA_J2000"
        m = lambda k: np.asarray(d[k], float)
        df = pd.DataFrame({"ALPHA_J2000": m(rk), "DELTA_J2000": m(dk), "MAG_APER_3": m("MAG_APER_3"),
                           "MAGERR_APER_3": m("MAGERR_APER_3"), "FLAGS": m("FLAGS"), "CLASS_STAR": m("CLASS_STAR"),
                           "FWHM_WORLD": m("FWHM_WORLD"), "ELONGATION": m("ELONGATION"),
                           "XI": m("X_IMAGE"), "YI": m("Y_IMAGE"), "M5": m("MAG_APER_1"), "M10": m("MAG_APER_3")})
        df["SNR_APER_3"] = m("SNR_APER_3") if "SNR_APER_3" in names else 1.0857 / np.clip(df.MAGERR_APER_3, 1e-4, None)
    df["F5"] = np.where(df.M5 < 90, 10 ** (-0.4 * df.M5), np.nan)
    df["F10"] = np.where(df.M10 < 90, 10 ** (-0.4 * df.M10), np.nan)
    return df


fr = pd.read_csv(PILOT / "pilot_frames.csv")
rows, bins = [], []
for _, f in fr.iterrows():
    name = Path(f.path).name[:-5] + "_cat.fits"
    cats = {v: PILOT / v / name for v in VARS}
    if not all(p.exists() for p in cats.values()):
        continue
    D = {v: load(p) for v, p in cats.items()}
    orig = ORIG[f.dataset] / name
    key = f"CAL_{f.object}" if f.dataset == "CALSPEC" else f"SPSS_{AN.norm(f.object)}"
    ref = TR.load_cache(key)
    if ref is None:
        continue
    from astropy.coordinates import SkyCoord, match_coordinates_sky
    coords = SkyCoord(ref["ra"], ref["dec"], unit="deg")
    fw, ft = curves[f["filter"]]
    sel0 = OT.select_stars_like(D["V0"])
    row = dict(dataset=f.dataset, object=f.object, filter=f["filter"], exptime=f.exptime, name=name, n_sel=len(sel0))
    # V0 reproduces the existing catalogue?
    if orig.exists():
        o = load(orig)
        from scipy.spatial import cKDTree
        t = cKDTree(np.c_[o.XI, o.YI]); dd, ii = t.query(np.c_[D["V0"].XI, D["V0"].YI])
        okm = (dd < 1.0) & (D["V0"].SNR_APER_3 > 50) & (D["V0"].M10 < 90) & (o.M10.to_numpy()[ii] < 90)
        row["v0_vs_orig_mmag"] = 1e3 * np.median(D["V0"].M10[okm] - o.M10.to_numpy()[ii][okm]) if okm.sum() > 10 else np.nan
    for v in VARS:
        sel = OT.select_stars_like(D[v])
        p, g, n = TR.estimate_pedestal(sel.F5.to_numpy(), sel.F10.to_numpy(), TR.AREA[5.0], TR.AREA[10.0],
                                       sel.SNR_APER_3.to_numpy())
        row[f"p_{v}"] = p
        # XP residual vs SNR
        src = SkyCoord(sel.ALPHA_J2000.to_numpy(), sel.DELTA_J2000.to_numpy(), unit="deg")
        idx, d2d, _ = match_coordinates_sky(src, coords)
        ok = np.asarray(d2d.arcsec < 2.0)
        if ok.sum() < 15:
            continue
        syn = R.syn_mag(TR.XP_WAVE, ref["cor"][idx[ok]].astype(float), fw, ft, 0.0)
        zc = syn - sel.MAG_APER_3.to_numpy()[ok]; snr = sel.SNR_APER_3.to_numpy()[ok]
        zp, zsd, nz = TR.clipped(zc[snr > 40])
        row[f"zp_{v}"] = zp
        res = zc - zp
        for k in range(len(SNRB) - 1):
            m = (snr >= SNRB[k]) & (snr < SNRB[k + 1]) & np.isfinite(res)
            if m.sum() >= 3:
                bins.append(dict(dataset=f.dataset, exptime=f.exptime, variant=v, snr_lo=SNRB[k],
                                 res=float(np.median(res[m])), n=int(m.sum()), name=name))
    # same-star change Vk - V0 vs the analysis-side prediction from p_V0
    from scipy.spatial import cKDTree
    t0 = cKDTree(np.c_[D["V0"].XI, D["V0"].YI])
    for v in ["V1", "V2"]:
        dd, ii = t0.query(np.c_[D[v].XI, D[v].YI])
        m = (dd < 1.0) & (D[v].M10 < 90) & (D["V0"].M10.to_numpy()[ii] < 90) & (D["V0"].SNR_APER_3.to_numpy()[ii] > 20)
        dm = (D[v].M10.to_numpy() - D["V0"].M10.to_numpy()[ii])[m]
        f10 = D["V0"].F10.to_numpy()[ii][m]
        pred = -2.5 * np.log10(np.clip(1 - row["p_V0"] * TR.AREA[10.0] / f10, 1e-6, None))
        snr0 = D["V0"].SNR_APER_3.to_numpy()[ii][m]
        faint = snr0 < 40; bright = snr0 > 100
        row[f"dm_faint_{v}"] = 1e3 * np.median(dm[faint]) if faint.sum() > 5 else np.nan
        row[f"dm_bright_{v}"] = 1e3 * np.median(dm[bright]) if bright.sum() > 5 else np.nan
        row[f"pred_faint"] = 1e3 * np.median(pred[faint]) if faint.sum() > 5 else np.nan
        row[f"slope_dm_vs_pred_{v}"] = float(np.polyfit(pred, dm, 1)[0]) if len(dm) > 20 else np.nan
    rows.append(row)
E = pd.DataFrame(rows); B = pd.DataFrame(bins)
E.to_csv(LOG / "pilot_eval.csv", index=False); B.to_csv(LOG / "pilot_bins.csv", index=False)
E["grp"] = E.dataset + " " + E.exptime.astype(int).astype(str) + "s"
cols = ["v0_vs_orig_mmag", "p_V0", "p_V1", "p_V2", "dm_faint_V1", "dm_faint_V2", "pred_faint", "dm_bright_V1", "dm_bright_V2",
        "slope_dm_vs_pred_V1", "slope_dm_vs_pred_V2"]
pd.set_option("display.width", 250)
print(E.groupby("grp")[cols].median().round(3).T.to_string())
E["dzp_V1"] = 1e3 * (E.zp_V1 - E.zp_V0); E["dzp_V2"] = 1e3 * (E.zp_V2 - E.zp_V0)
print("\nZP change vs V0 [mmag]:"); print(E.groupby("grp")[["dzp_V1", "dzp_V2"]].median().round(1).to_string())
# figure
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
B["grp"] = B.dataset + " " + B.exptime.astype(int).astype(str) + "s"
sty = {"V0": ("-", "o"), "V1": ("--", "s"), "V2": (":", "^")}
colr = {"CALSPEC 20s": "#2171b5", "CALSPEC 30s": "#6baed6", "SPSS 30s": "#cb181d", "SPSS 100s": "#fc9272", "CALSPEC 10s": "#08306b"}
for (grp, v), g in B.groupby(["grp", "variant"]):
    b = g.groupby("snr_lo").res.median()
    ax[0].plot(b.index, 1e3 * (b - b.iloc[-1]), sty[v][0], marker=sty[v][1], color=colr.get(grp, "k"), label=f"{grp} {v}")
ax[0].set(xscale="log", xlabel="SNR", ylabel="XP residual rel. to brightest bin [mmag]", title="field residual vs SNR per background variant")
ax[0].axhline(0, color="k", lw=0.6); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=6.5, ncol=2)
for v, c in [("V0", "#444"), ("V1", "#C25B4E"), ("V2", "#2E8B6E")]:
    ax[1].scatter(E.grp, E[f"p_{v}"], color=c, s=20, alpha=0.7, label=f"pedestal p, {v}")
ax[1].axhline(0, color="k", lw=0.6); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
ax[1].set(ylabel="p [ADU/pix] from 5/10-arcsec pair", title="per-frame pedestal by variant")
fig.suptitle("A8  Background pilot: V0 default GLOBAL/64, V1 LOCAL annulus 12\", V2 GLOBAL/32")
fig.tight_layout(); fig.savefig(LOG / "A8_bkg_pilot.png", dpi=130)
print("saved A8")
