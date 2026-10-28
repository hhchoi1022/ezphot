#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261003_calspec_atm_test.py — do the large red-filter CALSPEC offsets (section 8.1, S1: WD0308-565, WD2149+021, WD2039-682,
WD1919+145 in m800 / m825) go away with a different atmosphere?

Same frames and the same frame cleaning as section 8.1 (261001_calspec_manuf_beam.py: keep flags from
calspec_frames_flags.csv, std SNR > 10, faint-star bias, global field ZP without the standard), manufacturer·beam n_eff(lam0)
curve, filters m600 and redder.  Only the atmosphere changes, for the field ZP stars and the standard alike:
  - PWV 2.5 / 5.0 / 7.5 (the curve in use) / 10.0 / 20.0 / 30.0 mm at airmass 1.23: the El Sauce curve times the SkyCalc
    ratio T(PWV) / T(7.5 mm) (water only; reproduces the stored 5 / 10 mm curves; 30 mm is the SkyCalc maximum);
    15 mm extrapolates the 7.5 -> 10 mm ratio, 50 / 100 mm extrapolate the 7.5 -> 30 mm ratio with the water optical depth
    linear in PWV (unresolved saturated lines grow slower in reality, so these overestimate the absorption: an upper bound)
  - airmass X = 1.0 ... 2.2 at PWV 7.5 mm: T(X) = T(1.23)^(X / 1.23); "airmass-matched" picks per frame the X closest to
    the frame airmass
Outputs (~/ezphot/log/20261003_calspec_atm/): atm_frames.csv, atm_resid.csv (star / filter / variant), A1_atm_resid.png
  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 python -u ezphot/analysis/261003_calspec_atm_test.py --n-proc 32
"""
import argparse
import importlib
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from astropy.io import fits       # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
CM = importlib.import_module("261001_calspec_manuf_beam")
SO, SS, SF, R = CM.SO, CM.SS, CM.SF, CM.R
OUT = Path("/home/hhchoi1022/ezphot/log/20261003_calspec_atm")
CAL = Path("/home/hhchoi1022/ezphot/log/20261001_calspec_manuf")
ATM = Path("/home/hhchoi1022/ezphot/log/20260910_transmission_sources")
RED_MIN = 600
PWVS = [2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
XS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
X_REF = 1.23
STARS_FOCUS = ["WD0308-565", "WD2149+021", "WD2039-682", "WD1919+145"]


def atmospheres():
    """{label: transmission on the lam_air grid of atmosphere_elsauce.csv}"""
    a = pd.read_csv(ATM / "atmosphere_elsauce.csv")
    sky = {p: np.asarray(fits.open(ATM / f"skytable_elsauce_pwv{p}.fits")[1].data["trans"], float)
           for p in ("2.5", "5.0", "7.5", "10.0", "20.0", "30.0")}
    base = a.trans.to_numpy(float)
    ratio = lambda p: np.where(sky["7.5"] > 0, sky[p] / np.maximum(sky["7.5"], 1e-12), 1.0)   # noqa: E731
    out = {"pwv2.5": base * ratio("2.5"), "pwv5.0": base * ratio("5.0"), "pwv7.5": base, "pwv10.0": base * ratio("10.0"),
           "pwv15.0": base * ratio("10.0") ** 3,               # water optical depth is ~linear in PWV: 7.5 -> 15 = 3 x (7.5 -> 10)
           "pwv20.0": base * ratio("20.0"), "pwv30.0": base * ratio("30.0"),
           "pwv50.0": base * ratio("30.0") ** ((50 - 7.5) / 22.5), "pwv100.0": base * ratio("30.0") ** ((100 - 7.5) / 22.5)}
    for x in XS:
        out[f"X{x:.1f}"] = np.clip(base, 1e-12, None) ** (x / X_REF)
    return a.lam_air.to_numpy(float), out


def totals():
    TC, DS = SO.TC, SO.DS
    lam_air, atm = atmospheres()
    qe = pd.read_csv(TC.SRC / "QE.csv")
    opt = pd.read_csv(TC.SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    out = {}
    for p in sorted((TC.VBASE / "manufacturer").glob("m*.csv")):
        f = p.stem
        if f in TC.BAD_FILTERS or int(f[1:4]) < RED_MIN:
            continue
        d = pd.read_csv(p)
        lam = d.lam_nm.to_numpy(float); tr = d.trans.to_numpy(float)
        lam0 = np.trapz(lam * tr, lam) / np.trapz(tr, lam)
        tb = TC.dr500_transmission(lam, tr, n_eff=CM.n_eff(lam0), f_number=3.03, obscuration=0.59)
        base = np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
        for lab, t in atm.items():
            out.setdefault(("atm", lab), {})[f] = (DS.air_to_vac_nm(lam), tb * base * np.interp(lam, lam_air, t))
    return out


def _init():
    warnings.filterwarnings("ignore")
    SO.AN._init()
    SO.TW._init(40)
    SO._TOTALS = totals()


def run_frames(n_proc):
    M = pd.read_csv(SO.OUT / "rephot_meta.csv")
    red = M["filter"].str.startswith("m") & (pd.to_numeric(M["filter"].str[1:4], errors="coerce") >= RED_MIN)
    M = M[M.status.isin(["ok", "cached"]) & (M.dataset == "CALSPEC") & red]
    M = M[[(SO.CATD / (Path(f).name[:-5] + "_cat.fits")).exists() for f in M.file]]
    jobs = [(obj, list(zip(g.file, g["filter"], g.unit))) for (obj, _), g in M.groupby(["object", "filter"])]
    print(time.strftime("[%H:%M:%S]"), f"CALSPEC red frames {len(M)}, jobs {len(jobs)}", flush=True)
    rows = []
    with Pool(n_proc, initializer=_init) as pool:
        for r in pool.imap_unordered(SO.process_cal, jobs):
            rows += r
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "atm_frames.csv", index=False)
    print(time.strftime("[%H:%M:%S]"), f"saved {len(F)} rows, {F.file.nunique()} frames", flush=True)
    return F


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=32); ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fc = OUT / "atm_frames.csv"
    F = run_frames(a.n_proc) if a.rerun or not fc.exists() else pd.read_csv(fc)
    # the frame selection of section 8.1 (keep = not cloudy night / colour-term outlier / wrong filter)
    K = pd.read_csv(CAL / "calspec_frames_flags.csv", usecols=["file", "curve", "keep"])
    K = K[K.curve == "beam"][["file", "keep"]].drop_duplicates("file")
    F = F.merge(K, on="file", how="inner")
    F = F[F.keep & (F.std_snr > 10)].copy()
    bias = SS.expected_bias(F.std_peak_e.to_numpy(), "CALSPEC") / 1e3
    F["d"] = (F.d_gnd - F.zp + F.zp_x - bias) * 1e3
    F = F[F.d.notna() & (F.d.abs() <= 1e3 * SS.D_MAX)]
    # airmass-matched: per frame the X variant closest to the frame airmass
    xs = np.array(XS)
    fr = F[F.variant.str.startswith("X")].copy()
    fr["xv"] = fr.variant.str[1:].astype(float)
    near = xs[np.abs(fr.airmass.to_numpy()[:, None] - xs[None, :]).argmin(axis=1)]
    Xm = fr[np.isclose(fr.xv, near)].assign(variant="X_matched")
    F = pd.concat([F, Xm], ignore_index=True)
    P = F.groupby(["object", "filter", "variant"]).agg(resid=("d", "median"), n=("d", "size"), snr=("std_snr", "median"),
                                                     airmass=("airmass", "median")).reset_index()
    P.round(2).to_csv(OUT / "atm_resid.csv", index=False)
    # check: pwv7.5 reproduces section 8.1 (manuf·beam)
    L = pd.read_csv(CAL / "calspec_sed_long.csv")
    chk = P[P.variant == "pwv7.5"].merge(L[L.curve == "beam"][["object", "filter", "resid"]], on=["object", "filter"], suffixes=("", "_81"))
    print("pwv7.5 vs section 8.1 manuf·beam: max |diff| =", round(float((chk.resid - chk.resid_81).abs().max()), 3), "mmag over", len(chk))
    fk = lambda f: int(f[1:4])   # noqa: E731
    W = P.pivot_table(index=["object", "filter"], columns="variant", values="resid")
    order = [f"pwv{p}" for p in PWVS] + ["X_matched"] + [f"X{x:.1f}" for x in XS]
    W = W[[c for c in order if c in W.columns]]
    pd.set_option("display.width", 250)
    for o in STARS_FOCUS + sorted(set(P.object) - set(STARS_FOCUS)):
        q = W.loc[o] if o in W.index.get_level_values(0) else None
        if q is None:
            continue
        q = q.reindex(sorted(q.index, key=fk))
        print(f"== {o} (airmass median {P[P.object == o].airmass.median():.2f})\n", q.round(1).to_string())
    # the change caused by the atmosphere (variant - pwv7.5), per filter: median and max over stars
    D = W.sub(W["pwv7.5"], axis=0)
    print("change vs pwv7.5, |max| over stars per filter:\n", D.abs().groupby(level="filter").max().round(1).reindex(sorted(set(D.index.get_level_values(1)), key=fk)).to_string())

    # A1: residual per filter for the focus stars and the rest, PWV variants and airmass-matched
    stars = STARS_FOCUS + [o for o in sorted(P.object.unique()) if o not in STARS_FOCUS]
    nc = 4; nr = int(np.ceil(len(stars) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.3 * nr), sharex=True, sharey=True, squeeze=False)
    cols = {"pwv2.5": "#9ecae1", "pwv7.5": "k", "pwv15.0": "#fc9272", "pwv30.0": "#cb181d", "pwv50.0": "#a50f15", "pwv100.0": "#67000d",
            "X_matched": "#2ca25f"}
    for ax, o in zip(axs.ravel(), stars):
        for v, c in cols.items():
            q = P[(P.object == o) & (P.variant == v)].copy()
            if not len(q):
                continue
            q = q.assign(k=q["filter"].map(fk)).sort_values("k")
            ax.plot(q.k + (list(cols).index(v) - 2.5) * 1.2, q.resid, "o-" if v == "pwv7.5" else ".-", color=c, ms=4 if v == "pwv7.5" else 3,
                    lw=1.2 if v == "pwv7.5" else 0.8, label={"X_matched": "airmass matched (PWV 7.5)"}.get(v, f"PWV {v[3:]} mm (X 1.23)"))
        ax.axhline(0, color="k", lw=0.5); ax.axhspan(-10, 10, color="0.92", zorder=0); ax.grid(alpha=0.3); ax.set_ylim(-80, 130)
        ax.set_title(f"{o} (X {P[P.object == o].airmass.median():.2f})", fontsize=8.5, loc="left")
        for wl in (720, 760, 820):
            ax.axvline(wl, color="#d4b06a", lw=0.6, ls=":")
    for ax in axs.ravel()[len(stars):]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.supxlabel("filter centre [nm] (dotted: H2O 720, O2 A 760, H2O 820 nm)"); fig.supylabel("residual obs − syn [mmag], manuf·beam")
    fig.suptitle("A1  CALSPEC 2024 red filters: residual with different atmospheres (applied to the field ZP stars and the standard alike)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96)); fig.savefig(OUT / "A1_atm_resid.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
