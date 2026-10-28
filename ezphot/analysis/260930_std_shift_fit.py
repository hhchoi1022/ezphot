#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260930_std_shift_fit.py — curve position from the observed standard stars: regress each star's offset on its
sensitivity to a curve shift.

For a curve that is off by delta nm (delta = true - adopted centre), a standard's offset
  Delta_i = m_inst + ZP - syn_i  moves by  S_i x delta  (+ a constant common to all stars, absorbed with the ZP),
with S_i = dm/d(shift) of star i through the adopted curve [mmag/nm] (+-0.5 nm central difference, own ground
spectrum).  So per (dataset, filter, unit, curve) the slope of Delta_i against S_i estimates delta; the intercept is
free, which makes the fit independent of the field-star reference and of the ZP level (the field ZP only adds a
constant).  If the curve is right, delta = 0; different curves should give the same 'preferred centre'
c_adopted + delta (c = effective centroid of the total curve).  Checks: delta(nobeam) - delta(beam) ~ c(beam) -
c(nobeam) (the beam shift); delta(mhj) - delta(manufacturer) ~ -delta_lab.

Offsets (std_offset_frames.csv, screened; ground truth, local ZP):
  SPSS    : std SNR >= 100 (default); CALSPEC: std SNR > 10 with the faint-star bias removed (as in 9.4).
  per star: median over frames of that unit.  Fit: OLS; error = max(star bootstrap without a spread floor, analytic);
  Theil-Sen as a check; needs >= 5 stars and a spread of S >= 8 mmag/nm; one_star flags fits with a leverage >= 0.8.
Non-linear check (manufacturer beam/nobeam): the curve is also shifted on a grid of -5..+5 nm and delta is the grid
value minimising the rms of Delta_i - [m_i(adopted + delta) - m_i(adopted)] about a free constant (delta_scan).
Output (~/ezphot/log/20260930_std_shift_fit/): star_delta_S.csv, shift_fit.csv, beam_check.csv, O12_std_shift_scatter.png,
O13_std_shift_centre.png, O14_std_offsets_by_curve.png
"""
import importlib
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
SS = importlib.import_module("260927_std_offset_summary")
CT = importlib.import_module("260927_std_colorterm")
AN, TW, R, V2 = CT.AN, CT.TW, CT.R, CT.V2
D = CT.D
OUT = Path("/home/hhchoi1022/ezphot/log/20260930_std_shift_fit")
LAB = {"manufacturer": "manuf", "version_mhj": "mhj", "version_ky": "ky", "version_hyp": "hyp"}
MIN_STARS, MIN_SPREAD = 5, 8.0


def spectra(objs_spss, objs_cal):
    from astropy.table import Table
    AN._init(); TW._init(40)
    sp = pd.read_csv(AN.SPSS_CSV)
    ids = {AN.norm(n): i for n, i in zip(sp.Name, sp.SPSS)}
    out = {}
    for o in objs_spss:
        s = AN.load_spss_spec(ids[AN.norm(o)]) if AN.norm(o) in ids else None
        if s is not None:
            out[("SPSS", o)] = s
    for o in objs_cal:
        st = TW._INFO.get(o, {})
        if st.get("spec"):
            t = Table.read(TW.CALSPEC_DIR / st["spec"])
            wl = np.asarray(t["WAVELENGTH"], float) / 10.0
            fl = np.asarray(t["FLUX"], float) * 1e-2
            m = (wl > 300) & (wl < 1100) & np.isfinite(fl) & (fl > 0)
            out[("CALSPEC", o)] = (wl[m], fl[m])
    return out


def sens(w, f, fw, ft):
    sup = fw[ft > 1e-3 * ft.max()]
    if w[0] > sup.min() - 1 or w[-1] < sup.max() + 1:
        return np.nan
    return 1e3 * float(R.syn_mag(w, f, fw, ft, 0.5) - R.syn_mag(w, f, fw, ft, -0.5))


def centroid(fw, ft):
    return float(np.trapz(fw * ft, fw) / np.trapz(ft, fw))


def fit(p, nboot=1000, seed=0):
    x, y = p.S.to_numpy(), p.d.to_numpy()
    if len(x) < MIN_STARS or np.ptp(x) < MIN_SPREAD:
        return dict(delta=np.nan, delta_err=np.nan, delta_ts=np.nan, icpt=np.nan, rms=np.nan, max_hat=np.nan)
    k = np.polyfit(x, y, 1)
    res = y - np.polyval(k, x)
    err_an = float(np.sqrt(np.sum(res ** 2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2)))
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(nboot):                                   # no spread floor: a fit that rests on one lever star
        i = rng.integers(0, len(x), len(x))                  # gets the large error it deserves
        if np.ptp(x[i]) > 0:
            b.append(np.polyfit(x[i], y[i], 1)[0])
    hat = 1 / len(x) + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
    from scipy.stats import theilslopes
    return dict(delta=k[0], delta_err=float(max(np.std(b), err_an)), delta_ts=float(theilslopes(y, x)[0]), icpt=k[1],
                rms=float(np.std(res)), max_hat=float(hat.max()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tot = V2.build_totals()
    F = SS.screened(pd.read_csv(D / "std_offset_frames.csv"))
    bias = SS.expected_bias(F.std_peak_e.to_numpy(), "CALSPEC") / 1e3
    F["d"] = np.where(F.dataset == "SPSS", np.where(F.std_snr >= 100, F.dl_gnd, np.nan),
                      np.where(F.std_snr > 10, F.dl_gnd - bias, np.nan)) * 1e3
    F = F[F.d.notna() & (F.d.abs() <= 1e3 * SS.D_MAX)]
    spec = spectra(F[F.dataset == "SPSS"].object.unique(), F[F.dataset == "CALSPEC"].object.unique())
    S = pd.read_csv(D / "std_colours.csv")
    cls = {(d, k): c for d, k, c in zip(S.dataset, S.key, S.cls)}

    # per star x filter x unit x curve: median offset and sensitivity
    P = F.groupby(["dataset", "object", "filter", "unit", "version", "variant"]).d.median().reset_index()
    sv = {}
    for (ds, o, fl, ver, var) in P[["dataset", "object", "filter", "version", "variant"]].drop_duplicates().itertuples(index=False):
        if (ds, o) in spec and (ver, var) in tot and fl in tot[(ver, var)]:
            sv[(ds, o, fl, ver, var)] = sens(*spec[(ds, o)], *tot[(ver, var)][fl])
    P["S"] = [sv.get(k, np.nan) for k in zip(P.dataset, P.object, P["filter"], P.version, P.variant)]
    P["cls"] = [cls.get((d, o if d == "CALSPEC" else AN.norm(o)), "?") for d, o in zip(P.dataset, P.object)]
    P = P[P.S.notna()]
    P.to_csv(OUT / "star_delta_S.csv", index=False)

    rows = []
    for (ds, fl, un, ver, var), g in P.groupby(["dataset", "filter", "unit", "version", "variant"]):
        r = fit(g)
        fw, ft = tot[(ver, var)][fl]
        c = centroid(fw, ft)
        rows.append(dict(dataset=ds, filter=fl, pivot=int(fl[1:4]), unit=un, curve=f"{LAB[ver]}·{var}", version=ver, variant=var,
                         n=len(g), S_spread=round(np.ptp(g.S), 1), centroid=round(c, 2),
                         delta=round(r["delta"], 2), delta_err=round(r["delta_err"], 2), delta_ts=round(r["delta_ts"], 2),
                         preferred=round(c + r["delta"], 2), rms=round(r["rms"], 1), max_hat=round(r["max_hat"], 2),
                         one_star=bool(r["max_hat"] >= 0.8) if np.isfinite(r["max_hat"]) else False))
    T = pd.DataFrame(rows).sort_values(["dataset", "pivot", "unit", "curve"])
    grid = np.round(np.arange(-5.0, 5.01, 0.1), 2)
    mcache = {}
    scan = []
    for t in T[T.delta.notna() & T.version.eq("manufacturer")].itertuples():
        g = P[(P.dataset == t.dataset) & (P["filter"] == t.filter) & (P.unit == t.unit) & (P.version == t.version) & (P.variant == t.variant)]
        fw, ft = tot[(t.version, t.variant)][t.filter]
        M = []
        for o in g.object:
            key = (t.dataset, o, t.filter, t.variant)
            if key not in mcache:
                w, f = spec[(t.dataset, o)]
                m0 = float(R.syn_mag(w, f, fw, ft, 0.0))
                mcache[key] = 1e3 * (np.array([float(R.syn_mag(w, f, fw, ft, dd)) for dd in grid]) - m0)
            M.append(mcache[key])
        M = np.array(M)                                     # stars x grid: predicted change of Delta for true = adopted + dd
        res = g.d.to_numpy()[:, None] - M
        rms = np.sqrt(np.mean((res - res.mean(axis=0)) ** 2, axis=0))
        scan.append((t.Index, grid[int(np.argmin(rms))], float(rms.min())))
    for i, d_, r_ in scan:
        T.loc[i, "delta_scan"] = d_
        T.loc[i, "rms_scan"] = round(r_, 1)
    T.to_csv(OUT / "shift_fit.csv", index=False)
    print(T[T.version.eq("manufacturer") & T.delta.notna()][["dataset", "filter", "unit", "curve", "n", "delta", "delta_err", "delta_ts", "delta_scan", "rms", "rms_scan"]].to_string(index=False))
    pd.set_option("display.width", 230); pd.set_option("display.max_rows", 500)
    W = T[T.delta.notna()].pivot_table(index=["dataset", "filter", "unit"], columns="curve", values="delta")
    print(W.round(2).to_string())
    # consistency: beam shift and lab offset
    chk = []
    for (ds, fl, un), g in T[T.delta.notna()].groupby(["dataset", "filter", "unit"]):
        gg = g.set_index("curve")
        for lab in ("manuf", "mhj", "ky", "hyp"):
            if f"{lab}·beam" in gg.index and f"{lab}·nobeam" in gg.index:
                b, n = gg.loc[f"{lab}·beam"], gg.loc[f"{lab}·nobeam"]
                chk.append(dict(dataset=ds, filter=fl, unit=un, lab=lab, d_nobeam_minus_beam=round(n.delta - b.delta, 2),
                                expected=round(b.centroid - n.centroid, 2), err=round(np.hypot(n.delta_err, b.delta_err), 2)))
    C = pd.DataFrame(chk)
    C.to_csv(OUT / "beam_check.csv", index=False)
    print(C[C.lab == "manuf"].to_string(index=False))

    # P1: Delta vs S, manufacturer beam, SPSS, one panel per (filter, unit) with a fit
    sel = T[(T.dataset == "SPSS") & (T.curve == "manuf·beam") & T.delta.notna()].sort_values("pivot")
    n = len(sel); nc = 6; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.8 * nr), squeeze=False)
    col = {"WD": "tab:blue", "sdOB": "tab:cyan", "OBA": "tab:green", "FGK": "tab:orange", "M": "tab:red", "?": "0.5"}
    for ax, (_, t) in zip(axs.flat, sel.iterrows()):
        g = P[(P.dataset == "SPSS") & (P["filter"] == t["filter"]) & (P.unit == t.unit) & (P.version == "manufacturer") & (P.variant == "beam")]
        ax.scatter(g.S, g.d, c=[col.get(c, "0.5") for c in g.cls], s=14)
        xx = np.linspace(g.S.min(), g.S.max(), 10)
        ax.plot(xx, (g.d.median() - t.delta * g.S.median()) + t.delta * xx, "k-", lw=1)
        ax.set_title(f"{t['filter']} {t.unit}: δ = {t.delta:+.2f} ± {t.delta_err:.2f} nm", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    for ax in axs.flat[n:]:
        ax.axis("off")
    fig.supxlabel("S = dm/d(shift) of the star through the adopted curve [mmag/nm]", fontsize=9)
    fig.supylabel("Δ (ground truth, local ZP) [mmag]", fontsize=9)
    fig.suptitle("O12  SPSS standards: offset vs sensitivity per filter and unit (manufacturer·beam). Slope = curve shift δ (true − adopted); "
                 "colours: blue WD, orange FGK, red M", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "O12_std_shift_scatter.png", dpi=110)

    # P2: preferred effective centre relative to manufacturer·beam centre, per curve
    ref = T[T.curve == "manuf·beam"].set_index(["dataset", "filter", "unit"]).centroid
    T["pref_rel"] = [p - ref.get((d, f, u), np.nan) for p, d, f, u in zip(T.preferred, T.dataset, T["filter"], T.unit)]
    T["adopt_rel"] = [c - ref.get((d, f, u), np.nan) for c, d, f, u in zip(T.centroid, T.dataset, T["filter"], T.unit)]
    mk = {"manuf·beam": ("o", "k"), "manuf·nobeam": ("o", "0.6"), "mhj·beam": ("s", "tab:purple"), "mhj·nobeam": ("s", "plum"),
          "hyp·beam": ("^", "tab:green"), "hyp·nobeam": ("^", "yellowgreen"), "ky·beam": ("D", "tab:red"), "ky·nobeam": ("D", "salmon")}
    fig, axs = plt.subplots(4, 1, figsize=(15, 17))
    panels = [("SPSS", "delta", "δ = how far THIS curve is off [nm]",
               "SPSS (a): offset of each adopted curve, δ = true − adopted centre. 0 = that curve is right; + = it should be redder"),
              ("SPSS", "pref_rel", "adopted centre + δ − manuf·beam centre [nm]",
               "SPSS (b): the same fits as one position, measured from the manuf·beam centre. Points of a filter coincide = the fit is consistent"),
              ("CALSPEC", "delta", "δ = how far THIS curve is off [nm]", "CALSPEC (c): as (a)"),
              ("CALSPEC", "pref_rel", "adopted centre + δ − manuf·beam centre [nm]", "CALSPEC (d): as (b)")]
    for ax, (ds, col, ylab, title) in zip(axs, panels):
        t = T[(T.dataset == ds) & T.delta.notna()]
        keys = sorted(t[["pivot", "unit"]].drop_duplicates().itertuples(index=False))
        xpos = {k: i for i, k in enumerate(keys)}
        for cv, (m, c) in mk.items():
            q = t[t.curve == cv]
            x = np.array([xpos[(p, u)] for p, u in zip(q["pivot"], q.unit)]) + (list(mk).index(cv) - 3.5) * 0.08
            ax.errorbar(x, q[col], q.delta_err, fmt=m, color=c, ms=4, lw=0.8, label=cv)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([f"m{p}\n{u[-2:]}" for p, u in keys], fontsize=7)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylim(-5, 5); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=8, loc="upper right")
    fig.suptitle("O13  Curve position from the observed standards (Δ vs S regression; SPSS SNR ≥ 100, CALSPEC SNR > 10 bias-corrected; local ZP)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "O13_std_shift_centre.png", dpi=110)
    T.to_csv(OUT / "shift_fit.csv", index=False)

    # P3: how much a star's offset changes between curve versions (most sensitive observed stars)
    stars = [("CALSPEC", "BD-11_3759"), ("SPSS", "LP885-23"), ("SPSS", "LTT377"), ("CALSPEC", "HZ4"), ("SPSS", "WD2039-682"),
             ("SPSS", "GD71"), ("SPSS", "G24-25")]
    fig, axs = plt.subplots(len(stars), 1, figsize=(14, 2.2 * len(stars)), sharex=True)
    for ax, (ds, o) in zip(axs, stars):
        g = P[(P.dataset == ds) & (P.object == o)]
        if g.empty:
            continue
        g = g.groupby(["filter", "version", "variant"]).d.median().reset_index()
        g["pivot"] = g["filter"].str[1:4].astype(int)
        base = g[(g.version == "manufacturer") & (g.variant == "beam")].set_index("filter").d
        for cv, (m, c) in mk.items():
            lab, var = cv.split("·")
            ver = {v: k for k, v in LAB.items()}[lab]
            q = g[(g.version == ver) & (g.variant == var)].sort_values("pivot")
            ax.plot(q["pivot"], q.d - q["filter"].map(base), m + "-", color=c, ms=3, lw=0.8, label=cv)
        ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3); ax.set_ylim(-80, 80)
        ax.set_title(f"{o} ({ds})", fontsize=8, loc="left"); ax.set_ylabel("Δ − Δ(manuf·beam)\n[mmag]", fontsize=7)
    axs[0].legend(fontsize=7, ncol=8)
    axs[-1].set_xlabel("filter pivot [nm]")
    fig.suptitle("O14  How much the offset of a star changes when only the adopted curve changes (same photometry; includes the field-ZP change)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "O14_std_offsets_by_curve.png", dpi=110)


if __name__ == "__main__":
    main()
