#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261001_std_line_filters.py — standard stars with strong absorption lines, analysed only in the filters the lines fall in.

Line depth in a band: per observed standard (ground spectrum, SPSS V2 / CALSPEC) a pseudo-continuum C = upper envelope of
log F_lambda (asymmetric least squares, smoothness lam = 1e6 on a 0.5 nm grid 345-915 nm, weight 0.99 above / 0.01 below,
after a 2.5 nm median filter).  For each filter and curve
  L      = syn_mag(F) - syn_mag(C)                     [mag; + = the lines take that fraction of the band's photons]
  S_line = dm/dδ(F) - dm/dδ(C)                         [mmag/nm; the part of the curve-shift sensitivity due to lines]
  feature = line window (H, He I, Ca II, Mg b, Na D, TiO, ...) with the largest transmission-weighted absorption
Strong-line star: L >= 0.05 mag in at least one filter (manufacturer·beam); a line star in a filter has L >= 0.03.
Groups: hot (BP-RP < 0.3: DA, DB, DZ, sdB, sdO) and cool (BP-RP >= 0.5, not a white dwarf).  For DA stars in m375w-m425
the Balmer lines overlap the band edges on both sides and C is not defined (blend, flagged).
Residuals: ct_long.csv of 260930_std_ct_applied.py (10" aperture, local ZP without the standard): before and after the
field colour term, per star / filter / curve (frames: SPSS std SNR >= 100, CALSPEC SNR > 10 minus faint-star bias).
Star level: a star's median offset from the other stars of its group and dataset over its continuum filters (L < 0.02,
pivot >= 505 nm, >= 3 filters; manufacturer curve of the same beam / nobeam variant for every curve) is subtracted first
(removes flux-scale errors of the reference spectra, e.g. WD2105-820 -40 mmag in every filter); E_raw = the same fit
without it.
Line effect per filter and curve (all units pooled, per-star medians of ct_long) and per filter copy (unit; per-star
medians of the frames of that unit), separately for the hot and cool groups: over all stars of the group observed in that
filter (>= 2 per dataset), residual = c_dataset + b * (100 L) [model A] or + g * (BP-RP) [model B].
b [mmag per % of the band absorbed] measures the curve's relative transmission error where the lines sit (0 = the curve
weights the line wavelengths correctly); effect E = b * 100 * median L of the line stars (L >= 0.03); errors by bootstrap
over stars (the median L is not resampled).  Cells need >= 3 line stars, >= 5 stars and an L range >= 0.03; flagged:
lever (one star carries >= 50% of the within-dataset L variance) and blend; flagged cells are left out of L3 / L4.
Line position (edge lines, |S_line| >= 5 mmag/nm, per unit and pooled): where E crosses 0 as a function of the curve's
pivot wavelength (linear in the curves of that filter, which are close to pure shifts of each other), relative to
manufacturer·beam; 16-84% range of a joint bootstrap over stars (same resample for all curves) and the range when one
star is left out.  It re-expresses the information of section 9.11 through the lines of the hot stars only (same
photometry), so it is a consistency check, not an independent measurement.  SPSS V2 wavelengths are used as vacuum.
Figures (~/ezphot/log/20261001_std_line_filters/):
  L1_line_map.png        : L per strong-line star and filter, dominant feature
  L2_line_in_curves.png  : where the lines sit in each curve (normalized transmissions + F/C of representative stars)
  L3_line_residual.png   : residual vs L in the line filters (manufacturer·beam points, per-curve fits)
  L4_line_summary.png    : E per filter and curve, hot and cool stars, before / after the colour term
Tables: line_long.csv (star x filter x curve), line_effect.csv (pooled), line_effect_unit.csv, line_position.csv
"""
import importlib
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import scipy.sparse as sps               # noqa: E402
import scipy.sparse.linalg as spl        # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
MB = importlib.import_module("260930_std_mb_sed")
SF, R, V2 = MB.SF, MB.R, MB.V2
OUT = Path("/home/hhchoi1022/ezphot/log/20261001_std_line_filters")
CT_LONG = Path("/home/hhchoi1022/ezphot/log/20260930_std_ct_applied/ct_long.csv")
REF = "manuf·beam"
CURVES = MB.CURVES
GRID = np.arange(345.0, 915.01, 0.5)
L_STRONG, L_LINE = 0.05, 0.03
FEATURES = {   # name: (lo, hi) nm
    "Hε+Ca H": (392, 402), "Ca K": (390, 396), "Hδ": (402, 418), "Hγ": (425, 443), "Hβ": (474, 498), "Hα": (645, 668),
    "He I 447": (444, 450), "He I 471": (469, 474), "He I 492": (489, 495), "He I 502": (499, 504),
    "He I 588": (585, 590), "He I 668": (666, 670), "He I 707": (704, 709), "G band": (428, 434), "Mg b": (513, 522),
    "Na D": (587, 592), "Ca II IRT": (847, 869), "Paschen": (840, 900),
    "TiO 615-635": (614, 636), "TiO 655-680": (652, 682), "CaH 685-705": (684, 706), "TiO 705-730": (704, 731),
    "TiO/K I 755-790": (754, 791), "TiO 840-860": (840, 862),
}
H_LINES = {"Hε+Ca H", "Hδ", "Hγ", "Hβ", "Hα", "Paschen"}
HE_LINES = {"He I 447", "He I 471", "He I 492", "He I 502", "He I 588", "He I 668", "He I 707"}
METAL = {"Ca K", "G band", "Mg b", "Na D", "Ca II IRT"}
MOLEC = {"TiO 615-635", "TiO 655-680", "CaH 685-705", "TiO 705-730", "TiO/K I 755-790", "TiO 840-860"}
mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))   # noqa: E731


def continuum(w, f, lam=1e6, p=0.01):
    """upper envelope of log F_lambda by asymmetric least squares (Eilers 2005, weights reversed); returns C on w."""
    y = median_filter(np.interp(GRID, w, np.log10(np.clip(f, 1e-40, None))), 5)
    n = len(y)
    Dm = sps.diags([1, -2, 1], [0, -1, -2], shape=(n, n - 2))
    P = lam * Dm.dot(Dm.T)
    wt = np.ones(n)
    for _ in range(50):
        z = spl.spsolve((sps.spdiags(wt, 0, n, n) + P).tocsc(), wt * y)
        new = np.where(y > z, 1 - p, p)
        if np.array_equal(new, wt):
            break
        wt = new
    return 10 ** np.interp(w, GRID, z)


def feature(w, f, c, fw, ft, bprp):
    hot = bprp < 0.3
    allowed = H_LINES | ((HE_LINES | {"Ca K"}) if hot else METAL | (MOLEC if bprp >= 1.4 else set()))
    t = np.interp(w, fw, ft, left=0, right=0)
    t[t < 0.01 * ft.max()] = 0
    a = np.clip(1 - f / c, 0, None) * t * w
    best, val = "", 0.0
    for name, (lo, hi) in FEATURES.items():
        if name not in allowed:
            continue
        m = (w >= lo) & (w <= hi)
        if m.sum() > 1:
            v = np.trapz(a[m], w[m])
            if v > val:
                best, val = name, v
    tot_a = np.trapz(a, w)
    return best if tot_a > 0 and val >= 0.35 * tot_a else ("H/He lines" if hot else "metal lines")


def line_fit(g, ycol, colour, nb=1000, seed=0):
    """residual = c_dataset + b * (100 L) [+ g * BP-RP]; returns b, bootstrap error, intercepts per dataset."""
    ds_names = sorted(g.dataset.unique())
    dsi = np.searchsorted(ds_names, g.dataset.to_numpy())
    x, y, c = 100 * g.L.to_numpy(), g[ycol].to_numpy(), g.bprp.to_numpy()

    def solve(i):
        X = [(dsi[i] == k).astype(float) for k in range(len(ds_names))] + [x[i]] + ([c[i]] if colour else [])
        return np.linalg.lstsq(np.vstack(X).T, y[i], rcond=None)[0]

    k = solve(np.arange(len(g)))
    xd = x - np.array([x[dsi == j].mean() for j in range(len(ds_names))])[dsi]     # L lever within each dataset
    lever = float((xd ** 2).max() / (xd ** 2).sum()) if (xd ** 2).sum() > 0 else 1.0
    rng = np.random.default_rng(seed); bs = []
    for _ in range(nb):
        i = rng.integers(0, len(g), len(g))
        if np.bincount(dsi[i], minlength=len(ds_names)).min() >= 2 and np.ptp(x[i]) > 1:
            bs.append(solve(i)[len(ds_names)])
    return (float(k[len(ds_names)]), (float(np.std(bs)) if len(bs) > 50 else np.nan), dict(zip(ds_names, k[:len(ds_names)])),
            lever)


def line_position(D, E, keys, nb=1000, seed=1):
    """pivot wavelength at which the line effect E vanishes, per cell (keys) with a line on a band edge."""
    out = []
    for kv, e in E[~E.blend & ~E.lever].groupby(keys):
        if len(e) < 3 or REF not in e.curve.values or e.S_line_med.abs().median() < 5:
            continue
        kv = kv if isinstance(kv, tuple) else (kv,)
        lam0 = e.set_index("curve").lam
        m = np.ones(len(D), bool)
        for k, v in zip(keys, kv):
            m &= (D[k] == v).to_numpy()
        g = D[m & D.curve.isin(e.curve).to_numpy() & D.L.notna().to_numpy() & D.yb.notna().to_numpy()]
        g = g[g.groupby(["curve", "dataset"]).object.transform("size") >= 2]
        if g.empty:
            continue
        stars = g.drop_duplicates(["dataset", "object"])[["dataset", "object"]].apply(tuple, axis=1).tolist()
        G = {cv: q.set_index(["dataset", "object"]) for cv, q in g.groupby("curve")}

        def x0(sel):
            xs, ys = [], []
            for cv, q in G.items():
                qq = q.loc[[s_ for s_ in sel if s_ in q.index]].reset_index()
                if not len(qq) or qq.groupby("dataset").size().min() < 2 or np.ptp(qq.L) < 0.02:
                    return np.nan
                b, _, _, _ = line_fit(qq, "yb", False, nb=0)
                ln = qq[(qq.L >= L_LINE) & qq.strong_star]
                if not len(ln):
                    return np.nan
                xs.append(lam0[cv] - lam0[REF]); ys.append(b * 100 * ln.L.median())
            if len(xs) < 3:
                return np.nan
            k = np.polyfit(xs, ys, 1)
            return -k[1] / k[0] if k[0] != 0 else np.nan
        best = x0(stars)
        rng = np.random.default_rng(seed)
        bs = np.array([x0([stars[j] for j in rng.integers(0, len(stars), len(stars))]) for _ in range(nb)])
        bs = bs[np.isfinite(bs)]
        jk = {t: x0([s_ for s_ in stars if s_ != t]) for t in stars}
        jv = np.array([v for v in jk.values() if np.isfinite(v)])
        worst = max(jk, key=lambda t: abs(jk[t] - best) if np.isfinite(jk[t]) else -1)
        out.append(dict(zip(keys, kv), datasets="+".join(sorted(g.dataset.unique())), n_stars=len(stars), n_curves=len(e),
                        feature=e.feature.iloc[0], S_line_med=round(e.S_line_med.median(), 1), x0_nm=round(best, 2),
                        x0_p16=round(float(np.percentile(bs, 16)), 2) if len(bs) > 100 else np.nan,
                        x0_p84=round(float(np.percentile(bs, 84)), 2) if len(bs) > 100 else np.nan,
                        jack_min=round(float(jv.min()), 2) if len(jv) else np.nan,
                        jack_max=round(float(jv.max()), 2) if len(jv) else np.nan, jack_worst=worst[1]))
    return pd.DataFrame(out)


def effect_table(D, keys):
    """line effect per cell; D needs L, yb, ya, resid_before, strong_star, blend, lam, S_line, feature, bprp, dataset."""
    eff = []
    for kv, g in D[D.group.isin(["hot", "cool"]) & D.L.notna() & D.yb.notna()].groupby(keys):
        g = g[g.groupby("dataset").object.transform("size") >= 2]
        ln = g[(g.L >= L_LINE) & g.strong_star]
        if len(ln) < 3 or len(g) < 5 or np.ptp(g.L) < 0.03:
            continue
        bA, eA, icA, hA = line_fit(g, "yb", False)
        bB, eB, _, _ = line_fit(g, "yb", True)
        bC, eC, _, _ = line_fit(g, "ya", False)
        bR, eR, _, _ = line_fit(g, "resid_before", False)
        Lm = 100 * ln.L.median()
        eff.append(dict(zip(keys, kv), lam=g.lam.median(), n=len(g), n_line=len(ln), datasets="+".join(sorted(g.dataset.unique())),
                        L_med=round(Lm / 100, 4), L_max=round(g.L.max(), 4), S_line_med=round(ln.S_line.median(), 2),
                        feature=ln.feature.mode().iloc[0], blend=bool(ln.blend.any()), lever_share=round(hA, 2), lever=hA >= 0.5,
                        b=round(bA, 3), b_err=round(eA, 3), E=round(bA * Lm, 1), E_err=round(eA * Lm, 1),
                        b_col=round(bB, 3), b_col_err=round(eB, 3), E_col=round(bB * Lm, 1),
                        b_after=round(bC, 3), b_after_err=round(eC, 3), E_after=round(bC * Lm, 1), E_after_err=round(eC * Lm, 1),
                        E_raw=round(bR * Lm, 1), E_raw_err=round(eR * Lm, 1), icpt=icA))
    return pd.DataFrame(eff)


def unit_table(T):
    """per-star medians over the frames of each unit (filter copy), with L, group and star level from T."""
    CT = importlib.import_module("260930_std_ct_applied")
    F = CT.load_frames(CT.APS["10\""][0], "", CT.std_colours())
    F["curve"] = F.version.map(MB.LAB) + "·" + F.variant
    U = F.groupby(["dataset", "object", "filter", "curve", "unit"]).agg(
        resid_before=("d", "median"), resid=("d_ct", "median"), n_frames=("d", "size")).reset_index()
    keep = ["dataset", "object", "filter", "curve", "L", "S_line", "group", "strong_star", "blend", "lam", "bprp", "feature",
            "lvl_yb", "lvl_ya"]
    U = U.merge(T[keep], on=["dataset", "object", "filter", "curve"])
    U["yb"] = U.resid_before - U.lvl_yb
    U["ya"] = U.resid - U.lvl_ya
    return U


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tot = V2.build_totals()
    C = pd.read_csv(CT_LONG)
    C = C[C.aperture == "10\""].drop(columns="aperture")
    spec = SF.spectra(C[C.dataset == "SPSS"].object.unique(), C[C.dataset == "CALSPEC"].object.unique())
    cont = {k: continuum(*v) for k, v in spec.items()}

    cache = OUT / "line_long.csv"
    if cache.exists() and "--recompute" not in sys.argv:
        T = pd.read_csv(cache)
    else:
        T = line_table(C, spec, cont, tot)
        T.to_csv(cache, index=False)
    analyse(T, spec, cont, tot)


def line_table(C, spec, cont, tot):
    lab_of = {v: k for k, v in MB.LAB.items()}
    rows = []
    for r in C.itertuples(index=False):
        ver, var = lab_of[r.curve.split("·")[0]], r.curve.split("·")[1]
        w, f = spec[(r.dataset, r.object)]; c = cont[(r.dataset, r.object)]
        fw, ft = tot[(ver, var)][r.filter]
        sup = fw[ft > 1e-3 * ft.max()]
        ok = sup.min() > GRID[0] + 2 and sup.max() < GRID[-1] - 2
        L = float(R.syn_mag(w, f, fw, ft) - R.syn_mag(w, c, fw, ft)) if ok else np.nan
        s_line = (SF.sens(w, f, fw, ft) - SF.sens(w, c, fw, ft)) if ok else np.nan
        hot = r.bprp < 0.3
        rows.append(dict(r._asdict(), L=round(L, 5), S_line=round(s_line, 2), hot=hot,
                         group="hot" if hot else ("cool" if r.bprp >= 0.5 and r.cls != "WD" else "other"),
                         feature=feature(w, f, c, fw, ft, r.bprp) if ok and L >= 0.01 else ""))
    T = pd.DataFrame(rows)
    ref = T[T.curve == REF].groupby(["dataset", "object"]).L.max()
    T["strong_star"] = [ref.get((d, o), 0) >= L_STRONG for d, o in zip(T.dataset, T.object)]
    return T


def analyse(T, spec, cont, tot):
    T["blend"] = T.sptype.str.startswith("DA") & (T.lam < 430)       # Balmer lines on both band edges, C undefined
    # star level: a star's median offset from the other stars of its group and dataset over its continuum filters
    # (L < 0.02 with manufacturer·beam, pivot >= 505 nm, >= 3 filters); removes spectrum flux-scale errors
    Lref = T[T.curve == REF].set_index(["dataset", "object", "filter"]).L
    contf = np.array([Lref.get((d, o, f), np.nan) < 0.02 for d, o, f in zip(T.dataset, T.object, T["filter"])]) & (T.lam >= 505)
    # (taken from the manufacturer curve of the same beam / nobeam variant, which covers every filter)
    var = T.curve.str.split("·").str[1]
    man = T.curve.str.startswith("manuf")
    for col, out in (("resid_before", "yb"), ("resid", "ya")):
        dev = (T[col] - T.groupby(["dataset", "group", "filter", "curve"])[col].transform("median")).where(contf & man)
        n = dev.notna().groupby([T.dataset, T.object, var]).transform("sum")
        lv = dev.groupby([T.dataset, T.object, var]).transform("median").where(n >= 3)
        lv = pd.Series(lv.where(man).groupby([T.dataset, T.object, var]).transform("max").to_numpy(), index=T.index)
        T[out] = T[col] - lv
        T["lvl_" + out] = lv
    # line effect: regression of the star-normalized residual on L with a level per dataset; pooled and per unit
    E = effect_table(T, ["group", "filter", "curve"]).sort_values(["lam", "group", "curve"])
    T["y_before"] = np.nan
    for e in E.itertuples(index=False):
        m = (T.group == e.group) & (T["filter"] == e.filter) & (T.curve == e.curve) & T.yb.notna()
        T.loc[m, "y_before"] = T.yb[m] - T.dataset[m].map(e.icpt)
    E.drop(columns="icpt").to_csv(OUT / "line_effect.csv", index=False)
    U = unit_table(T)
    EU = effect_table(U, ["group", "filter", "unit", "curve"]).sort_values(["lam", "unit", "curve"])
    EU.drop(columns="icpt").to_csv(OUT / "line_effect_unit.csv", index=False)
    Pz = pd.concat([line_position(T, E, ["group", "filter"]).assign(unit="all"),
                    line_position(U, EU, ["group", "filter", "unit"])], ignore_index=True)
    SFT = pd.read_csv(Path("/home/hhchoi1022/ezphot/log/20260930_std_shift_fit/shift_fit.csv"))
    SFT = SFT[(SFT.version == "manufacturer") & (SFT.variant == "beam")]
    d911 = {(f, u): "; ".join(f"{d} {v:+.2f}±{e:.2f}" for d, v, e in zip(q.dataset, q.delta, q.delta_err) if np.isfinite(v))
            for (f, u), q in SFT.groupby(["filter", "unit"])}
    Pz["delta_9_11"] = [d911.get((f, u), "") for f, u in zip(Pz["filter"], Pz.unit)]
    Pz.to_csv(OUT / "line_position.csv", index=False)
    E = E[~E.blend & ~E.lever]

    # L1: line-depth map of the strong-line stars (manufacturer·beam)
    filt_order = sorted(T["filter"].unique(), key=lambda s: int(s[1:4]))
    S = T[(T.curve == REF) & T.strong_star].copy()
    S["star"] = S.object + " (" + S.sptype + ", " + S.dataset.str[0] + ")"
    stars = S.drop_duplicates("star").sort_values(["group", "bprp"]).star.tolist()
    M = S.pivot_table(index="star", columns="filter", values="L").reindex(index=stars, columns=filt_order)
    Fm = S.pivot_table(index="star", columns="filter", values="feature", aggfunc="first").reindex(index=stars, columns=filt_order)
    fig, ax = plt.subplots(figsize=(19, 0.36 * len(stars) + 2.5))
    im = ax.imshow(100 * M.to_numpy(), aspect="auto", cmap="magma_r", vmin=0, vmax=30)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if np.isfinite(v) and v >= L_LINE:
                ax.text(j, i, f"{100 * v:.0f}\n{Fm.iat[i, j]}", ha="center", va="center", fontsize=4.6,
                        color="w" if v > 0.15 else "k")
    ax.set_xticks(range(len(filt_order))); ax.set_xticklabels(filt_order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(stars))); ax.set_yticklabels(stars, fontsize=7)
    plt.colorbar(im, ax=ax, pad=0.01).set_label("L = line absorption in the band [%]")
    ax.set_title(f"L1  Strong-line standards (L ≥ {100 * L_STRONG:.0f}% in some filter): fraction of the band's photons taken by "
                 f"absorption lines, manufacturer·beam; number + dominant feature where L ≥ {100 * L_LINE:.0f}%; blank = not observed "
                 f"in that filter", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "L1_line_map.png", dpi=110); plt.close(fig)

    # line filters (group, filter) with a fit for at least one curve, ordered by wavelength
    LF = E.assign(isref=E.curve != REF).sort_values(["isref", "lam"]).drop_duplicates(["group", "filter"]).sort_values("lam")
    keyf = [(g, f) for g, f in zip(LF.group, LF["filter"])]
    nc = 5; nr = int(np.ceil(len(keyf) / nc))

    # L2: line profiles in the curves of each line filter (representative = median-L line star)
    fig, axs = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.0 * nr), squeeze=False)
    for ax, (g, f) in zip(axs.ravel(), keyf):
        cv0 = LF[(LF.group == g) & (LF["filter"] == f)].curve.iloc[0]
        q = T[(T.curve == cv0) & T.strong_star & (T["filter"] == f) & (T.group == g) & (T.L >= L_LINE)].sort_values("L")
        rp = q.iloc[len(q) // 2]
        w, fl = spec[(rp.dataset, rp.object)]; c = cont[(rp.dataset, rp.object)]
        lo = hi = None
        for (ver, var) in CURVES:
            if f not in tot[(ver, var)]:
                continue
            fw, ft = tot[(ver, var)][f]
            mk, col = MB.STYLE[f"{MB.LAB[ver]}·{var}"]
            ax.plot(fw, ft / ft.max(), "-", color=col, lw=0.9, label=f"{MB.LAB[ver]}·{var}")
            sup = fw[ft > 0.01 * ft.max()]
            lo = sup.min() if lo is None else min(lo, sup.min()); hi = sup.max() if hi is None else max(hi, sup.max())
        m = (w > lo - 8) & (w < hi + 8)
        ax.plot(w[m], (fl / c)[m], "-", color="tab:blue", lw=0.9)
        ax.set_xlim(lo - 8, hi + 8); ax.set_ylim(0, 1.12); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        ax.set_title(f"{f} ({g}): {rp.feature} — {rp.object} {rp.sptype}, L {100 * rp.L:.0f}%", fontsize=7.5, loc="left")
    for ax in axs.ravel()[len(keyf):]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=5.5, loc="lower left")
    fig.suptitle("L2  Line filters: normalized total transmission of each curve (lines) and the line profile F / continuum of a "
                 "representative strong-line star (blue)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "L2_line_in_curves.png", dpi=105); plt.close(fig)

    # L3: residual minus the fitted dataset level vs L; points manufacturer·beam, lines b per curve
    fig, axs = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.2 * nr), squeeze=False)
    for ax, (g, f) in zip(axs.ravel(), keyf):
        for cv in MB.STYLE:
            e = E[(E.group == g) & (E["filter"] == f) & (E.curve == cv)]
            if e.empty:
                continue
            mk, col = MB.STYLE[cv]
            q = T[(T["filter"] == f) & (T.group == g) & (T.curve == cv) & T.y_before.notna()]
            if cv == REF or (REF not in E[(E.group == g) & (E["filter"] == f)].curve.values and cv == e.curve.iloc[0]):
                for ds, mfc in (("SPSS", col), ("CALSPEC", "none")):
                    z = q[q.dataset == ds]
                    ax.plot(100 * z.L, z.y_before, mk, mfc=mfc, mec=col, ms=4, ls="none")
            xx = np.array([0, 100 * q.L.max()])
            ax.plot(xx, e.b.iloc[0] * xx, "-", color=col, lw=0.9, label=f"{cv}: b {e.b.iloc[0]:+.2f}±{e.b_err.iloc[0]:.2f}")
        ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        e0 = LF[(LF.group == g) & (LF["filter"] == f)].iloc[0]
        ax.set_title(f"{f} ({g}, {e0.feature})", fontsize=8, loc="left")
        ax.set_xlabel("L [%]", fontsize=7); ax.set_ylabel("Δ − level [mmag]", fontsize=7)
        ax.legend(fontsize=5, loc="best")
    for ax in axs.ravel()[len(keyf):]:
        ax.axis("off")
    fig.suptitle("L3  Residual before the colour term minus the star's own level (continuum filters) and the fitted dataset level, "
                 "against the line absorption L,\nfor the hot (BP−RP < 0.3) or cool (≥ 0.5) stars in each line filter; points = "
                 "manufacturer·beam (filled SPSS, open CALSPEC), lines = slope b of each curve [mmag per %]", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "L3_line_residual.png", dpi=105); plt.close(fig)

    # L4: line effect E = b x median L of the line stars, per filter and curve
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    xpos = {k: i for i, k in enumerate(keyf)}
    for ax, col_, err_, tt in ((axs[0], "E", "E_err", "before the colour term"),
                               (axs[1], "E_after", "E_after_err", "after the colour term")):
        for j, cv in enumerate(MB.STYLE):
            q = E[E.curve == cv]
            if q.empty:
                continue
            mk, col = MB.STYLE[cv]
            x = np.array([xpos[(g, f)] for g, f in zip(q.group, q["filter"])]) + (j - 3.5) * 0.09
            ax.errorbar(x, q[col_], q[err_], fmt=mk, color=col, ms=4.5, capsize=0, lw=0.8, label=cv)
        ax.axhline(0, color="k", lw=0.5); ax.axhspan(-10, 10, color="0.92", zorder=0); ax.grid(alpha=0.3)
        ax.set_ylim(-80, 50)
        ax.set_ylabel("E = b × L_med [mmag]", fontsize=9)
        ax.set_title(f"{tt}: residual of a typical line star relative to a line-free star of the same group and dataset, "
                     f"from the fit residual = level + b·L; bootstrap error over stars", fontsize=9, loc="left")
    lab = [f"{f}\n{e.feature}\n{g}, n={e.n_line}/{e.n}\nL {100 * e.L_med:.0f}%"
           for (g, f), e in zip(keyf, [LF[(LF.group == g) & (LF["filter"] == f)].iloc[0] for g, f in keyf])]
    axs[1].set_xticks(range(len(keyf))); axs[1].set_xticklabels(lab, fontsize=6.5)
    axs[0].legend(fontsize=7, ncol=8, loc="upper right")
    fig.suptitle("L4  Line filters only: line effect per curve (n = line stars / all stars of the group in the fit; L = median line "
                 "absorption of the line stars)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "L4_line_summary.png", dpi=110); plt.close(fig)

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print("strong-line stars:", S.drop_duplicates("star").groupby("group").star.apply(list).to_dict())
    print(E.drop(columns=["blend"]).to_string(index=False))
    print("\nline position (E = 0) relative to manufacturer·beam [nm]:\n" + Pz.to_string(index=False))
    print("\nper-unit line effect (hot, edge filters):")
    print(EU[(EU.group == "hot") & EU["filter"].isin(["m466w", "m475", "m500", "m650", "m675"])][
        ["filter", "unit", "curve", "datasets", "n", "n_line", "E", "E_err", "lever_share"]].to_string(index=False))
    print("\nK/M stars (no star level, not in the fits): residual before the colour term in their strongest line filters")
    KM = T[(T.group == "cool") & (T.bprp >= 1.4)]
    top = KM[KM.curve == REF].sort_values("L", ascending=False).groupby("object").head(4)[["dataset", "object", "filter", "feature", "L"]]
    pv = KM.merge(top[["dataset", "object", "filter"]]).pivot_table(index=["object", "filter"], columns="curve", values="resid_before")
    print(pv.join(top.set_index(["object", "filter"])[["feature", "L"]]).round(3).to_string())
    print("\nper-star values in the line filters, manuf·beam:")
    P = T[(T.curve == REF) & T.y_before.notna()]
    print(P[["dataset", "object", "sptype", "group", "filter", "feature", "L", "S_line", "resid_before", "y_before", "resid"]]
          .sort_values(["filter", "group", "L"]).round(3).to_string(index=False))

if __name__ == "__main__":
    main()
