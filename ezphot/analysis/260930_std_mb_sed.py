#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260930_std_mb_sed.py — medium-band SED of every observed standard star for each transmission curve, compared with the
synthetic photometry of its reference spectrum, and the residuals.

For each star, filter and curve (manufacturer / version_mhj / version_hyp / version_ky x nobeam / beam):
  observed  m_obs = MAG_APER_3 + local ZP   (= Delta + syn, from std_offset_frames.csv; the ZP is set by field stars
                                             with corrected XP through the same curve)
  synthetic m_syn = AB magnitude of the star's ground spectrum (SPSS V2 / CALSPEC) through the same curve
  residual  Delta = m_obs - m_syn  [mmag]
Frames: screened (260927_std_offset_summary.screened); SPSS std SNR >= 100, CALSPEC std SNR > 10 with the faint-star
bias removed (as in section 9.4); per star and filter the median over frames of all units.
Figures (~/ezphot/log/20260930_std_mb_sed/):
  Q1_sed_<dataset>_<k>.png : per star, top = reference spectrum (AB mag) + observed (filled) and synthetic (open)
                             medium-band points for each curve; bottom = residual per filter for each curve
  Q2_residual_summary.png  : over all stars, median residual +- scatter (1.4826 MAD) per filter and curve
Tables: mb_sed_long.csv (star x filter x curve), residual_summary.csv
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
SF = importlib.import_module("260930_std_shift_fit")
CT = SF.CT
R, V2, AN, D = SF.R, SF.V2, SF.AN, SF.D
OUT = Path("/home/hhchoi1022/ezphot/log/20260930_std_mb_sed")
CURVES = [("manufacturer", "beam"), ("manufacturer", "nobeam"), ("version_mhj", "beam"), ("version_mhj", "nobeam"),
          ("version_hyp", "beam"), ("version_hyp", "nobeam"), ("version_ky", "beam"), ("version_ky", "nobeam")]
LAB = {"manufacturer": "manuf", "version_mhj": "mhj", "version_hyp": "hyp", "version_ky": "ky"}
STYLE = {"manuf·beam": ("o", "k"), "manuf·nobeam": ("o", "0.6"), "mhj·beam": ("s", "tab:purple"), "mhj·nobeam": ("s", "plum"),
         "hyp·beam": ("^", "tab:green"), "hyp·nobeam": ("^", "yellowgreen"), "ky·beam": ("D", "tab:red"), "ky·nobeam": ("D", "salmon")}
C_NM = 2.99792458e17                      # nm / s
FNU_AB = 3.631e-23                        # W m-2 Hz-1


def ab_spectrum(w, f):
    """AB magnitude per wavelength of a spectrum in W m-2 nm-1."""
    fnu = f * w ** 2 / C_NM
    return -2.5 * np.log10(np.clip(fnu, 1e-40, None) / FNU_AB)


def pivot(fw, ft):
    return float(np.sqrt(np.trapz(fw * ft, fw) / np.trapz(ft / fw, fw)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tot = V2.build_totals()
    F = SS.screened(pd.read_csv(D / "std_offset_frames.csv"))
    bias = SS.expected_bias(F.std_peak_e.to_numpy(), "CALSPEC") / 1e3
    F["d"] = np.where(F.dataset == "SPSS", np.where(F.std_snr >= 100, F.dl_gnd, np.nan),
                      np.where(F.std_snr > 10, F.dl_gnd - bias, np.nan)) * 1e3
    F = F[F.d.notna() & (F.d.abs() <= 1e3 * SS.D_MAX)]
    spec = SF.spectra(F[F.dataset == "SPSS"].object.unique(), F[F.dataset == "CALSPEC"].object.unique())
    S = pd.read_csv(D / "std_colours.csv")
    info = {(d, k): (c, t, b) for d, k, c, t, b in zip(S.dataset, S.key, S.cls, S.sptype, S.bprp)}

    P = F.groupby(["dataset", "object", "filter", "version", "variant"]).agg(d=("d", "median"), n=("d", "size")).reset_index()
    rows = []
    for r in P.itertuples(index=False):
        if (r.dataset, r.object) not in spec or (r.version, r.variant) not in tot or r.filter not in tot[(r.version, r.variant)]:
            continue
        w, f = spec[(r.dataset, r.object)]
        fw, ft = tot[(r.version, r.variant)][r.filter]
        sup = fw[ft > 1e-3 * ft.max()]
        if w[0] > sup.min() - 1 or w[-1] < sup.max() + 1:
            continue
        m_syn = float(R.syn_mag(w, f, fw, ft))
        key = (r.dataset, r.object if r.dataset == "CALSPEC" else AN.norm(r.object))
        cls, spt, bp = info.get(key, ("?", "?", np.nan))
        rows.append(dict(dataset=r.dataset, object=r.object, cls=cls, sptype=spt, bprp=bp, filter=r.filter,
                         curve=f"{LAB[r.version]}·{r.variant}", lam=round(pivot(fw, ft), 2), m_syn=round(m_syn, 4),
                         m_obs=round(m_syn + r.d / 1e3, 4), resid=round(r.d, 2), n_frames=r.n))
    L = pd.DataFrame(rows)
    L.to_csv(OUT / "mb_sed_long.csv", index=False)

    # Q1: per-star SED + residual, grids of 12 stars per figure
    order = L.drop_duplicates(["dataset", "object"]).sort_values(["dataset", "bprp"])
    curves = list(STYLE)
    for ds, gds in order.groupby("dataset"):
        stars = list(gds.object)
        for k in range(0, len(stars), 12):
            chunk = stars[k:k + 12]
            nc = 4; nr = int(np.ceil(len(chunk) / nc))
            fig = plt.figure(figsize=(5.2 * nc, 6.0 * nr))
            outer = fig.add_gridspec(nr, nc, hspace=0.32, wspace=0.28)
            for i, o in enumerate(chunk):
                rr, cc = divmod(i, nc)
                inner = outer[rr, cc].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.06)
                ax = fig.add_subplot(inner[0]); axr = fig.add_subplot(inner[1], sharex=ax)
                g = L[(L.dataset == ds) & (L.object == o)]
                w, f = spec[(ds, o)]
                m = (w > 360) & (w < 900)
                ax.plot(w[m], ab_spectrum(w[m], f[m]), "-", color="0.75", lw=0.7, zorder=1)
                for j, cv in enumerate(curves):
                    q = g[g.curve == cv].sort_values("lam")
                    if q.empty:
                        continue
                    mk, c = STYLE[cv]
                    dx = (j - 3.5) * 0.6
                    ax.plot(q.lam + dx, q.m_obs, mk, color=c, ms=3.2, zorder=3)
                    ax.plot(q.lam + dx, q.m_syn, mk, mfc="none", mec=c, ms=4.2, mew=0.7, zorder=2)
                    axr.plot(q.lam + dx, q.resid, mk + "-", color=c, ms=2.8, lw=0.7, label=cv)
                cls, spt = g.cls.iloc[0], g.sptype.iloc[0]
                ax.invert_yaxis(); ax.grid(alpha=0.3); ax.tick_params(labelsize=7, labelbottom=False)
                ax.set_title(f"{o} ({spt}, BP−RP {g.bprp.iloc[0]:.2f})", fontsize=8, loc="left")
                ax.set_ylabel("AB mag", fontsize=7)
                axr.axhline(0, color="k", lw=0.5); axr.grid(alpha=0.3); axr.tick_params(labelsize=7)
                lo, hi = np.nanpercentile(g.resid, [2, 98]) if len(g) > 3 else (-50, 50)
                pad = max(10, 0.15 * (hi - lo))
                axr.set_ylim(max(-150, lo - pad), min(150, hi + pad))
                axr.set_ylabel("obs − syn [mmag]", fontsize=7)
                axr.set_xlim(360, 900)
                if i == 0:
                    axr.legend(fontsize=5.5, ncol=4, loc="lower left")
            fig.suptitle(f"Q1  {ds} standards: 7DT medium-band SED (filled = observed, MAG_APER_3 + local ZP; open = synthetic photometry "
                         f"of the reference spectrum through the same curve; grey = reference spectrum) and residual obs − syn, "
                         f"for each transmission curve", fontsize=10)
            fig.savefig(OUT / f"Q1_sed_{ds}_{k // 12 + 1}.png", dpi=105, bbox_inches="tight")
            plt.close(fig)

    # Q2: median +- scatter of residuals over all stars, per filter and curve
    mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
    Sm = L.groupby(["dataset", "filter", "curve"]).agg(lam=("lam", "median"), n=("resid", "size"), median=("resid", "median"),
                                                        scatter=("resid", mad), p16=("resid", lambda v: np.percentile(v, 16)),
                                                        p84=("resid", lambda v: np.percentile(v, 84))).reset_index()
    Sm = Sm[Sm.n >= 3]
    Sm.round(2).to_csv(OUT / "residual_summary.csv", index=False)
    fig, axs = plt.subplots(2, 1, figsize=(15, 9.5), sharex=True)
    for ax, ds in zip(axs, ("SPSS", "CALSPEC")):
        t = Sm[Sm.dataset == ds]
        for j, cv in enumerate(curves):
            q = t[t.curve == cv].sort_values("lam")
            if q.empty:
                continue
            mk, c = STYLE[cv]
            ax.errorbar(q.lam + (j - 3.5) * 0.9, q["median"], q.scatter, fmt=mk + "-", color=c, ms=4, lw=0.8, capsize=0, label=cv)
        ax.axhline(0, color="k", lw=0.5); ax.axhspan(-10, 10, color="0.9", zorder=0)
        ax.set_ylim(-60, 60); ax.grid(alpha=0.3)
        nst = L[L.dataset == ds].object.nunique()
        ax.set_ylabel("residual obs − syn [mmag]", fontsize=9)
        ax.set_title(f"{ds} ({nst} stars): median over stars ± scatter (1.4826 MAD) per filter; filters with ≥ 3 stars", fontsize=9, loc="left")
    axs[0].legend(fontsize=7, ncol=8, loc="upper right")
    axs[1].set_xlabel("filter pivot wavelength [nm]")
    fig.suptitle("Q2  Residual of the observed medium-band magnitudes against synthetic photometry of the reference spectra, all standards, "
                 "for each transmission curve (grey band ±10 mmag)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "Q2_residual_summary.png", dpi=115)
    pd.set_option("display.width", 220)
    print(Sm[Sm.dataset == "SPSS"].pivot_table(index="filter", columns="curve", values="median").round(1).to_string())
    print(Sm[Sm.dataset == "SPSS"].pivot_table(index="filter", columns="curve", values="scatter").round(1).to_string())


if __name__ == "__main__":
    main()
