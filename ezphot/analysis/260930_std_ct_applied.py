#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260930_std_ct_applied.py — standard-star offsets after the field colour term is applied, for each transmission curve,
with the 10" aperture (MAG_APER_3), and as aperture checks with the 15" aperture (MAG_APER_4) and MAG_AUTO.

Per frame and curve (260927_std_offset.py columns, all without the standard itself among the field ZP stars): the field
colour term a_x (slope of ZP_i vs BP-RP of the field stars with corrected XP), their median colour bprp0 and the local ZP
at bprp0 (zl_c0, field stars within 1500 px of the standard).  The ZP at the standard's colour is
zl_c0 + a_x (BP-RP_std - bprp0), so
  before  Delta    = MAG(std) + zp_loc_x                            - syn(ground spectrum)   (section 9.12, but there the
                                                                                              standard was in zp_loc)
  after   Delta_ct = MAG(std) + zl_c0 + a_x (BP-RP_std - bprp0)     - syn(ground spectrum)
BP-RP_std = Gaia DR3 BP-RP, as for the field stars (SPSS: std_colours.csv; CALSPEC: GAIA_BEST tile source nearest to the
proper-motion-corrected position, cal_bprp_gaia.csv).  Frames as in 260930_std_mb_sed.py (screened; SPSS std SNR >= 100;
CALSPEC std SNR > 10 minus the faint-star bias measured for that aperture); before and after on the same frames; per star
and filter the median over frames of all units.
15" / MAG_AUTO: 260927_std_offset.py --mag APER_4 / AUTO (same field-star and standard selection, that magnitude for
both), faint-star bias from 260927_std_faintcheck.py 400 APER_4 / AUTO.
Figures (~/ezphot/log/20260930_std_ct_applied/):
  C1_sed_<dataset>_<k>.png : as Q1 (section 9.12) but after the colour term, 10" aperture
  C2_ct_summary.png        : median +- scatter over stars per filter and curve, after the colour term (markers) and
                             before (dotted), for 10", 15" and MAG_AUTO
  C3_aperture.png          : 15" - 10" and MAG_AUTO - 10" on paired frames: per-filter change of the residual, per-frame
                             change vs standard SNR, change of the field colour term
Tables: ct_long.csv (aperture x star x filter x curve), ct_summary.csv
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
MB = importlib.import_module("260930_std_mb_sed")
SS, SF, R, V2, AN, D = MB.SS, MB.SF, MB.R, MB.V2, MB.AN, MB.D
OUT = Path("/home/hhchoi1022/ezphot/log/20260930_std_ct_applied")
APS = {"10\"": ("std_offset_frames.csv", ""), "15\"": ("std_offset_frames_aper_4.csv", "_aper_4"),
       "AUTO": ("std_offset_frames_auto.csv", "_auto")}
APLAB = {"10\"": "MAG_APER_3 (10″)", "15\"": "MAG_APER_4 (15″)", "AUTO": "MAG_AUTO"}
REF = "manuf·beam"
mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))   # noqa: E731


def std_colours():
    """(dataset, key) -> Gaia BP-RP of the standard: SPSS from std_colours.csv (Gaia), CALSPEC from the GAIA_BEST tiles."""
    S = pd.read_csv(D / "std_colours.csv")
    col = {(d, k): b for d, k, b in zip(S.dataset, S.key, S.bprp) if d == "SPSS"}
    fc = OUT / "cal_bprp_gaia.csv"
    if not fc.exists():
        TW = SF.TW
        AN._init(); TW._init(40)
        rows = []
        for o in S[S.dataset == "CALSPEC"].key:
            st0 = TW._INFO[o]
            ref = AN.gather_reference(st0["ra"], st0["dec"], radius=0.05)
            ra = st0["ra"] + st0["pmra"] * 16.0 / 3.6e6 / np.cos(np.deg2rad(st0["dec"]))      # J2000 -> Gaia epoch 2016
            de = st0["dec"] + st0["pmdec"] * 16.0 / 3.6e6
            sep = np.hypot((np.asarray(ref["ra"]) - ra) * np.cos(np.deg2rad(de)), np.asarray(ref["dec"]) - de) * 3600
            j = int(np.argmin(sep))
            rows.append(dict(object=o, bprp_gaia=float(ref["bprp"][j]), sep_arcsec=round(float(sep[j]), 3)))
        pd.DataFrame(rows).to_csv(fc, index=False)
    C = pd.read_csv(fc)
    col.update({("CALSPEC", o): b for o, b, s in zip(C.object, C.bprp_gaia, C.sep_arcsec) if s < 0.5})
    return col


def load_frames(fname, sfx, colour):
    F = SS.screened(pd.read_csv(D / fname))
    key = np.where(F.dataset == "CALSPEC", F.object, F.object.map(AN.norm))
    F["bprp_std"] = [colour.get(k, np.nan) for k in zip(F.dataset, key)]
    bias = np.where(F.dataset == "CALSPEC", SS.expected_bias(F.std_peak_e.to_numpy(), "CALSPEC", sfx) / 1e3, 0.0)
    good = np.where(F.dataset == "SPSS", F.std_snr >= 100, F.std_snr > 10)
    F["d"] = (F.dl_gnd - F.zp_loc + F.zp_loc_x - bias) * 1e3
    F["d_ct"] = (F.dl_gnd - F.zp_loc + F.zl_c0 + F.a_x * (F.bprp_std - F.bprp0) - bias) * 1e3
    F = F[good & F.d.notna() & F.d_ct.notna()]
    return F[(F.d.abs() <= 1e3 * SS.D_MAX) & (F.d_ct.abs() <= 1e3 * SS.D_MAX)]


def plot_sed(L, spec, ds, stars, fname):
    curves = list(MB.STYLE)
    nc = 4; nr = int(np.ceil(len(stars) / nc))
    fig = plt.figure(figsize=(5.2 * nc, 6.0 * nr))
    outer = fig.add_gridspec(nr, nc, hspace=0.32, wspace=0.28)
    for i, o in enumerate(stars):
        rr, cc = divmod(i, nc)
        inner = outer[rr, cc].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.06)
        ax = fig.add_subplot(inner[0]); axr = fig.add_subplot(inner[1], sharex=ax)
        g = L[(L.dataset == ds) & (L.object == o)]
        w, f = spec[(ds, o)]
        m = (w > 360) & (w < 900)
        ax.plot(w[m], MB.ab_spectrum(w[m], f[m]), "-", color="0.75", lw=0.7, zorder=1)
        for j, cv in enumerate(curves):
            q = g[g.curve == cv].sort_values("lam")
            if q.empty:
                continue
            mk, c = MB.STYLE[cv]
            dx = (j - 3.5) * 0.6
            ax.plot(q.lam + dx, q.m_obs, mk, color=c, ms=3.2, zorder=3)
            ax.plot(q.lam + dx, q.m_syn, mk, mfc="none", mec=c, ms=4.2, mew=0.7, zorder=2)
            axr.plot(q.lam + dx, q.resid, mk + "-", color=c, ms=2.8, lw=0.7, label=cv)
        ax.invert_yaxis(); ax.grid(alpha=0.3); ax.tick_params(labelsize=7, labelbottom=False)
        ax.set_title(f"{o} ({g.sptype.iloc[0]}, BP−RP {g.bprp.iloc[0]:.2f})", fontsize=8, loc="left")
        ax.set_ylabel("AB mag", fontsize=7)
        axr.axhline(0, color="k", lw=0.5); axr.grid(alpha=0.3); axr.tick_params(labelsize=7)
        lo, hi = np.nanpercentile(g.resid, [2, 98]) if len(g) > 3 else (-50, 50)
        pad = max(10, 0.15 * (hi - lo))
        axr.set_ylim(max(-150, lo - pad), min(150, hi + pad))
        axr.set_ylabel("obs − syn [mmag]", fontsize=7)
        axr.set_xlim(360, 900)
        if i == 0:
            axr.legend(fontsize=5.5, ncol=4, loc="lower left")
    fig.suptitle(f"C1  {ds} standards after the field colour term: filled = observed (MAG_APER_3 + local ZP at the star's "
                 f"BP−RP), open = synthetic photometry of the reference spectrum through the same curve, grey = reference "
                 f"spectrum; bottom = residual obs − syn, for each transmission curve", fontsize=10)
    fig.savefig(OUT / fname, dpi=105, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tot = V2.build_totals()
    S = pd.read_csv(D / "std_colours.csv")
    colour = std_colours()
    info = {(d, k): (c, t, colour.get((d, k), np.nan)) for d, k, c, t in zip(S.dataset, S.key, S.cls, S.sptype)}
    FR = {ap: load_frames(fn, sfx, colour) for ap, (fn, sfx) in APS.items()}
    objs = pd.concat(FR.values())
    spec = SF.spectra(objs[objs.dataset == "SPSS"].object.unique(), objs[objs.dataset == "CALSPEC"].object.unique())

    rows, syn = [], {}
    for ap, F in FR.items():
        P = F.groupby(["dataset", "object", "filter", "version", "variant"]).agg(
            d=("d", "median"), d_ct=("d_ct", "median"), a=("a_x", "median"), n=("d", "size")).reset_index()
        for r in P.itertuples(index=False):
            k = (r.dataset, r.object, r.filter, r.version, r.variant)
            if k not in syn:
                syn[k] = None
                if (r.dataset, r.object) in spec and (r.version, r.variant) in tot and r.filter in tot[(r.version, r.variant)]:
                    w, f = spec[(r.dataset, r.object)]
                    fw, ft = tot[(r.version, r.variant)][r.filter]
                    sup = fw[ft > 1e-3 * ft.max()]
                    if not (w[0] > sup.min() - 1 or w[-1] < sup.max() + 1):
                        syn[k] = (float(R.syn_mag(w, f, fw, ft)), MB.pivot(fw, ft))
            if syn[k] is None:
                continue
            m_syn, lam = syn[k]
            cls, spt, bp = info.get((r.dataset, r.object if r.dataset == "CALSPEC" else AN.norm(r.object)), ("?", "?", np.nan))
            rows.append(dict(aperture=ap, dataset=r.dataset, object=r.object, cls=cls, sptype=spt, bprp=bp, filter=r.filter,
                             curve=f"{MB.LAB[r.version]}·{r.variant}", lam=round(lam, 2), m_syn=round(m_syn, 4),
                             m_obs=round(m_syn + r.d_ct / 1e3, 4), resid=round(r.d_ct, 2), resid_before=round(r.d, 2),
                             a_field=round(r.a, 4), n_frames=r.n))
    L = pd.DataFrame(rows)
    L.to_csv(OUT / "ct_long.csv", index=False)

    # C1: per-star SED after the colour term, 10" aperture
    L10 = L[L.aperture == "10\""]
    order = L10.drop_duplicates(["dataset", "object"]).sort_values(["dataset", "bprp"])
    for ds, gds in order.groupby("dataset"):
        stars = list(gds.object)
        for k in range(0, len(stars), 12):
            plot_sed(L10, spec, ds, stars[k:k + 12], f"C1_sed_{ds}_{k // 12 + 1}.png")

    # summary over stars
    Sm = L.groupby(["aperture", "dataset", "filter", "curve"]).agg(
        lam=("lam", "median"), n=("resid", "size"), median=("resid", "median"), scatter=("resid", mad),
        median_before=("resid_before", "median"), scatter_before=("resid_before", mad)).reset_index()
    Sm = Sm[Sm.n >= 3]
    Sm.round(2).to_csv(OUT / "ct_summary.csv", index=False)

    # C2: after (markers) and before (dotted) the colour term, per aperture
    curves = list(MB.STYLE)
    fig, axs = plt.subplots(2, len(APS), figsize=(7.2 * len(APS), 9.5), sharex=True, sharey=True)
    for i, ds in enumerate(("SPSS", "CALSPEC")):
        for jj, ap in enumerate(APS):
            ax = axs[i, jj]; t = Sm[(Sm.dataset == ds) & (Sm.aperture == ap)]
            for j, cv in enumerate(curves):
                q = t[t.curve == cv].sort_values("lam")
                if q.empty:
                    continue
                mk, c = MB.STYLE[cv]
                x = q.lam + (j - 3.5) * 0.9
                ax.plot(x, q.median_before, ":", color=c, lw=0.9, alpha=0.7)
                ax.errorbar(x, q["median"], q.scatter, fmt=mk + "-", color=c, ms=3.5, lw=0.8, capsize=0, label=cv)
            ax.axhline(0, color="k", lw=0.5); ax.axhspan(-10, 10, color="0.9", zorder=0)
            ax.set_ylim(-60, 60); ax.grid(alpha=0.3)
            nst = L[(L.dataset == ds) & (L.aperture == ap)].object.nunique()
            ax.set_title(f"{ds} ({nst} stars), {APLAB[ap]}", fontsize=9, loc="left")
            if jj == 0:
                ax.set_ylabel("residual obs − syn [mmag]", fontsize=9)
    axs[0, 0].legend(fontsize=6.5, ncol=4, loc="upper right")
    for ax in axs[1]:
        ax.set_xlabel("filter pivot wavelength [nm]")
    fig.suptitle("C2  Standard-star residual after the field colour term is applied at each star's BP−RP (markers: median ± "
                 "1.4826 MAD over stars) and before (dotted), per filter and transmission curve (the curves that exist for "
                 "that filter); filters with ≥ 3 stars; grey band ±10 mmag", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "C2_ct_summary.png", dpi=105)
    plt.close(fig)

    # C3: aperture diagnostics against 10" (reference curve; the aperture change hardly depends on the curve)
    pair = ["file", "version", "variant"]
    A0 = pd.read_csv(D / APS["10\""][0])
    others = [ap for ap in APS if ap != "10\""]
    fig, axs = plt.subplots(len(others), 3, figsize=(20, 5.4 * len(others)))
    Wall, PFall = {}, {}
    lam_of = L.drop_duplicates(["filter", "curve"]).set_index(["filter", "curve"]).lam
    for ii, ap in enumerate(others):
        cols = pair + ["dataset", "object", "filter", "d", "d_ct"]
        Q = FR["10\""][cols].merge(FR[ap][pair + ["d", "d_ct"]], on=pair, suffixes=("", "_x"))
        Q = Q[(Q.version == "manufacturer") & (Q.variant == "beam")]
        W = Q.assign(fb=Q.d_x - Q.d, fa=Q.d_ct_x - Q.d_ct).groupby(["dataset", "object", "filter"]).agg(
            dd_before=("fb", "median"), dd_after=("fa", "median")).reset_index()
        W["lam"] = [lam_of.get((f, REF), np.nan) for f in W["filter"]]
        A1 = pd.read_csv(D / APS[ap][0])
        PF = A0.merge(A1[pair + ["dl_gnd", "zp_loc", "zp_loc_x", "a_x"]], on=pair, suffixes=("", "_o"))
        PF = PF[(PF.version == "manufacturer") & (PF.variant == "beam")]
        PF = PF.loc[SS.screened(PF).index]
        PF["dd"] = 1e3 * ((PF.dl_gnd_o - PF.zp_loc_o + PF.zp_loc_x_o) - (PF.dl_gnd - PF.zp_loc + PF.zp_loc_x))
        PF["da"] = 1e3 * (PF.a_x_o - PF.a_x)
        Wall[ap], PFall[ap] = W, PF
        for ds, c, off in (("SPSS", "tab:blue", -1.5), ("CALSPEC", "tab:orange", 1.5)):
            g = W[W.dataset == ds].groupby("filter").agg(lam=("lam", "median"), b=("dd_before", "median"),
                                                        bs=("dd_before", mad), a=("dd_after", "median"),
                                                        n=("dd_before", "size")).reset_index()
            g = g[g.n >= 3].sort_values("lam")
            axs[ii, 0].errorbar(g.lam + off, g.b, g.bs, fmt="o-", color=c, ms=4, lw=0.8, capsize=0, label=f"{ds}: before the colour term")
            axs[ii, 0].plot(g.lam + off, g.a, "s--", mfc="none", color=c, ms=5, lw=0.8, label=f"{ds}: after the colour term")
            p = PF[(PF.dataset == ds) & PF.dd.notna()]
            axs[ii, 1].plot(p.std_snr, p.dd, ".", color=c, ms=2, alpha=0.3, label=ds)
            bins = np.logspace(1, 4, 16); ix = np.digitize(p.std_snr, bins)
            md = [(np.sqrt(bins[k - 1] * bins[k]), np.median(p.dd[ix == k])) for k in range(1, len(bins)) if (ix == k).sum() >= 10]
            if md:
                axs[ii, 1].plot(*zip(*md), "-", color=c, lw=2)
            h = p.groupby("filter").agg(lam=("filter", lambda v: int(v.iloc[0][1:4])), m=("da", "median"),
                                        lo=("da", lambda v: np.percentile(v, 16)), hi=("da", lambda v: np.percentile(v, 84)),
                                        n=("da", "size")).reset_index()
            h = h[h.n >= 10].sort_values("lam")
            axs[ii, 2].errorbar(h.lam + off, h.m, [h.m - h.lo, h.hi - h.m], fmt="o", color=c, ms=4, capsize=0, label=ds)
        axs[ii, 0].set_title(f"({'ab'[ii]}1) residual change {APLAB[ap]} − 10″ [mmag]: per star the median over paired frames,\n"
                             f"then median ± scatter over stars ({REF})", fontsize=9, loc="left")
        axs[ii, 0].set_xlabel("filter pivot wavelength [nm]"); axs[ii, 0].set_ylim(-30, 30); axs[ii, 0].legend(fontsize=7)
        axs[ii, 1].set_xscale("log"); axs[ii, 1].set_ylim(-60, 60); axs[ii, 1].set_xlabel("standard SNR (10″)")
        axs[ii, 1].set_title(f"({'ab'[ii]}2) per frame: Δ({APLAB[ap]}) − Δ(10″) [mmag] = ({ap} − 10″) of the standard\nminus "
                             f"that of the local field stars; lines = running median", fontsize=9, loc="left")
        axs[ii, 1].legend(fontsize=7, markerscale=5)
        axs[ii, 2].set_xlabel("filter pivot wavelength [nm]"); axs[ii, 2].set_ylim(-30, 30); axs[ii, 2].legend(fontsize=7)
        axs[ii, 2].set_title(f"({'ab'[ii]}3) field colour term change, a({APLAB[ap]}) − a(10″)\n[mmag per mag BP−RP], "
                             f"median and 16–84% over frames", fontsize=9, loc="left")
    for ax in axs.ravel():
        ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3)
    fig.suptitle("C3  Does the aperture matter? 15″ aperture and MAG_AUTO against the 10″ aperture, same stars and frames",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "C3_aperture.png", dpi=105)
    plt.close(fig)

    pd.set_option("display.width", 250)
    for ds in ("SPSS", "CALSPEC"):
        for ap in APS:
            t = Sm[(Sm.dataset == ds) & (Sm.aperture == ap)]
            print(f"\n== {ds} {ap}: median after | before")
            print(t.pivot_table(index="filter", columns="curve", values="median").round(1).to_string())
            print(t.pivot_table(index="filter", columns="curve", values="median_before").round(1).to_string())
            sp = t.groupby("filter").agg(lam=("lam", "median"), after=("median", np.ptp), before=("median_before", np.ptp))
            for lab, m in (("all", sp.lam > 0), ("<= m466w", sp.lam < 470), (">= m475", sp.lam >= 470)):
                print(f"spread across the curves of each filter (max-min), median over filters [{lab}]: "
                      f"after {sp.after[m].median():.1f} before {sp.before[m].median():.1f}")
            for cv in curves:
                q = t[t.curve == cv]
                print(f"  {cv:13s} |median| med after {q['median'].abs().median():5.1f} before {q.median_before.abs().median():5.1f}"
                      f"  scatter med after {q.scatter.median():5.1f} before {q.scatter_before.median():5.1f}")
            c = L[(L.dataset == ds) & (L.aperture == ap) & (L.curve == REF)]
            print("  by class (manuf·beam) median after / before:",
                  c.groupby("cls").agg(after=("resid", "median"), before=("resid_before", "median"), n=("object", "nunique")).round(1).to_dict("index"))
    # hot (BP-RP < 0.2) minus FGK standards per curve, blue filters (level-free)
    H = L[(L.aperture == "10\"") & (L.dataset == "SPSS")].copy()
    H["grp"] = np.where(H.bprp < 0.2, "hot", np.where(H.cls == "FGK", "FGK", "other"))
    H = H[H.grp != "other"].groupby(["filter", "curve", "grp"]).agg(b=("resid_before", "median"), a=("resid", "median"),
                                                                    af=("a_field", "median"), n=("object", "size")).unstack("grp")
    H = pd.DataFrame({"n_hot": H[("n", "hot")], "a_field": 1e3 * H[("af", "hot")],
                      "hot-FGK before": H[("b", "hot")] - H[("b", "FGK")], "hot-FGK after": H[("a", "hot")] - H[("a", "FGK")]})
    print("\nSPSS 10\" hot (BP-RP<0.2) - FGK per curve, blue filters [mmag; a_field in mmag/mag]:")
    print(H[H.index.get_level_values(0).isin(["m400", "m425", "m438", "m450", "m466w", "m475", "m483"])].round(1).to_string())
    # effect of removing the standard from its own field ZP stars on the before value (sections 9.2-9.12 kept it)
    E = pd.read_csv(D / APS["10\""][0])
    E = E.loc[SS.screened(E).index]
    E = E[(E.version == "manufacturer") & (E.variant == "beam") & np.where(E.dataset == "SPSS", E.std_snr >= 100, E.std_snr > 10)]
    E["dz"] = 1e3 * (E.zp_loc_x - E.zp_loc); E["da"] = 1e3 * (E.a_x - E.a_field)
    print("\nstandard removed from its own ZP stars (10\", manuf·beam): fraction of frames where it was a ZP star, "
          "median / 5-95% of zp_loc_x - zp_loc [mmag] and a_x - a_field [mmag/mag]:")
    print(E.groupby("dataset").agg(frac=("n_std_zp", lambda v: (v > 0).mean()), dz=("dz", "median"),
                                   dz5=("dz", lambda v: np.nanpercentile(v, 5)), dz95=("dz", lambda v: np.nanpercentile(v, 95)),
                                   da=("da", "median"), da5=("da", lambda v: np.nanpercentile(v, 5)),
                                   da95=("da", lambda v: np.nanpercentile(v, 95))).round(2).to_string())
    for ap in others:
        W, PF = Wall[ap], PFall[ap]
        print(f"\n{ap} - 10\" per-star residual change ({REF}): median / scatter before, median after")
        print(W.groupby("dataset").agg(b=("dd_before", "median"), bs=("dd_before", mad), a=("dd_after", "median")).round(1).to_string())
        print(W.groupby(["dataset", "filter"]).dd_before.median().unstack(0).round(1).to_string())
        print(f"field colour term change {ap}-10\" [mmag/mag] by filter:")
        print(PF.groupby(["dataset", "filter"]).da.median().unstack(0).round(1).to_string())


if __name__ == "__main__":
    main()
