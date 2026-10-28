#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260927_std_colorterm.py — do CALSPEC / SPSS standards show a colour term (offset vs BP-RP), and does it match the field
colour term?  Uses 260927_std_offset.py frames (re-photometered, local ZP).

Standard offset Delta = MAG_APER_3 + ZP_local - syn(truth) = ZP - (syn - inst).  The ZP is one number set by field stars at
the field median colour, so MINUS the slope of the per-star Delta against the standards' BP-RP is a colour term on the same
convention as the field colour term a_field = d(syn - inst)/d(BP-RP) (corrected XP, same frames, same curve), and with the
ground truth it does not depend on the corrected-XP colour errors.  Equal -> bandpass effect; different -> XP (or a colour
dependence that differs between the standards' and the field stars' colour ranges).
Sign check: a curve bluer than the real one makes red field stars' syn - inst larger (a_field > 0) and red standards'
Delta more negative (Delta slope < 0); moving no-beam -> beam changes a_field by +10 and the Delta slope by -11 (median).

Standards: screens of 260927_std_offset_summary (field colour term within 0.1, FLAGS 0, peak <= 50k, |Delta| <= 0.5);
CALSPEC SNR > 10 with the expected low-signal bias (faintcheck, per frame from the peak in e-) subtracted, SPSS SNR >= 100.
Colours: Gaia BP-RP (GAIA_BEST corrected-XP tables); synthetic colour m450 - m650 from the ground spectrum (manufacturer,
no beam) for every star; BP-RP of stars without Gaia is filled from a linear fit BP-RP(syn colour).
Per (dataset, filter, version, variant): standard colour term ct = -(OLS slope of per-star Delta vs BP-RP), error = bootstrap
over stars (1000); Theil-Sen version; FGK/M-only fit (*_fgkm, field-like stars); white dwarfs vs F-M stars (same sign
convention); median field colour term.
Output (~/ezphot/log/20260927_std_newbkg/): std_colours.csv, std_colorterm_slopes.csv, O5_std_colorterm_blue.png,
O6_std_vs_field_colorterm.png
"""
import importlib
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
SS = importlib.import_module("260927_std_offset_summary")
SO = importlib.import_module("260927_std_offset")
TC, AN, TW, K, R, V2 = SO.TC, SO.AN, SO.TW, SO.K, SO.R, SO.V2
D = Path("/home/hhchoi1022/ezphot/log/20260927_std_newbkg")
FIGD = Path("/home/hhchoi1022/ezphot/log/20260921_curve_versions")
BLUE = ["m375w", "m386", "m400", "m425", "m438", "m450", "m466w", "m475"]
CLS_COL = {"WD": "tab:blue", "sdOB": "tab:cyan", "OBA": "tab:green", "FGK": "tab:orange", "M": "tab:red", "?": "0.5"}


def sp_class(t):
    t = str(t).strip()
    if t.startswith("D") or t.startswith("WD"):
        return "WD"
    if t.startswith("sd"):
        return "sdOB"
    if t[:1] in "OBA":
        return "OBA"
    if t[:1] in "FGK":
        return "FGK"
    if t[:1] == "M":
        return "M"
    return "?"


def standard_colours():
    AN._init(); TW._init(40)
    curves = V2.build_totals()[("manufacturer", "nobeam")]
    from astropy.table import Table
    from astropy.coordinates import SkyCoord, match_coordinates_sky
    rows = []
    sp = pd.read_csv(AN.SPSS_CSV)
    seen = {AN.norm(o) for o in pd.read_csv(D / "std_offset_frames.csv", usecols=["dataset", "object"]).query("dataset == 'SPSS'").object.unique()}
    sp = sp[sp.Name.map(AN.norm).isin(seen)]                  # only the observed SPSS stars
    for _, s in sp.iterrows():
        spec = AN.load_spss_spec(s["SPSS"])
        if spec is None:
            continue
        ref = AN.gather_reference(s["RA_ICRS"], s["DE_ICRS"])
        bp = np.nan
        if ref is not None and pd.notna(s["GaiaEDR3"]):
            j = np.where(ref["sid"] == str(int(s["GaiaEDR3"])))[0]
            bp = float(ref["bprp"][j[0]]) if len(j) else np.nan
        syn = float(R.syn_mag(spec[0], spec[1], *curves["m450"], 0.0) - R.syn_mag(spec[0], spec[1], *curves["m650"], 0.0))
        rows.append(dict(dataset="SPSS", object=s.Name, key=AN.norm(s.Name), sptype=str(s.SpType), bprp_gaia=bp, syn_450_650=syn))
    cal = pd.read_csv("/home/hhchoi1022/ezphot/log/20260908_zp_color_check/calspec_frames.csv",
                      usecols=["object", "spectral_type"]).drop_duplicates("object")
    ctyp = {o: str(t) for o, t in zip(cal.object, cal.spectral_type)}
    ctyp["BD+02_3375"] = "F sdF"                               # SIMBAD: sdF (no type in the frame table)
    for obj, st0 in TW._INFO.items():
        if not st0.get("spec"):
            continue
        t = Table.read(TW.CALSPEC_DIR / st0["spec"])
        wl = np.asarray(t["WAVELENGTH"], float) / 10.0; fl = np.asarray(t["FLUX"], float) * 1e-2
        m = (wl > 300) & (wl < 1100) & np.isfinite(fl) & (fl > 0)
        syn = float(R.syn_mag(wl[m], fl[m], *curves["m450"], 0.0) - R.syn_mag(wl[m], fl[m], *curves["m650"], 0.0))
        bp = np.nan
        got = K.get_reference("GAIAXP_CORR_LAMOST", obj, st0["ra"], st0["dec"], 1.34, 0.90)
        if got is not None and got[1] is not None and len(got[1]):
            tbl, coords, _ = got
            i, d2d, _ = match_coordinates_sky(SkyCoord(st0["ra"], st0["dec"], unit="deg"), coords)
            if d2d.arcsec[0] < 5:
                bp = float(tbl[int(i)]["bp-rp"])
        rows.append(dict(dataset="CALSPEC", object=obj, key=obj, sptype=ctyp.get(obj, "?"), bprp_gaia=bp, syn_450_650=syn))
    S = pd.DataFrame(rows)
    ok = S.bprp_gaia.notna()
    c = np.polyfit(S.syn_450_650[ok], S.bprp_gaia[ok], 1)
    S["bprp"] = np.where(ok, S.bprp_gaia, np.polyval(c, S.syn_450_650))
    S["bprp_src"] = np.where(ok, "gaia", "syn")
    S["cls"] = S.sptype.map(sp_class)
    print(f"BP-RP = {c[0]:.3f} x (m450-m650) + {c[1]:.3f}  (fit on {ok.sum()} stars); filled {(~ok).sum()}")
    return S


def frames():
    F = SS.screened(pd.read_csv(D / "std_offset_frames.csv"))
    F = F[F.std_snr >= np.where(F.dataset == "CALSPEC", 10.0001, 100)].copy()
    bias = np.where(F.dataset == "CALSPEC", SS.expected_bias(F.std_peak_e.to_numpy(), "CALSPEC"), 0.0) / 1e3
    for tag in ("gnd", "cor", "raw"):
        F[f"x_{tag}"] = F[f"dl_{tag}"] - bias
    return F


def fit(per, nboot=1000, seed=0):
    x, y = per.bprp.to_numpy(), per.d.to_numpy()
    if len(x) < 4 or np.ptp(x) < 0.5:
        return np.nan, np.nan
    s = -np.polyfit(x, y, 1)[0]                                  # colour term = -dDelta/d(BP-RP)
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(nboot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0.3:
            b.append(-np.polyfit(x[i], y[i], 1)[0])
    return s, np.std(b)


if __name__ == "__main__":
    S = standard_colours()
    S.to_csv(D / "std_colours.csv", index=False)
    key = dict(zip(zip(S.dataset, S.key), zip(S.bprp, S.cls, S.bprp_src)))
    F = frames()
    F["k"] = [o if d == "CALSPEC" else AN.norm(o) for d, o in zip(F.dataset, F.object)]
    F["bprp"] = [key.get((d, k), (np.nan,))[0] for d, k in zip(F.dataset, F.k)]
    F["cls"] = [key.get((d, k), (np.nan, "?"))[1] for d, k in zip(F.dataset, F.k)]
    F = F[F.bprp.notna()]
    rows = []
    for (ds, f, ver, var), g in F.groupby(["dataset", "filter", "version", "variant"]):
        row = dict(dataset=ds, filter=f, pivot=int(f[1:4]), version=ver, variant=var, n_frames=len(g),
                   a_field=round(1e3 * g.a_field.median(), 1))
        for tag in ("gnd", "cor"):
            per = g.dropna(subset=[f"x_{tag}"]).groupby("object").agg(d=(f"x_{tag}", lambda v: 1e3 * np.median(v)),
                                                                        bprp=("bprp", "first"), cls=("cls", "first"))
            s, e = fit(per)
            row[f"ct_{tag}"], row[f"ct_{tag}_err"], row[f"n_{tag}"] = round(s, 1), round(e, 1), len(per)
            if len(per) >= 5 and np.ptp(per.bprp) >= 0.5:
                from scipy.stats import theilslopes
                ts = theilslopes(per.d, per.bprp)
                row[f"ct_{tag}_theil"], row[f"ct_{tag}_theil_lo"], row[f"ct_{tag}_theil_hi"] = round(-ts[0], 1), round(-ts[3], 1), round(-ts[2], 1)
            ms = per[per.cls.isin(["FGK", "M"])]                     # main-sequence FGK/M only (field-like stars)
            sm, em = fit(ms)
            row[f"ct_{tag}_fgkm"], row[f"ct_{tag}_fgkm_err"], row[f"n_{tag}_fgkm"] = round(sm, 1), round(em, 1), len(ms)
            if tag == "gnd":
                wd, red = per[per.cls.isin(["WD", "sdOB"])], per[per.cls.isin(["FGK", "M"])]
                row["n_wd"], row["n_red"] = len(wd), len(red)
                if len(wd) >= 2 and len(red) >= 2:
                    dc = red.bprp.median() - wd.bprp.median()
                    row["ct_wd_red"] = round(-(red.d.median() - wd.d.median()) / dc, 1)
                row["bprp_range"] = round(np.ptp(per.bprp), 2) if len(per) else np.nan
        rows.append(row)
    T = pd.DataFrame(rows).sort_values(["dataset", "pivot", "version", "variant"])
    T.to_csv(D / "std_colorterm_slopes.csv", index=False)
    pd.set_option("display.width", 250)
    print(T[(T.version == "manufacturer") & (T.variant == "beam")].drop(columns=["version", "variant"]).to_string(index=False))

    # O5: blue filters, Delta (ground) vs BP-RP, manufacturer beam
    G = F[(F.version == "manufacturer") & (F.variant == "beam") & F["filter"].isin(BLUE)]
    panels = [(ds, f) for ds in ("CALSPEC", "SPSS") for f in BLUE if ((G.dataset == ds) & (G["filter"] == f)).any()]
    ncol = 4; nrow = int(np.ceil(len(panels) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(20, 4.3 * nrow), squeeze=False)
    for ax in axs.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (ds, f) in zip(axs.flat, panels):
        g = G[(G.dataset == ds) & (G["filter"] == f)]
        per = g.dropna(subset=["x_gnd"]).groupby("object").agg(d=("x_gnd", lambda v: 1e3 * np.median(v)),
                                                                e=("x_gnd", lambda v: 1e3 * 1.2533 * 1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(len(v))),
                                                                bprp=("bprp", "first"), cls=("cls", "first"))
        for c, gg in per.groupby("cls"):
            ax.errorbar(gg.bprp, gg.d, gg.e, ls="none", marker="o", ms=6, capsize=2, color=CLS_COL.get(c, "0.5"), label=c)
        t = T[(T.dataset == ds) & (T["filter"] == f) & (T.version == "manufacturer") & (T.variant == "beam")].iloc[0]
        xx = np.array([-0.6, 2.8])
        if np.isfinite(t.ct_gnd) and len(per) >= 4:
            c1, c0 = np.polyfit(per.bprp, per.d, 1)
            ax.plot(xx, c0 + c1 * xx, "k-", lw=1.2, label=f"standards (ground): colour term {t.ct_gnd:+.0f} ± {t.ct_gnd_err:.0f}")
            y09 = c0 + c1 * 0.9
        else:
            y09 = per.d.median() if len(per) else 0
        ax.plot(xx, y09 - t.a_field * (xx - 0.9), "--", color="tab:purple", lw=1.4,
                label=f"field colour term {t.a_field:+.0f} (corrected XP), drawn as Δ slope {-t.a_field:+.0f}")
        if np.isfinite(t.ct_cor):
            ax.plot([], [], " ", label=f"standards (corrected XP truth): {t.ct_cor:+.0f} ± {t.ct_cor_err:.0f}")
        ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3)
        ax.set_title(f"{ds}  {f}  ({len(per)} standards)", fontsize=10)
        ax.set_xlabel("standard BP−RP"); ax.set_ylabel("Δ (ground truth) [mmag]")
        ax.legend(fontsize=6.5, loc="best")
    fig.suptitle("O5  Colour dependence of the standard-star offsets in the blue filters (manufacturer·beam; local ZP; CALSPEC SNR > 10 bias-corrected, SPSS SNR ≥ 100)\n"
                 "colour term = −dΔ/d(BP−RP) [mmag/mag], same convention as the field colour term; dashed = what the field colour term predicts for Δ (through the standards' line at BP−RP 0.9)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(D / "O5_std_colorterm_blue.png", dpi=120); shutil.copy2(D / "O5_std_colorterm_blue.png", FIGD)

    # O6: standards slope vs field colour term, all filters, manufacturer beam and nobeam
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, var in zip(axs, ("nobeam", "beam")):
        for ds, mk in (("CALSPEC", "s"), ("SPSS", "o")):
            t = T[(T.dataset == ds) & (T.version == "manufacturer") & (T.variant == var) & T.ct_gnd.notna() & (T.n_gnd >= 5)]
            sc = ax.scatter(t.a_field, t.ct_gnd, c=t["pivot"], cmap="viridis_r", vmin=370, vmax=880, marker=mk, s=45,
                            edgecolor="k", lw=0.4, label=ds, zorder=3)
            ax.errorbar(t.a_field, t.ct_gnd, t.ct_gnd_err, ls="none", color="0.6", lw=0.8, zorder=2)
            for _, r in t[t["pivot"] < 480].iterrows():
                ax.annotate(r["filter"], (r.a_field, r.ct_gnd), fontsize=7, xytext=(3, 3), textcoords="offset points")
        lim = [-80, 160]
        ax.plot(lim, lim, "k--", lw=0.8, label="1:1"); ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set(xlim=lim, ylim=lim, xlabel="field colour term (corrected XP) [mmag/mag]", title=f"manufacturer · {var}")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    axs[0].set_ylabel("standard-star colour term −dΔ/d(BP−RP), ground truth [mmag/mag]")
    fig.colorbar(sc, ax=axs, label="filter pivot [nm]", shrink=0.8)
    fig.suptitle("O6  Standard-star vs field colour term per filter (same sign convention; local ZP; CALSPEC SNR > 10 bias-corrected, SPSS SNR ≥ 100; ≥ 5 standards)", fontsize=11)
    fig.savefig(D / "O6_std_vs_field_colorterm.png", dpi=120, bbox_inches="tight"); shutil.copy2(D / "O6_std_vs_field_colorterm.png", FIGD)
    print("saved O5, O6")
