#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260930_std_sensitivity.py — which CALSPEC / SPSS standards react most to a small shift of a filter curve.

For every standard spectrum and every medium-band filter (manufacturer curve + f/3.03 beam, x QE x optics x El Sauce
atmosphere, vacuum; 260926_truthcompare_v2.build_totals) the synthetic magnitude is computed with the curve shifted by
+-0.5 nm:  S = dm/d(delta) [mmag/nm] (m(+0.5) - m(-0.5)).
A common shift of all stars is absorbed by the field ZP, which is set by field stars (corrected Gaia XP).  What a
standard can measure is therefore its sensitivity RELATIVE to the field stars:
  S_rel = S - S_field,   S_field = S of GAIA_BEST corrected-XP field stars at BP-RP = 0.85 (the median colour of the
SNR >= 100 field ZP stars), from a straight-line fit of S against BP-RP over 0.5 <= BP-RP <= 1.3, around 12 SPSS fields.
A standard's offset Delta = m_inst + ZP - syn moves by S_rel x delta, delta = true - adopted curve centre (nm).
Filters whose curve support (T > 1e-3 of peak) is not covered by a spectrum are left out for that star.

Spectra:
  SPSS  : the 111 Gaia SPSS V2 flux tables (260919_spss_analysis.load_spss_spec) with name, coordinates, V, SpType,
          quality Flag from the SPSS table.
  CALSPEC: /qso/data3/doabdi/intern/synthetic/calspec_data, one spectrum per star from the CALSPEC table calV.csv
          (STIS version if listed, else the model), coordinates from calspecTable.csv.
Flags: observable = Dec <= +25 deg (El Sauce, airmass <= ~1.9 at transit); usable brightness 9.5 <= V <= 15.
Output (~/ezphot/log/20260930_std_sensitivity/): sensitivity_long.csv (star x filter), star_summary.csv,
top_per_filter.csv, O10_std_sensitivity_map.png, O11_std_sensitivity_spectra.png
"""
import importlib
import re
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
CT = importlib.import_module("260927_std_colorterm")
AN, R, V2 = CT.AN, CT.R, CT.V2
OUT = Path("/home/hhchoi1022/ezphot/log/20260930_std_sensitivity")
CALD = Path("/qso/data3/doabdi/intern/synthetic")
XP_WAVE = np.arange(336.0, 1020.01, 2.0)
DEC_MAX, V_LO, V_HI = 25.0, 9.5, 15.0
LINES = {"Hα": 656.3, "Hβ": 486.1, "Hγ": 434.0, "Hδ": 410.2, "Ca H&K": 395.0, "Mg b": 517.5, "Na D": 589.3,
         "TiO": 710.0, "O2 A": 760.5}


def sens(w, f, curves, filters):
    """S per filter; NaN where the spectrum does not cover the curve support (T > 1e-3 of peak, +-1 nm)."""
    out = {}
    for fl in filters:
        fw, ft = curves[fl]
        sup = fw[ft > 1e-3 * ft.max()]
        if w[0] > sup.min() - 1.0 or w[-1] < sup.max() + 1.0:
            out[fl] = np.nan
            continue
        mp = R.syn_mag(w, f, fw, ft, 0.5)
        mm = R.syn_mag(w, f, fw, ft, -0.5)
        out[fl] = 1e3 * (mp - mm)
    return out


def sexa(ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    return c.ra.deg, c.dec.deg


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_calspec():
    V = pd.read_csv(CALD / "calV.csv", encoding="utf-8-sig", skiprows=[1])
    V = V.rename(columns={"Star name": "star", "Sp. T.": "sptype", "STIS**": "stis"})
    V = V[V.star.notna() & V.Name.notna()]
    T = pd.read_csv(CALD / "calspecTable.csv", encoding="utf-8-sig", skiprows=[1])
    T = T[T["Star Name"].notna() & T["R.A."].notna()]
    pos = {norm(n): (r, d) for n, r, d in zip(T["Star Name"], T["R.A."], T["Decl."])}
    rows = []
    for _, v in V.iterrows():
        base = str(v.Name).strip()
        cand = []
        for col in ("stis", "Model"):
            tag = str(v.get(col, "")).strip().replace("*", "")
            if tag and tag != "nan":
                cand.append(CALD / "calspec_data" / f"{base}{tag}.fits")
        cand += sorted((CALD / "calspec_data").glob(f"{base}_stis_*.fits"))[-1:] + sorted((CALD / "calspec_data").glob(f"{base}_mod_*.fits"))[-1:]
        p = next((c for c in cand if c.exists()), None)
        if p is None or norm(v.star) not in pos:
            continue
        ra, dec = sexa(*pos[norm(v.star)])
        try:
            vmag = float(v.V)
        except (TypeError, ValueError):
            vmag = np.nan
        rows.append(dict(dataset="CALSPEC", name=str(v.star).strip(), sptype=str(v.sptype), V=vmag, ra=ra, dec=dec,
                         flag=np.nan, spec=p.name))
    return pd.DataFrame(rows)


def read_calspec(fn):
    from astropy.table import Table
    t = Table.read(CALD / "calspec_data" / fn)
    wl = np.asarray(t["WAVELENGTH"], float) / 10.0
    fl = np.asarray(t["FLUX"], float) * 1e-2
    m = (wl > 300) & (wl < 1100) & np.isfinite(fl) & (fl > 0)
    return wl[m], fl[m]


def load_spss():
    sp = pd.read_csv(AN.SPSS_CSV)
    sp = sp[sp.inV2 == "yes"]
    return pd.DataFrame(dict(dataset="SPSS", name=sp.Name.values, sptype=sp.SpType.astype(str).values, V=sp.Vmag.values,
                             ra=sp.RA_ICRS.values, dec=sp.DE_ICRS.values, flag=sp.Flag.values, spec=sp.SPSS.values))


def field_reference(curves, filters, sp):
    AN._init()
    ss, cols = [], []
    for _, s in sp[(sp.dec <= DEC_MAX)].sample(12, random_state=1).iterrows():
        ref = AN.gather_reference(s.ra, s.dec, radius=0.6)
        if ref is None:
            continue
        m = (ref["bprp"] >= 0.5) & (ref["bprp"] <= 1.3)
        if m.sum() < 20:
            continue
        ss.append(np.column_stack([1e3 * (R.syn_mag(XP_WAVE, ref["flux"][m], *curves[f], 0.5)
                                         - R.syn_mag(XP_WAVE, ref["flux"][m], *curves[f], -0.5)) for f in filters]))
        cols.append(ref["bprp"][m])
    S = np.vstack(ss)
    c = np.concatenate(cols)
    print(f"field reference: {len(S)} XP stars from {len(ss)} fields, median BP-RP {np.median(c):.2f}; evaluated at 0.85")
    at, sd = {}, {}
    for j, f in enumerate(filters):
        ok = np.isfinite(S[:, j])
        k = np.polyfit(c[ok], S[ok, j], 1)
        r = S[ok, j] - np.polyval(k, c[ok])
        at[f], sd[f] = float(np.polyval(k, 0.85)), float(1.4826 * np.median(np.abs(r - np.median(r))))
    return at, sd


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves = V2.build_totals()[("manufacturer", "beam")]
    filters = sorted(curves, key=lambda f: int(f[1:4]))
    sp = load_spss()
    cal = load_calspec()
    ref, ref_sd = field_reference(curves, filters, sp)
    rows = []
    for _, s in pd.concat([sp, cal], ignore_index=True).iterrows():
        spec = AN.load_spss_spec(s.spec) if s.dataset == "SPSS" else read_calspec(s.spec)
        if spec is None:
            continue
        w, f = spec
        S = sens(w, f, curves, filters)
        for fl in filters:
            if np.isfinite(S[fl]):
                rows.append(dict(s.drop("spec"), filter=fl, pivot=int(fl[1:4]), S=round(S[fl], 2), S_rel=round(S[fl] - ref[fl], 2)))
    L = pd.DataFrame(rows)
    L["observable"] = L.dec <= DEC_MAX
    L["usable_V"] = L.V.between(V_LO, V_HI)
    L.to_csv(OUT / "sensitivity_long.csv", index=False)
    pd.DataFrame(dict(filter=filters, S_field=[ref[f] for f in filters], S_field_sd=[ref_sd[f] for f in filters])).to_csv(OUT / "field_reference.csv", index=False)

    # per star summary
    g = L.groupby(["dataset", "name"])
    Sm = g.agg(sptype=("sptype", "first"), V=("V", "first"), dec=("dec", "first"), flag=("flag", "first"),
               observable=("observable", "first"), usable_V=("usable_V", "first"),
               n_filters=("S_rel", "size"), rms_S_rel=("S_rel", lambda v: float(np.sqrt(np.mean(v ** 2)))),
               n_ge10=("S_rel", lambda v: int((v.abs() >= 10).sum())), n_ge20=("S_rel", lambda v: int((v.abs() >= 20).sum())))
    idx = L.loc[L.groupby(["dataset", "name"]).S_rel.apply(lambda v: v.abs().idxmax())]
    Sm = Sm.join(idx.set_index(["dataset", "name"])[["filter", "S_rel"]].rename(columns={"filter": "max_filter", "S_rel": "max_S_rel"}))
    Sm = Sm.reset_index().sort_values("rms_S_rel", ascending=False)
    seen = pd.read_csv("/home/hhchoi1022/ezphot/log/20260927_std_newbkg/std_colours.csv")
    obs = {(d, AN.norm(o)) for d, o in zip(seen.dataset, seen.object)}
    Sm["observed_7dt"] = [(d, AN.norm(n)) in obs or (d == "CALSPEC" and AN.norm(n.replace(" ", "_")) in obs) for d, n in zip(Sm.dataset, Sm.name)]
    Sm.to_csv(OUT / "star_summary.csv", index=False)

    # top targets per filter (observable, usable brightness; SPSS flag <= 1)
    ok = L[L.observable & L.usable_V & ~(L.flag >= 2)]
    top = ok.assign(absS=ok.S_rel.abs()).sort_values(["pivot", "absS"], ascending=[True, False]).groupby("filter").head(5)
    top.to_csv(OUT / "top_per_filter.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
    print(pd.DataFrame(dict(S_field=ref, S_field_sd=ref_sd)).T.round(1).to_string())
    print(Sm[Sm.observable & Sm.usable_V].head(30)[["dataset", "name", "sptype", "V", "dec", "flag", "rms_S_rel", "n_ge10", "n_ge20", "max_filter", "max_S_rel", "observed_7dt"]].round(1).to_string(index=False))

    # F1: map of S_rel for the 40 most sensitive observable/usable stars
    sel = Sm[Sm.observable & Sm.usable_V & ~(Sm.flag >= 2)].head(40)
    M = L.set_index(["dataset", "name"]).loc[list(zip(sel.dataset, sel.name))].reset_index().pivot_table(index=["dataset", "name"], columns="pivot", values="S_rel")
    M = M.loc[list(zip(sel.dataset, sel.name))]
    fig, ax = plt.subplots(figsize=(15, 11))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-40, vmax=40, aspect="auto")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"m{c}" for c in M.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([f"{n} ({d[:3]}, {t}, V {v:.1f})" for (d, n), t, v in zip(M.index, sel.sptype, sel.V)], fontsize=7)
    fig.colorbar(im, ax=ax, label="S_rel = d(Delta)/d(shift) relative to field stars [mmag/nm]", shrink=0.6)
    ax.set_title("O10  Sensitivity of standard-star offsets to a curve shift (manufacturer·beam), 40 most sensitive "
                 "observable standards (Dec <= +25, 9.5 <= V <= 15, SPSS Flag <= 1), sorted by rms over filters", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "O10_std_sensitivity_map.png", dpi=120)

    # F2: spectrum shape of representative standards (one per class) and their S_rel per filter
    reps = [("CALSPEC", "BD-11 3759"), ("SPSS", "GJ745A"), ("SPSS", "LTT377"), ("CALSPEC", "HZ4"), ("CALSPEC", "GD71"),
            ("SPSS", "WD2216-657"), ("SPSS", "WD0308-565"), ("SPSS", "G24-25")]
    allst = pd.concat([sp, cal]).drop_duplicates(["dataset", "name"]).set_index(["dataset", "name"])
    fig, axs = plt.subplots(len(reps), 2, figsize=(15, 2.3 * len(reps)), gridspec_kw=dict(width_ratios=[2.2, 1]), sharex="col")
    piv = np.array([int(f[1:4]) for f in filters])
    for k, key in enumerate(reps):
        if key not in allst.index:
            alt = [k2 for k2 in allst.index if k2[1] == key[1]]
            if not alt:
                continue
            key = alt[0]
        row = allst.loc[key]
        w, f = AN.load_spss_spec(row.spec) if key[0] == "SPSS" else read_calspec(row.spec)
        m = (w > 340) & (w < 920)
        fn = f[m] / np.interp(550.0, w, f)
        ax = axs[k, 0]
        ax.semilogy(w[m], fn, "k-", lw=0.7)
        for lab, lw_ in LINES.items():
            ax.axvline(lw_, color="tab:red", lw=0.5, ls=":")
            if k == 0:
                ax.text(lw_, 1.02, lab, transform=ax.get_xaxis_transform(), fontsize=6, ha="center", color="tab:red")
        for x in piv:
            ax.axvline(x, color="0.85", lw=0.5, zorder=0)
        ax.set_ylabel("f_λ / f_λ(550)", fontsize=7)
        ax.set_title(f"{key[1]} ({key[0]}, {row.sptype}, V {row.V:.1f})", fontsize=8, loc="left")
        sr = L[(L.dataset == key[0]) & (L.name == key[1])].set_index("pivot").S_rel.reindex(piv)
        ax2 = axs[k, 1]
        ax2.bar(piv, sr.values, width=6, color=["tab:red" if v > 0 else "tab:blue" for v in sr.fillna(0).values])
        ax2.axhline(0, color="k", lw=0.5); ax2.set_ylim(-45, 45); ax2.grid(alpha=0.3)
        ax2.set_ylabel("S_rel [mmag/nm]", fontsize=7)
    axs[-1, 0].set_xlabel("wavelength [nm]  (grey: filter pivots; red dotted: Balmer, Ca H&K, Mg b, Na D, TiO, O2 A)")
    axs[-1, 1].set_xlabel("filter pivot [nm]")
    fig.suptitle("O11  Spectrum shape (left, normalised at 550 nm) and sensitivity to a curve shift relative to field stars (right) for representative standards", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "O11_std_sensitivity_spectra.png", dpi=120)

if __name__ == "__main__":
    main()
