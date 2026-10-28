#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260927_transmission_paper_fig.py

Paper-ready variants of the 7DT filter-response figure (Fig. 1 candidates).

Data
----
* Response curves actually used for synthetic photometry:
    ~/ezphot/config/common/transmission/m*.csv   (wavelength [Angstrom], response)
  They are identical to Filter x QE x Optics x Atmosphere built from Original_sources
  (checked 2026-09-27: ratio = 1.000 for all bands), so the figure shows T_i(lambda)
  as used in the analysis, nothing re-derived.
* Component curves for the appendix figure: Original_sources/{QE.csv,
  optics.efficiency.dr350.dr500.csv, skytable.fits}.

Design decisions (from reviewer feedback + own review)
------------------------------------------------------
* Only the analysis response T_i(lambda) is drawn; no vendor curves, no component
  envelopes in the main figure (those go to the appendix figure, variant 5).
* Lines (1 pt) carry the information; fills are off or at alpha <= 0.08.
* Figure is designed at final print size (7.0 in = two-column text width) with
  8 pt axis text and 6.5 pt band labels; nothing is meant to be rescaled.
* Same wavelength -> same colour in every panel: colour = f(pivot wavelength),
  fixed domain 380-880 nm, using an equal-lightness version of 'turbo' so the
  yellow/green bands do not vanish on white paper.
* Band names are set in dark-grey ink, never in the band colour (yellow text on
  white is unreadable).
* Shared x (340-900 nm) and y ranges in all panels; x-label only on the bottom panel.
* Out-of-band leaks (< 0.4 %, e.g. m832w at 459 nm) are not drawn; only the
  contiguous main band of each response is plotted.

Outputs -> ~/ezphot/config/common/transmission/fig_variants/*.{pdf,png}

Run:
  python ezphot/analysis/260927_transmission_paper_fig.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from astropy.io import fits  # noqa: E402

BASE = Path("~/ezphot/config/common/transmission").expanduser()
SRC = BASE / "Original_sources"
OUT = BASE / "fig_variants"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- style
INK = "#333333"          # text/ink colour for band names
FULL_W = 7.0             # in, two-column text width (ApJ 7.0 in, MNRAS 508 pt)
COL_W = 3.5              # in, single column
XLIM = (340.0, 900.0)
CMAP_DOMAIN = (380.0, 880.0)   # pivot wavelength -> colour, fixed for all panels

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "axes.labelpad": 3.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,     # embed TrueType (journal requirement)
    "savefig.dpi": 300,
})


# ----------------------------------------------------------------------------- colour
def _srgb_to_lab(rgb):
    rgb = np.asarray(rgb, float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _lab_to_linear(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    d = 6.0 / 29.0

    def finv(t):
        return np.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))

    xyz = np.stack([finv(fx) * 0.95047, finv(fy), finv(fz) * 1.08883], -1)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return xyz @ np.linalg.inv(M).T


def _linear_to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def equal_lightness_spectrum(n=256, L_end=50.0, L_mid=58.0):
    """'turbo' hue sequence with CIELAB lightness pinned to L_end..L_mid so that every
    band colour has similar visual weight on white paper (no vanishing yellow)."""
    t = np.linspace(0, 1, n)
    base = plt.get_cmap("turbo")(0.02 + 0.96 * t)[:, :3]
    lab = _srgb_to_lab(base)
    L_target = L_end + (L_mid - L_end) * np.sin(np.pi * t) ** 2
    out = np.empty_like(base)
    for i in range(n):
        L, a, b = L_target[i], lab[i, 1], lab[i, 2]
        for _ in range(80):                       # shrink chroma until inside sRGB gamut
            lin = _lab_to_linear(np.array([L, a, b]))
            if np.all(lin >= -1e-6) and np.all(lin <= 1 + 1e-6):
                break
            a, b = 0.97 * a, 0.97 * b
        out[i] = _linear_to_srgb(lin)
    return ListedColormap(out, name="turbo_eqL")


CMAP = equal_lightness_spectrum()


def band_colour(pivot_nm):
    t = (pivot_nm - CMAP_DOMAIN[0]) / (CMAP_DOMAIN[1] - CMAP_DOMAIN[0])
    return CMAP(float(np.clip(t, 0, 1)))


# ----------------------------------------------------------------------------- data
def pivot_wavelength(lam, tr):
    return np.sqrt(np.trapezoid(tr * lam, lam) / np.trapezoid(tr / lam, lam))


def main_band(lam, tr, thresh=1e-4):
    """Contiguous region around the peak where tr > thresh, padded with one zero on each side."""
    k = int(np.argmax(tr))
    lo = k
    while lo > 0 and tr[lo - 1] > thresh:
        lo -= 1
    hi = k
    while hi < len(tr) - 1 and tr[hi + 1] > thresh:
        hi += 1
    lo, hi = max(lo - 1, 0), min(hi + 1, len(tr) - 1)
    return lam[lo:hi + 1], tr[lo:hi + 1]


def half_power_edges(lam, tr):
    """Half-maximum crossing wavelengths of the contiguous band around the peak (linear interpolation)."""
    k = int(np.argmax(tr))
    h = tr[k] / 2.0
    lo = k
    while lo > 0 and tr[lo - 1] > h:
        lo -= 1
    hi = k
    while hi < len(tr) - 1 and tr[hi + 1] > h:
        hi += 1

    def cross(i0, i1):
        x0, x1, y0, y1 = lam[i0], lam[i1], tr[i0], tr[i1]
        return x0 + (h - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0

    lo_e = cross(lo - 1, lo) if lo > 0 else lam[lo]
    hi_e = cross(hi, hi + 1) if hi < len(tr) - 1 else lam[hi]
    return float(lo_e), float(hi_e)


class Band:
    def __init__(self, name):
        self.name = name
        d = np.loadtxt(BASE / f"{name}.csv", delimiter=",", skiprows=1)
        lam, tr = d[:, 0] / 10.0, d[:, 1]              # Angstrom -> nm
        self.lam_full, self.tr_full = lam, tr
        self.lam, self.tr = main_band(lam, tr)
        self.pivot = pivot_wavelength(self.lam, self.tr)
        self.peak = float(self.tr.max())
        self.lam_peak = float(self.lam[np.argmax(self.tr)])
        self.wide = name.endswith("w")
        self.on_grid25 = (not self.wide) and int(name[1:]) % 25 == 0
        self.colour = band_colour(self.pivot)
        # vendor (filter-only) half-power edges, for width statements in tags/captions
        v = np.loadtxt(SRC / f"{name}.csv", delimiter=",", skiprows=1)
        o = np.argsort(v[:, 0])                     # m466w is stored descending
        self.fwhm_lo, self.fwhm_hi = half_power_edges(v[o, 0], v[o, 1])
        self.fwhm = self.fwhm_hi - self.fwhm_lo
        self.w_eq = float(np.trapezoid(self.tr, self.lam) / self.peak)   # equivalent width of the response


def load_bands():
    names = sorted([p.stem for p in BASE.glob("m*.csv")])
    bands = [Band(n) for n in names]
    bands.sort(key=lambda b: b.pivot)
    narrow = [b for b in bands if not b.wide]
    wide = [b for b in bands if b.wide]
    return narrow, wide


# ----------------------------------------------------------------------------- layout helpers
def stacked_axes(n, fig_w=FULL_W, panel_h=1.55, left_in=0.55, right_in=0.08,
                 bottom_in=0.40, top_in=0.06, hspace_in=0.10):
    """n vertically stacked panels with exact margins in inches (no tight_layout surprises)."""
    fig_h = bottom_in + top_in + n * panel_h + (n - 1) * hspace_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for i in range(n):
        y0 = bottom_in + (n - 1 - i) * (panel_h + hspace_in)
        ax = fig.add_axes([left_in / fig_w, y0 / fig_h, (fig_w - left_in - right_in) / fig_w, panel_h / fig_h])
        axes.append(ax)
    for ax in axes[1:]:
        ax.sharex(axes[0])
        ax.sharey(axes[0])
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    return fig, axes


def style_axes(ax, ylim, ymajor, yminor, xlabel=True):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_major_locator(MultipleLocator(ymajor))
    ax.yaxis.set_minor_locator(MultipleLocator(yminor))
    if xlabel:
        ax.set_xlabel("Wavelength (nm)")


def draw_bands(ax, bands, lw=1.0, fill_alpha=0.08, normalize=False):
    for b in bands:
        y = b.tr / b.peak if normalize else b.tr
        if fill_alpha > 0:
            ax.fill_between(b.lam, 0, y, color=b.colour, alpha=fill_alpha, lw=0)
        ax.plot(b.lam, y, color=b.colour, lw=lw, solid_joinstyle="round")


def nm_per_pt(ax):
    """Data-units (nm) per typographic point along x, for label-collision estimates."""
    bbox = ax.get_position()
    fig_w_in = ax.figure.get_figwidth()
    return (XLIM[1] - XLIM[0]) / (bbox.width * fig_w_in * 72.0)


def units_per_pt_y(ax, ylim):
    bbox = ax.get_position()
    fig_h_in = ax.figure.get_figheight()
    return (ylim[1] - ylim[0]) / (bbox.height * fig_h_in * 72.0)


def label_bands(ax, bands, ylim, mode="peak", fontsize=6.5, y_base=None, gap_pt=2.5, normalize=False):
    """Greedy label placement: labels are sorted by x; a label that would overlap the last
    label in a row is pushed to the next row. mode 'peak' puts labels above each peak
    (horizontal), 'peak-v' the same rotated 90 deg, 'strip' rotated at a fixed height."""
    nmpp = nm_per_pt(ax)
    upp = units_per_pt_y(ax, ylim)
    rows_end = []
    for b in bands:
        if mode == "peak":
            w_pt = fontsize * 0.56 * len(b.name)
            h_units = fontsize * 1.05 * upp
            rot = 0
        else:
            w_pt = fontsize * 1.15
            h_units = fontsize * 0.56 * len(b.name) * upp
            rot = 90
        w_nm = w_pt * nmpp
        x0, x1 = b.pivot - w_nm / 2, b.pivot + w_nm / 2
        row = None
        for r, end in enumerate(rows_end):
            if x0 > end + gap_pt * nmpp:
                row = r
                break
        if row is None:
            rows_end.append(-np.inf)
            row = len(rows_end) - 1
        rows_end[row] = x1
        if mode == "strip":
            y = y_base + row * (h_units + 3 * upp)
        else:
            ypk = 1.0 if normalize else b.peak
            y = ypk + 2.5 * upp + row * (h_units + 2 * upp)
        ax.text(b.pivot, y, b.name, rotation=rot, ha="center", va="bottom",
                fontsize=fontsize, color=INK, clip_on=False)


def panel_tag(ax, text, loc="upper left", fontsize=8):
    if loc == "upper left":
        ax.text(0.012, 0.95, text, transform=ax.transAxes, ha="left", va="top", fontsize=fontsize)
    else:
        ax.text(0.988, 0.95, text, transform=ax.transAxes, ha="right", va="top", fontsize=fontsize)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"saved {OUT / stem}.pdf/.png")


# ----------------------------------------------------------------------------- variants
def v1_two_panel_clean(narrow, wide, fill_alpha, stem):
    """Reviewer's primary suggestion: two panels, analysis response only, no band names."""
    ylim = (0, 0.78)
    fig, (a, b) = stacked_axes(2, panel_h=1.55)
    draw_bands(a, narrow, fill_alpha=fill_alpha)
    draw_bands(b, wide, fill_alpha=fill_alpha)
    for ax, xl in ((a, False), (b, True)):
        style_axes(ax, ylim, 0.2, 0.05, xlabel=xl)
    panel_tag(a, "(a) 25 nm medium bands (30)")
    panel_tag(b, "(b) 50 nm medium bands (7)")
    fig.supylabel("System throughput", x=0.012, fontsize=8)
    save(fig, stem)


def v2_two_panel_labels(narrow, wide, stem):
    """Two panels but every band named: rotated 6.5 pt ink labels in one/two rows above the curves."""
    ylim = (0, 1.02)
    fig, (a, b) = stacked_axes(2, panel_h=1.75)
    draw_bands(a, narrow, fill_alpha=0.08)
    draw_bands(b, wide, fill_alpha=0.08)
    for ax, xl in ((a, False), (b, True)):
        style_axes(ax, ylim, 0.2, 0.05, xlabel=xl)
    label_bands(a, narrow, ylim, mode="strip", y_base=0.70)
    label_bands(b, wide, ylim, mode="strip", y_base=0.70)
    panel_tag(a, "(a) 25 nm")
    panel_tag(b, "(b) 50 nm")
    fig.supylabel("System throughput", x=0.012, fontsize=8)
    save(fig, stem)


def v3_three_panel(narrow, wide, normalize, fill_alpha, stem):
    """Reviewer's three-panel layout: alternate 25 nm bands in (a)/(b), 50 nm bands in (c);
    horizontal ink labels directly above each peak."""
    set_a, set_b = narrow[0::2], narrow[1::2]
    if normalize:
        ylim, ymaj, ymin = (0, 1.32), 0.5, 0.1
        ylabel = "Normalized response"
    else:
        ylim, ymaj, ymin = (0, 0.82), 0.2, 0.05
        ylabel = "System throughput"
    fig, axes = stacked_axes(3, panel_h=1.42)
    for ax, bands, xl, tag in zip(axes, (set_a, set_b, wide), (False, False, True),
                                  ("(a) 25 nm bands, set 1", "(b) 25 nm bands, set 2", "(c) 50 nm bands")):
        draw_bands(ax, bands, fill_alpha=fill_alpha, normalize=normalize)
        style_axes(ax, ylim, ymaj, ymin, xlabel=xl)
        label_bands(ax, bands, ylim, mode="peak", normalize=normalize)
        panel_tag(ax, tag)
    fig.supylabel(ylabel, x=0.012, fontsize=8)
    save(fig, stem)


def v6_three_groups(narrow, wide, normalize, fill_alpha, stem):
    """Grouping by filter set instead of odd/even: (a) the 20 bands on the 25 nm grid (m400-m875),
    (b) the 10 intermediate bands with individual widths, (c) the 7 w bands. Each tag quotes the
    measured vendor FWHM range of its group, so no panel claims a width its filters do not have."""
    on_grid = [b for b in narrow if b.on_grid25]
    inter = [b for b in narrow if not b.on_grid25]

    def rng(bands):
        f = [b.fwhm for b in bands]
        return f"FWHM {min(f):.0f}\u2013{max(f):.0f} nm"

    tags = (f"(a) 25 nm grid, m400\u2013m875 ({len(on_grid)}; {rng(on_grid)})",
            f"(b) intermediate bands ({len(inter)}; {rng(inter)})",
            f"(c) w bands ({len(wide)}; {rng(wide)})")
    # a little more headroom than v3 so the panel tag clears the band labels below it
    if normalize:
        ylim, ymaj, ymin = (0, 1.36), 0.5, 0.1
        ylabel = "Normalized response"
    else:
        ylim, ymaj, ymin = (0, 0.90), 0.2, 0.05
        ylabel = "System throughput"
    fig, axes = stacked_axes(3, panel_h=1.42)
    for ax, bands, xl, tag in zip(axes, (on_grid, inter, wide), (False, False, True), tags):
        draw_bands(ax, bands, fill_alpha=fill_alpha, normalize=normalize)
        style_axes(ax, ylim, ymaj, ymin, xlabel=xl)
        label_bands(ax, bands, ylim, mode="peak", normalize=normalize)
        panel_tag(ax, tag)
    fig.supylabel(ylabel, x=0.012, fontsize=8)
    save(fig, stem)


def write_width_table(narrow, wide):
    """CSV for the paper's filter table: vendor FWHM (filter only) and response-based quantities."""
    path = OUT / "filter_widths.csv"
    with open(path, "w") as f:
        f.write("name,group,lam_cen_filter_nm,fwhm_filter_nm,half_lo_nm,half_hi_nm,"
                "lam_pivot_response_nm,w_eq_response_nm,peak_response\n")
        for b in narrow + wide:
            grp = "wide" if b.wide else ("grid25" if b.on_grid25 else "intermediate")
            f.write(f"{b.name},{grp},{0.5 * (b.fwhm_lo + b.fwhm_hi):.1f},{b.fwhm:.1f},{b.fwhm_lo:.1f},{b.fwhm_hi:.1f},"
                    f"{b.pivot:.1f},{b.w_eq:.1f},{b.peak:.3f}\n")
    print(f"saved {path}")


def v5_components(stem):
    """Appendix figure: the wavelength-dependent efficiencies common to all bands."""
    with fits.open(SRC / "skytable.fits") as hdul:
        lam = np.asarray(hdul[1].data["lam"], float)
        tsky = np.asarray(hdul[1].data["trans"], float)
    # 1 nm mean-binning of the SkyCalc line forest (display only)
    edges = np.arange(300.0, 1001.0, 1.0)
    idx = np.digitize(lam, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    grid = 0.5 * (edges[:-1] + edges[1:])
    sums = np.bincount(idx[ok], weights=tsky[ok], minlength=len(grid))
    cnts = np.bincount(idx[ok], minlength=len(grid))
    sky = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)

    qe = np.loadtxt(SRC / "QE.csv", delimiter=",", skiprows=1)
    rows = [l.split() for l in open(SRC / "optics.efficiency.dr350.dr500.csv")
            if not l.startswith("#") and not l.startswith("nm")]
    onm = np.array([r[0] for r in rows], float)
    otot = np.array([r[-1] for r in rows], float)

    t_qe = np.interp(grid, qe[:, 0], qe[:, 1])
    t_opt = np.interp(grid, onm, otot)
    slope = (otot[1] - otot[0]) / (onm[1] - onm[0])
    below = grid < onm[0]
    t_opt[below] = np.maximum(otot[0] + slope * (grid[below] - onm[0]), 0.0)
    total = t_qe * t_opt * sky

    fig = plt.figure(figsize=(COL_W, 2.75))
    ax = fig.add_axes([0.135, 0.148, 0.845, 0.70])
    ax.plot(grid, sky, color="#8c8c8c", lw=0.8, label="Atmosphere (SkyCalc)")
    ax.plot(grid, t_qe, color="#009E73", lw=1.0, label="Detector QE (IMX455)")
    meas = ~below
    ax.plot(grid[meas], t_opt[meas], color="#0072B2", lw=1.0, label="Optics (DR500)")
    ax.plot(grid[below | (grid == onm[0])], t_opt[below | (grid == onm[0])], color="#0072B2", lw=1.0, ls=(0, (2.5, 1.8)))
    ax.plot(onm, otot, "o", color="#0072B2", ms=2.6, mew=0)
    ax.plot(grid, total, color="black", lw=1.4, label="Product (excluding filter)")
    ax.set_xlim(300, 1000)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Efficiency")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
              handlelength=1.8, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0)
    save(fig, stem)


def colour_key(narrow, wide, stem):
    """Diagnostic strip: the equal-lightness colours actually assigned to each band."""
    fig, ax = plt.subplots(figsize=(FULL_W, 0.9))
    for i, b in enumerate(narrow + wide):
        ax.bar(i, 1, color=b.colour, width=1.0)
        ax.text(i, -0.08, b.name, rotation=90, ha="center", va="top", fontsize=5.5, color=INK)
    ax.set_xlim(-0.5, len(narrow + wide) - 0.5)
    ax.set_ylim(-1.2, 1)
    ax.axis("off")
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.98)
    save(fig, stem)


def main():
    narrow, wide = load_bands()
    print(f"{len(narrow)} narrow (25 nm) + {len(wide)} wide (50 nm) bands")
    for b in narrow + wide:
        print(f"  {b.name:6s} pivot {b.pivot:6.1f} nm  peak {b.peak:.3f}  band {b.lam[0]:.0f}-{b.lam[-1]:.0f} nm")

    v1_two_panel_clean(narrow, wide, fill_alpha=0.08, stem="v1_two_panel_clean_fill")
    v1_two_panel_clean(narrow, wide, fill_alpha=0.0, stem="v1_two_panel_clean_lines")
    v2_two_panel_labels(narrow, wide, stem="v2_two_panel_labels")
    v3_three_panel(narrow, wide, normalize=False, fill_alpha=0.08, stem="v3_three_panel_throughput")
    v3_three_panel(narrow, wide, normalize=True, fill_alpha=0.08, stem="v4_three_panel_normalized")
    v6_three_groups(narrow, wide, normalize=False, fill_alpha=0.08, stem="v6_three_groups_throughput")
    v6_three_groups(narrow, wide, normalize=True, fill_alpha=0.08, stem="v6_three_groups_normalized")
    write_width_table(narrow, wide)
    v5_components(stem="v5_appendix_components")
    # final choice (2026-09-27): v6 grouping by filter set -> paper Fig. 1 files next to the response data
    import shutil
    for src, dst in (("v6_three_groups_throughput", "7DT_transmission_fig1"),
                     ("v6_three_groups_normalized", "7DT_transmission_fig1_normalized")):
        for ext in (".pdf", ".png"):
            shutil.copyfile(OUT / f"{src}{ext}", BASE / f"{dst}{ext}")
            print(f"final: {BASE / f'{dst}{ext}'}")
    colour_key(narrow, wide, stem="zz_colour_key")


if __name__ == "__main__":
    main()
