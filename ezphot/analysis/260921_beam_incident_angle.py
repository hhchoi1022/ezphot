#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Incident-angle map on the 7DT filter across the image plane (DR500 f/3.03).

Geometry: aperture stop = primary. Exit pupil = image of the primary formed by
the secondary, solved from published DR500 numbers (D=508, FL=1537, secondary
286 mm, back focus ~250 mm behind primary vertex, OTA 892 mm). The near-focal
3-lens corrector has little pupil power and is neglected.

v2 (2026-09-21): adds the 50 mm filter as a vignetting stop (clear radius and
distance fitted to the measured m650 flat illumination profile, FB2). Ray
angles are computed exactly (pupil point -> image point), so the chief-ray
tilt and the clipping interact correctly: the clipped rays are the LOW-angle
ones (same side as the image point), which pushes corner CWL further blue.
CWL model: dlam/lam = sqrt(1 - <sin^2 th>/n_eff^2) - 1.
v4 (2026-10-01): n_eff depends on the filter, n_eff(lam0) = 1.962 - 0.147 (lam0 - 480) / 100 [lam0 in nm] (user's
determination); shifts at 550 / 875 nm use n_eff(550) / n_eff(875).  Panel B shows the incidence angle on the filter
(effective = rms over the surviving light cone) instead of the chief-ray tilt; the central-ray angle stays in panel C.
Panel A marks the blocked light: the central obscuration (secondary shadow) and, for the corner field, the half of the
meridional beam that hits the 50 mm filter cell; A' zooms on the filter.
Old outputs (n_eff 1.5): FB1_incident_angle_neff1p5.png, beam_incident_angle_neff1p5.csv.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGD = Path("/home/hhchoi1022/ezphot/log/20260921_curve_versions")

# ---------- published DR500 numbers ----------
D_AP = 508.0        # aperture [mm]
FL = 1537.0         # system focal length [mm]
FNUM = FL / D_AP    # 3.025
EPS = 0.59          # central obscuration (diameter ratio)
D_SEC = 286.0       # secondary diameter [mm]
B = 250.0           # focal plane behind primary vertex [mm]
NEFF0, NEFF1 = 1.962, -0.00147            # n_eff(lam0) = NEFF0 + NEFF1 (lam0 - 480); v3 used 2.2898, -0.002651 (discarded)


def n_eff(lam):
    return NEFF0 + NEFF1 * (lam - 480.0)


FILT_A = 24.25      # filter clear radius [mm]  (fit to m650 flat profile, FB2)
FILT_Z = 59.0       # filter distance before focus [mm] (same fit)

# ---------- solve Cassegrain layout: separation d such that the on-axis
# beam footprint on the secondary matches its 286 mm diameter ----------
best = None
for d in np.arange(400.0, 750.0, 1.0):
    f1 = FL * d / (FL - d - B)
    beam = D_AP * (1.0 - d / f1)
    if best is None or abs(beam - D_SEC) < abs(best[2] - D_SEC):
        best = (d, f1, beam)
d, f1, beam = best
m = (d + B) / (f1 - d)
f2 = 1.0 / (1.0 / (d + B) - 1.0 / (f1 - d))
s_i = 1.0 / (1.0 / f2 - 1.0 / d)
L_XP = abs(s_i) + d + B                    # exit pupil to focal plane [mm]
D_XP = D_AP * abs(s_i) / d                 # exit pupil diameter [mm]
R_XP = D_XP / 2
print(f"layout: d={d:.0f}  f1={f1:.0f} (f/{f1/D_AP:.2f})  m={m:.3f}  beam@sec={beam:.0f}")
print(f"exit pupil: L_xp={L_XP:.0f} mm  D_xp={D_XP:.0f} mm  check f/# = {L_XP/D_XP:.3f}")

# ---------- unvignetted cone statistics ----------
TAN_OUT = 1.0 / (2.0 * FNUM)
th_out = np.degrees(np.arctan(TAN_OUT))
th_in = np.degrees(np.arctan(EPS * TAN_OUT))

# ---------- exact ray model over the pupil ----------
n = 601
Xp, Yp = np.meshgrid(np.linspace(-R_XP, R_XP, n), np.linspace(-R_XP, R_XP, n))
rr = np.hypot(Xp, Yp)
ANN = (rr >= EPS * R_XP) & (rr <= R_XP)

def field_point(h, clip=True):
    """(survival, <sin^2 th>, rms, min, max, mean-direction tilt) for image height h.
    All angle statistics are over the SURVIVING rays when clip=True."""
    tan2 = ((Xp - h)**2 + Yp**2) / L_XP**2
    s2 = tan2 / (1 + tan2)
    if clip:
        P2 = (h * (1 - FILT_Z / L_XP) + (FILT_Z / L_XP) * Xp)**2 \
             + ((FILT_Z / L_XP) * Yp)**2
        msk = ANN & (P2 <= FILT_A**2)
    else:
        msk = ANN
    s2m = s2[msk].mean()
    th = np.degrees(np.arctan(np.sqrt(tan2[msk])))
    tilt_eff = np.degrees(np.arctan(abs(((Xp - h)[msk] / L_XP).mean())))
    return (msk.sum() / ANN.sum(), s2m, np.degrees(np.arcsin(np.sqrt(s2m))),
            th.min(), th.max(), tilt_eff)

_, S2_0, th_rms = field_point(0.0, clip=False)[:3]


def g0(lam):           # on-axis full-beam CWL factor
    return np.sqrt(1 - S2_0 / n_eff(lam)**2)


print(f"cone: {th_in:.2f}-{th_out:.2f} deg, rms {th_rms:.2f} deg; center shift "
      f"550nm {550*(g0(550)-1):+.2f} nm (n_eff {n_eff(550):.3f}), 875nm {875*(g0(875)-1):+.2f} nm (n_eff {n_eff(875):.3f})")

def dcwl(s2, lam):     # CWL change vs on-axis full beam [nm]
    return lam * (np.sqrt(1 - s2 / n_eff(lam)**2) - g0(lam))

# radial profile (rotational symmetry)
h_c = np.hypot(18, 12)
r1d = np.linspace(0, h_c, 120)
prof = np.array([field_point(h) for h in r1d])            # illum, s2, rms
prof_nc = np.array([field_point(h, clip=False) for h in r1d])
d550 = dcwl(prof[:, 1], 550.0) * 1e3                      # pm
d550_nc = dcwl(prof_nc[:, 1], 550.0) * 1e3

# ---------- reference points table ----------
import csv
pts = [("center", 0.0), ("short-edge (12 mm)", 12.0), ("long-edge (18 mm)", 18.0),
       ("corner (21.6 mm)", h_c)]
rows = []
for name, r in pts:
    fr, s2v, thr = field_point(r)[:3]
    thc = np.degrees(np.arctan(r / L_XP))
    rows.append((name, r, thc, thr, fr, dcwl(s2v, 550), dcwl(s2v, 875)))
    print(f"{name:20s} r={r:5.1f}  th_c={thc:.3f}  th_rms={thr:.3f}  illum={fr:.3f}  "
          f"dCWL550={rows[-1][5]*1e3:+.0f} pm  dCWL875={rows[-1][6]*1e3:+.0f} pm")
with open(FIGD / "beam_incident_angle.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["position", "r_mm", "central_ray_incidence_deg", "theta_rms_deg",
                "illumination", "dCWL550_nm", "dCWL875_nm"])
    for r_ in rows:
        w.writerow([r_[0]] + [f"{v:.4f}" for v in r_[1:]])

# ---------- figure ----------
fig = plt.figure(figsize=(18, 9.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 0.85, 1.0], hspace=0.32, wspace=0.4)

# (A) side-view schematic, with the blocked light: central obscuration and (corner field) the filter cell
ax = fig.add_subplot(gs[0, 0])
zP, rO, rI = -L_XP, R_XP, EPS * R_XP
zF = -FILT_Z
r_at_filter = lambda xp, h: h * (1 - FILT_Z / L_XP) + xp * FILT_Z / L_XP   # meridional ray pupil -> image point h


def draw_beams(ax, lw=1.0):
    for s in (+1, -1):
        ax.plot([zP, 0], [s * rO, 0], color="#2F5D9E", lw=1.6 * lw)
        ax.plot([zP, 0], [s * rI, 0], color="#2F5D9E", lw=1.0 * lw, ls="--")
    ax.fill([zP, zP, 0], [rI, rO, 0], color="#2F5D9E", alpha=0.10)
    ax.fill([zP, zP, 0], [-rI, -rO, 0], color="#2F5D9E", alpha=0.10)
    # central obscuration: the secondary's shadow, no light inside the inner rays
    ax.fill([zP, zP, 0], [-rI, rI, 0], facecolor="0.85", edgecolor="0.55", hatch="///", lw=0, alpha=0.6)
    # corner field: lower half of the meridional annulus reaches the corner image point ...
    ax.fill([zP, zP, 0], [-rO, -rI, h_c], color="#C25B4E", alpha=0.12)
    # ... the upper half hits the filter cell (outside the 24.25 mm clear radius) and stops there
    ax.fill([zP, zP, zF, zF], [rI, rO, r_at_filter(rO, h_c), r_at_filter(rI, h_c)],
            facecolor="none", edgecolor="#C25B4E", hatch="xxx", lw=0.6)
    for xp in (-rO, -rI):
        ax.plot([zP, 0], [xp, h_c], color="#C25B4E", lw=1.0 * lw, alpha=0.75)
    for xp in (rI, rO):
        ax.plot([zP, zF], [xp, r_at_filter(xp, h_c)], color="#C25B4E", lw=1.0 * lw, alpha=0.75, ls=":")
    ax.plot([zP, 0], [0, h_c], color="#C25B4E", lw=1.8 * lw)


draw_beams(ax)
ax.text(-1180, 0, "central obscuration\n(secondary shadow, ε = 0.59)", fontsize=7.5, color="0.3", ha="left", va="center",
        bbox=dict(fc="w", ec="none", alpha=0.7, pad=1))
# filter stop
ax.plot([-FILT_Z, -FILT_Z], [FILT_A, FILT_A + 55], color="k", lw=3)
ax.plot([-FILT_Z, -FILT_Z], [-FILT_A, -FILT_A - 55], color="k", lw=3)
ax.annotate(f"50 mm filter\n(clear r {FILT_A} mm, z {FILT_Z:.0f} mm)",
            (-FILT_Z, FILT_A + 55), xytext=(-700, 285), fontsize=7.5, va="top",
            arrowprops=dict(arrowstyle="-", lw=0.7, color="0.4"))
ax.axvline(0, color="0.2", lw=1.2)
ax.text(6, -rO * 0.9, "focal\nplane", fontsize=8, color="0.2")
ax.annotate(f"exit pupil\nL={L_XP:.0f} mm, D={D_XP:.0f} mm\n(annulus, eps=0.59)",
            (zP, -rO), xytext=(zP + 60, -rO * 0.80), fontsize=7.5, color="#2F5D9E")
ax.text(-620, 95, f"marginal ray {th_out:.2f} deg", fontsize=7.5, color="#2F5D9E",
        rotation=-10)
ax.text(-620, 33, f"inner ray {th_in:.2f} deg", fontsize=7.5, color="#2F5D9E",
        rotation=-6)
ax.set(xlabel="z from focal plane [mm]", ylabel="radial position [mm]",
       title=f"A. beam geometry (f/{FNUM:.2f})")
ax.set_xlim(-1500, 130); ax.set_ylim(-260, 300)
ax.grid(alpha=0.2)

# (A') zoom on the filter: what is blocked where
ax = fig.add_subplot(gs[0, 1])
draw_beams(ax, lw=0.8)
ax.plot([zF, zF], [FILT_A, 42], color="k", lw=4, solid_capstyle="butt")
ax.plot([zF, zF], [-FILT_A, -42], color="k", lw=4, solid_capstyle="butt")
ax.plot([zF, zF], [-FILT_A, FILT_A], color="0.6", lw=1.0, ls="-")
ax.plot(0, h_c, "o", color="#C25B4E", ms=4); ax.plot(0, 0, "o", color="#2F5D9E", ms=4)
ax.text(4, h_c, f"corner\nimage\n{h_c:.1f} mm", fontsize=6.5, color="#C25B4E", va="center")
ax.text(4, -1.5, "center", fontsize=6.5, color="#2F5D9E", va="top")
ax.text(zF + 3, 40, "filter cell", fontsize=7, va="top")
ax.text(zF + 3, FILT_A - 1, f"clear r {FILT_A} mm", fontsize=6.5, va="top", color="0.35")
from matplotlib.patches import Patch   # noqa: E402
hd = [Patch(facecolor="#2F5D9E", alpha=0.25, label="center: light cone (annulus)"),
      Patch(facecolor="0.85", edgecolor="0.55", hatch="///", label="no light: secondary shadow (ε 0.59)"),
      Patch(facecolor="#C25B4E", alpha=0.25, label="corner: part that reaches the image"),
      Patch(facecolor="none", edgecolor="#C25B4E", hatch="xxx", label="corner: part blocked by the filter cell")]
ax.legend(handles=hd, fontsize=7, loc="lower left", framealpha=0.95)
ax.set(xlim=(-140, 24), ylim=(-62, 44), xlabel="z from focal plane [mm]", ylabel="radial position [mm]",
       title="A′. zoom near the filter (meridional section)")
ax.tick_params(labelsize=7); ax.grid(alpha=0.2)

# (B) incidence-angle map: effective (rms) incidence angle of the light reaching each detector point
x = np.linspace(-18, 18, 361); y = np.linspace(-12, 12, 241)
X, Y = np.meshgrid(x, y)
Rm = np.hypot(X, Y)
TH_E = np.interp(Rm, r1d, prof[:, 2])
ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(TH_E, extent=[-18, 18, -12, 12], origin="lower", cmap="viridis")
cs = ax.contour(X, Y, TH_E, levels=[7.75, 7.9, 8.05], colors="w", linewidths=0.8)
ax.clabel(cs, fmt="%.2f°", fontsize=7, inline_spacing=2)
fig.colorbar(im, ax=ax, label="incidence angle, rms of the cone [deg]", shrink=0.85)
ax.text(0, 0, f"center {prof[0, 2]:.2f}°\n(cone {th_in:.1f}–{th_out:.1f}°)", color="w", ha="center", va="center", fontsize=8)
ax.set(xlabel="detector x [mm]", ylabel="detector y [mm]",
       title="B. Incidence angle on the filter\n(detector IMX455, 36 x 24 mm)")

# (C) incidence angle + illumination vs field radius (surviving-ray statistics)
ax = fig.add_subplot(gs[1, :2])
thc1d = np.degrees(np.arctan(r1d / L_XP))
ax.fill_between(r1d, prof[:, 3], prof[:, 4], color="#2F5D9E", alpha=0.15,
                label="incidence range of SURVIVING rays")
ax.plot(r1d, prof_nc[:, 3], color="#2F5D9E", lw=0.9, ls=":")
ax.plot(r1d, prof_nc[:, 2], color="#2F5D9E", lw=1.4, ls="--",
        label="rms angle, no vignetting")
ax.plot(r1d, prof[:, 2], color="#2F5D9E", lw=2,
        label="rms angle of SURVIVING rays (50 mm filter)")
ax.plot(r1d, thc1d, color="#C25B4E", lw=1.6, label="incidence angle of the central ray")
ax.plot(r1d, prof[:, 5], color="#C25B4E", lw=1.6, ls="--",
        label="mean direction of SURVIVING rays")
ax.axvspan(15.1, r1d[-1], color="#C25B4E", alpha=0.06, lw=0)
ax.axvline(15.1, color="0.6", lw=0.9, ls=":")
ax.text(15.1, 4.3, "filter vignetting onset", rotation=90, fontsize=7.5,
        color="0.35", ha="right", va="bottom")
ax2 = ax.twinx()
ax2.plot(r1d, prof[:, 0], color="0.45", lw=1.3)
ax2.set_ylabel("pupil survival fraction", color="0.45")
ax2.set_ylim(0, 1.05); ax2.tick_params(colors="0.45")
ax.set(xlabel="field radius on detector [mm]", ylabel="incidence angle on filter [deg]",
       title="C. Incidence angle vs field radius", ylim=(0, 11))
ax.legend(fontsize=7.5, loc="center left")
ax.grid(alpha=0.25)

# (D) total CWL shift map (chief ray + filter vignetting)
D550 = np.interp(Rm, r1d, d550)
ax = fig.add_subplot(gs[1, 2])
im = ax.imshow(D550, extent=[-18, 18, -12, 12], origin="lower", cmap="magma_r")
cs = ax.contour(X, Y, D550, levels=[-200, -100, -50, -10], colors="w", linewidths=0.8)
ax.clabel(cs, fmt="%.0f pm", fontsize=7)
fig.colorbar(im, ax=ax, label="extra CWL blueshift at 550 nm [pm]", shrink=0.85)
ax.set(xlabel="detector x [mm]", ylabel="detector y [mm]",
       title=f"D. Extra CWL shift, 550 nm\nn_eff {n_eff(550):.2f}; center itself {550*(g0(550)-1):.2f} nm")
D875 = np.interp(Rm, r1d, dcwl(prof[:, 1], 875.0) * 1e3)
print(f"detector-area mean extra shift vs center: 550 nm {D550.mean():+.1f} pm, 875 nm {D875.mean():+.1f} pm; "
      f"r<15 mm max |shift| 550 {abs(D550[Rm < 15]).max():.1f} pm, 875 {abs(D875[Rm < 15]).max():.1f} pm")

fig.suptitle("Incidence angle on the 7DT filter across the image plane "
             f"(DR500 f/{FNUM:.2f}, eps={EPS}, 50 mm filter stop; n_eff = {NEFF0} − {100 * abs(NEFF1):.3f} (λ0 − 480)/100)",
             fontsize=12.5)
fig.savefig(FIGD / "FB1_incident_angle.png", dpi=140, bbox_inches="tight")
print("saved", FIGD / "FB1_incident_angle.png")
