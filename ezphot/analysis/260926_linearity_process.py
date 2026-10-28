#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260926_linearity_process.py — raw exposure-time series (one field, e.g. g band on all units) -> calibrated,
astrometrically solved science frames for 260926_linearity_test.py.  Adapted from 260918_spss_process.py.

  masters per unit from the night's BIAS/DARK/FLAT (generate_masterframe); a filter without a flat that night falls
  back to the nearest archived master flat (get_masterframe_from_image, 30 days)
  -> correct_bdf (nearest-exposure dark, scaled) -> solve_astrometry (astrometry.net).  No photometry here.
Writes the processed-frame list to ~/ezphot/log/20260926_linearity/<tag>/frames.txt (one path per line).
Resume: raw frames already listed as ok in process_meta_<unit>.csv are skipped.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260926_linearity_process.py --night 2026-09-26_gain2750 --filter g --tag g_20260926 --n-proc 16
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
import portalocker                  # noqa: E402


class _NoLock:
    def __init__(self, *a, **k): pass
    def __enter__(self): return None
    def __exit__(self, *a): return False


portalocker.Lock = _NoLock
portalocker.lock = lambda *a, **k: None
portalocker.unlock = lambda *a, **k: None

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot")

BASE = "/lyman/data1/obsdata/obsdata_from_mcs"
CALIB = Path("/home/hhchoi1022/ezphot/data/calibdata/7DT/7DT_C361K_HIGH_1x1")
OUTROOT = Path("/home/hhchoi1022/ezphot/log/20260926_linearity")
SKIP_OBJ = {"BIAS", "DARK", "FLAT"}
_G = {}


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def parse(name):
    parts = name.split("_")
    if len(parts) < 8:
        return None
    obj = "_".join(parts[3:-4])
    return obj, parts[-4], float(parts[-2].replace("s", ""))        # object, filter, exptime


def science_frames(unit, nights, filt, obj_sub, root=BASE):
    out = []
    for n in nights:
        files = sorted(glob.glob(f"{root}/{unit}/{n}/*.fits")) or sorted(glob.glob(f"{root}/{unit}/image/{n}/*.fits"))
        for p in files:
            r = parse(Path(p).name)
            if r is None:
                continue
            obj, f, exp = r
            if obj in SKIP_OBJ or obj.startswith("7DT-main") or f != filt or (obj_sub and obj_sub not in obj):
                continue
            out.append((p, obj, f, exp, n))
    return out


def build_unit(unit, nights):
    from ezphot.helper import Helper
    from ezphot.imageobjects import CalibrationImage
    from ezphot.methods import Preprocess
    telinfo = Helper().get_telinfo(telescope="7DT", ccd="C361K", readoutmode="HIGH", binning=1)
    calib = []
    for n in nights:
        for pat in ("*BIAS*.fits", "*DARK*.fits", "*FLAT*.fits"):
            calib += glob.glob(f"{BASE}/{unit}/image/{n}/{pat}")
    if not calib:
        return telinfo, None
    pre = Preprocess()
    imgs = [CalibrationImage(p, telinfo=telinfo, load=False) for p in sorted(calib)]
    masters = pre.generate_masterframe(imgs, n_proc=8, verbose=False, save=True)
    return telinfo, list(masters.values())[0]


def calibdata_masters(unit, filt, dark_exp=100.0, since="2026"):
    """latest archived master BIAS / DARK (dark_exp s; scaled to EXPTIME by correct_bdf) / FLAT(filt) in calibdata."""
    from ezphot.imageobjects import MasterImage
    b = sorted(glob.glob(str(CALIB / "BIAS" / unit / f"{unit}_{since}*_BIAS_*.com.fits")))
    d = sorted(glob.glob(str(CALIB / "DARK" / unit / f"{unit}_{since}*_DARK_*_{dark_exp:.1f}s_*.com.fits")))
    f = sorted(glob.glob(str(CALIB / "FLAT" / unit / f"{unit}_{since}*_FLAT_{filt}_1x1_*.com.fits")))
    if not (b and d and f):
        return None
    return {"BIAS": MasterImage(b[-1]), "DARK": {dark_exp: MasterImage(d[-1])}, "FLAT": {filt: MasterImage(f[-1])},
            "files": (b[-1], d[-1], f[-1])}


def _init_worker(telinfo, mset):
    warnings.filterwarnings("ignore")
    _G["telinfo"], _G["mset"] = telinfo, mset
    from ezphot.methods import Preprocess, Platesolve
    _G["pre"], _G["ps"] = Preprocess(), Platesolve()


def process(args):
    raw, obj, filt, exp, night = args
    row = dict(raw=raw, object=obj, filter=filt, exptime=exp, night=night, unit=Path(raw).name[:5])
    try:
        from ezphot.imageobjects import ScienceImage
        telinfo, mset, pre = _G["telinfo"], _G["mset"], _G["pre"]
        sci = ScienceImage(raw, telinfo=telinfo, load=False)
        bias = mset["BIAS"] if mset else pre.get_masterframe_from_image(sci, "BIAS", 30)[0]
        if mset and mset.get("DARK"):
            mdark = mset["DARK"][sorted(mset["DARK"], key=lambda k: abs(float(k) - exp))[0]]
        else:
            mdark = pre.get_masterframe_from_image(sci, "DARK", 30)[0]
        mflat = (mset or {}).get("FLAT", {}).get(filt) if mset else None
        row["flat_source"] = "night"
        if mflat is None:
            mflat = pre.get_masterframe_from_image(sci, "FLAT", 30)[0]
            row["flat_source"] = "archive"
        if bias is None or mdark is None or mflat is None:
            return dict(row, status="missing_master")
        cal = pre.correct_bdf(target_img=sci, bias_image=bias, dark_image=mdark, flat_image=mflat, save=True, verbose=False)
        out = _G["ps"].solve_astrometry(cal, overwrite=True, verbose=False)
        if out is None:
            return dict(row, status="astrometry_failed")
        return dict(row, path=str(out.savepath.savepath), status="ok")
    except Exception as e:                                           # noqa: BLE001
        return dict(row, status="error:" + str(e)[:150])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", nargs="+", required=True)
    ap.add_argument("--filter", default="g")
    ap.add_argument("--object", default="", help="substring of the field name (optional)")
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="only list the frames that would be processed")
    ap.add_argument("--raw-root", default=BASE, help="raw root: <root>/<unit>/<night>/ or <root>/<unit>/image/<night>/")
    ap.add_argument("--calibdata-masters", action="store_true",
                    help="use the latest archived masters in calibdata (2026) instead of the night's calibration frames")
    a = ap.parse_args()
    out = OUTROOT / a.tag; out.mkdir(parents=True, exist_ok=True)
    units = a.units or sorted({Path(p).parts[-3] for n in a.night for p in glob.glob(f"{a.raw_root}/7DT*/{n}/*.fits")}
                              | {Path(p).parts[-4] for n in a.night for p in glob.glob(f"{a.raw_root}/7DT*/image/{n}/*.fits")})
    for unit in units:
        frames = science_frames(unit, a.night, a.filter, a.object, a.raw_root)
        metap = out / f"process_meta_{unit}.csv"
        done = set(pd.read_csv(metap).query("status == 'ok'").raw) if metap.exists() else set()
        todo = [f for f in frames if f[0] not in done]
        log(f"=== {unit}: {len(frames)} {a.filter} frames ({len(frames) - len(todo)} done) "
            f"exptimes {sorted({f[3] for f in frames})} objects {sorted({f[1] for f in frames})}")
        if not todo or a.dry_run:
            continue
        if a.calibdata_masters:
            from ezphot.helper import Helper
            telinfo = Helper().get_telinfo(telescope="7DT", ccd="C361K", readoutmode="HIGH", binning=1)
            mset = calibdata_masters(unit, a.filter)
            if mset is None:
                log(f"{unit}: missing calibdata masters, skipping"); continue
            log(f"{unit}: calibdata masters {[Path(x).name for x in mset.pop('files')]}")
        else:
            telinfo, mset = build_unit(unit, a.night)
            log(f"{unit}: masters " + (f"darks {list(mset['DARK'])}, flats {list(mset['FLAT'])}" if mset else "none (archive fallback)"))
        rows = pd.read_csv(metap).to_dict("records") if metap.exists() else []
        with Pool(a.n_proc, initializer=_init_worker, initargs=(telinfo, mset)) as pool:
            for i, r in enumerate(pool.imap_unordered(process, todo), 1):
                rows.append(r)
                if i % 20 == 0:
                    pd.DataFrame(rows).to_csv(metap, index=False)
        df = pd.DataFrame(rows).drop_duplicates("raw", keep="last")
        df.to_csv(metap, index=False)
        log(f"{unit} status: {df.status.str[:24].value_counts().to_dict()}")
    if a.dry_run:
        return
    allm = pd.concat([pd.read_csv(p) for p in sorted(out.glob("process_meta_*.csv"))], ignore_index=True)
    ok = allm[allm.status == "ok"]
    (out / "frames.txt").write_text("\n".join(ok.path) + "\n")
    log(f"frames.txt: {len(ok)} processed frames; status {allm.status.str[:24].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
