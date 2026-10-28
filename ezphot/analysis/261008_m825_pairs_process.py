#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261008_m825_pairs_process.py — frames for the 2024-vs-2026 m825 comparison of the same standards (WD2149+021 = G93-48 etc.).

  part "reobs": raw CALSPEC re-observation frames of night 2026-10-02 (obsdata_from_mcs, no calibration frames that night)
                for WD0227+050 / WD0308-565 in m800 / m825 / m850 -> 260918_spss_process chain (masters from the unit's
                2026-09-27/28 BIAS/DARK/FLAT in /lyman/data1/obsdata/<unit>/<night>, correct_bdf, astrometry) -> the
                processed frame in the ezphot scidata tree -> 260927_std_rephot.process (new background, 10" MAG_APER_3).
                7DT11 skipped (filter wheel unreliable since the SPSS campaign).
  part "rephot2024": every 2024 CALSPEC frame of the 7DT10 m800 / m825 and 7DT11 m850 sets (260927_std_rephot used at most
                8 per set) -> 260927_std_rephot.process.
Raw/pipeline trees are only read.  Catalogs go to the usual 20260927_std_newbkg/catalogs (existing ones are skipped);
the frame list of this study goes to ~/ezphot/log/20261008_m825_pairs/frames_<part>.csv.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib python -u ezphot/analysis/261008_m825_pairs_process.py --part reobs --n-proc 8
  MPLCONFIGDIR=/tmp/ezphot-matplotlib python -u ezphot/analysis/261008_m825_pairs_process.py --part rephot2024 --n-proc 32
"""
import argparse
import glob
import importlib
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import pandas as pd                 # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
P = importlib.import_module("260918_spss_process")          # applies the portalocker no-op patch
RP = importlib.import_module("260927_std_rephot")

OUT = Path("/home/hhchoi1022/ezphot/log/20261008_m825_pairs")
REOBS_BASE = "/lyman/data1/obsdata/obsdata_from_mcs/{unit}/image/{night}"
REOBS_NIGHT = "2026-10-02_gain2750"
CALIB_BASE = "/lyman/data1/obsdata/{unit}/{night}"
CALIB_NIGHTS = ["2026-09-27_gain2750", "2026-09-28_gain2750"]
REOBS_OBJ = ("WD0227+050", "WD0308-565")
FILTERS = ("m800", "m825", "m850")


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def reobs_frames(unit):
    out = []
    for p in sorted(glob.glob(REOBS_BASE.format(unit=unit, night=REOBS_NIGHT) + "/*.fits")):
        parts = Path(p).name.split("_")
        if len(parts) >= 8 and parts[3] in REOBS_OBJ and parts[4] in FILTERS:
            out.append((p, parts[3], parts[4], REOBS_NIGHT))
    return out


def build_unit(unit):
    """P.build_unit with the calibration frames taken from the raw tree of the nearest nights that have them."""
    P.BASE, P.CALIB_NIGHTS = CALIB_BASE, CALIB_NIGHTS
    try:
        return P.build_unit(unit)
    finally:
        P.BASE = REOBS_BASE


def run_reobs(n_proc, only_units=None):
    P.CATD = OUT / "process_catalogs"                       # P's own (old-background) catalogs, not used downstream
    P.CATD.mkdir(parents=True, exist_ok=True)
    rows = []
    units = sorted({Path(u).parts[-3] for u in glob.glob(REOBS_BASE.format(unit="7DT*", night=REOBS_NIGHT))})
    for unit in sorted(units, key=lambda x: x != "7DT12"):                     # 7DT12 (m825) first
        if unit == "7DT11" or (only_units and unit not in only_units):
            continue
        todo = reobs_frames(unit)
        if not todo:
            continue
        done = [t for t in todo if (P.CATD / (Path(t[0]).name[:-5] + "_cat.fits")).exists()]
        if len(done) == len(todo):                                  # already processed: no masters needed
            rows += [dict(file=t[0], object=t[1], filter=t[2], night=t[3], status="done", unit=unit) for t in todo]
            log(f"{unit}: {len(todo)} frames already processed"); continue
        log(f"{unit}: {len(todo)} frames, building masters ...")
        telinfo, _, mset = build_unit(unit)
        if mset is None:
            log(f"{unit}: no calibration frames"); continue
        log(f"{unit}: darks {list(mset['DARK'])}, flats {list(mset['FLAT'])}")
        with Pool(n_proc, initializer=P._init_worker, initargs=(telinfo, mset)) as pool:
            for r in pool.imap_unordered(P.process, todo):
                rows.append(dict(r, unit=unit))
        pd.DataFrame(rows).to_csv(OUT / "process_meta_reobs.csv", index=False)
    M = pd.DataFrame(rows)
    log("processing status: " + repr(M.status.str[:24].value_counts().to_dict()))
    ok = M[M.status.isin(["ok", "done"])]
    jobs = [("CALSPEC26", f"{RP.SCI}/{o}/{u}/{f}/{Path(r).name}", o, f, u) for r, o, f, u in zip(ok.file, ok.object, ok["filter"], ok.unit)]
    return jobs


def run_rephot2024():
    c = pd.read_csv(RP.CAL_META)
    c = c[(c.status == "ok") & (((c.unit == "7DT10") & c["filter"].isin(["m800", "m825"])) | ((c.unit == "7DT11") & (c["filter"] == "m850")))]
    # order: m825 of the stars also observed in 2026 first, then the other m825 frames, then m800 / m850
    pri = {"WD2149+021", "WD2039-682", "GJ754.1A", "FEIGE110", "WD0227+050", "WD0308-565"}
    c = c.assign(rank=[(0 if f == "m825" else 2) + (0 if o in pri else 1) for f, o in zip(c["filter"], c.object)]).sort_values(["rank", "file"])
    return [("CALSPEC", f, o, flt, u) for f, o, flt, u in zip(c.file, c.object, c["filter"], c.unit)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["reobs", "rephot2024"], required=True)
    ap.add_argument("--n-proc", type=int, default=8)
    ap.add_argument("--units", nargs="*", default=None, help="reobs: only these units")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); RP.CATD.mkdir(parents=True, exist_ok=True); RP.WORK.mkdir(parents=True, exist_ok=True)
    jobs = run_reobs(a.n_proc, a.units) if a.part == "reobs" else run_rephot2024()
    log(f"rephot: {len(jobs)} frames")
    rows = []
    with Pool(a.n_proc) as pool:
        for i, r in enumerate(pool.imap_unordered(RP.process, jobs, chunksize=1), 1):
            rows.append(r)
            if i % 50 == 0:
                log(f"  {i}/{len(jobs)}")
    M = pd.DataFrame(rows)
    M.to_csv(OUT / f"frames_{a.part}.csv", index=False)
    log("rephot status: " + repr(M.status.str[:40].value_counts().to_dict()))


if __name__ == "__main__":
    main()
