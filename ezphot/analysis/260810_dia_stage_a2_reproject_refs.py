#!/usr/bin/env python
"""DIA Stage A2 — reproject the PS1 references onto the exact LOAO stack WCS.

Stage A found 1.3-2.4 pixel median offsets between the HiPS2FITS grids and
the stack grid, so each reference is interpolated (``reproject_interp``,
flux-conserving enough for a template that HOTPANTS rescales anyway) onto the
WCS of a real LOAO stack. All stacks share one grid, so this is one
reprojection per band. Outputs ``ref.PS1.<band>.aligned.fits`` beside the
originals; afterwards Stage A verification is rerun against the aligned files.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ezphot-matplotlib")

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

REF_DIR = Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/reference_images")
GRID_STACK = Path(
    "/home/hhchoi1022/ezphot/data/scidata/LOAO/LOAO_E2V_2x2/SN2026kid/"
    "1.0-m KASI/I105/stack.SN2026kid.2026_0610.I105.fits"
)


def main() -> None:
    grid_header = fits.getheader(GRID_STACK, memmap=False)
    grid_wcs = WCS(grid_header)
    shape = (int(grid_header["NAXIS2"]), int(grid_header["NAXIS1"]))
    for band in ("g", "r", "i"):
        source_path = REF_DIR / f"ref.PS1.{band}.fits"
        with fits.open(source_path, memmap=False) as hdul:
            aligned, footprint = reproject_interp(hdul[0], grid_wcs, shape_out=shape)
            source_header = hdul[0].header.copy()
        aligned = np.asarray(aligned, dtype=np.float32)
        aligned[footprint < 0.5] = np.nan
        header = grid_wcs.to_header()
        for key in ("SURVEY", "HIPSID", "FILTER", "OBJECT", "REFPXSCL", "REFQUTC"):
            if key in source_header:
                header[key] = (source_header[key], source_header.comments[key])
        header["REFALIGN"] = (True, "Reprojected onto the LOAO stack WCS")
        header["REFGRID"] = (GRID_STACK.name, "WCS source stack")
        output_path = REF_DIR / f"ref.PS1.{band}.aligned.fits"
        fits.writeto(output_path, aligned, header, overwrite=True, checksum=True)
        finite = float(np.mean(np.isfinite(aligned)))
        print(f"{band}: {output_path.name} finite_fraction={finite:.4f}")


if __name__ == "__main__":
    main()
