"""Validate solved LOAO WCS headers and make an astrometry QC figure."""
from pathlib import Path
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import matplotlib.pyplot as plt

REPORT = Path('/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/astrometry_all_obsdates_report.json')
OUT = Path('/home/hhchoi1022/ezphot/log/20260805_223146_codex_run')
REF = Path('/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/reference.APASS_DR9_BVRI.ecsv')

def main():
    d = json.loads(REPORT.read_text())
    bad = []
    scales, offsets, methods = [], [], {}
    for r in d['results']:
        p = Path(r['path'])
        try:
            with fits.open(p, checksum=True, memmap=False) as hdul:
                hdul.verify('exception')
                if hdul[0].verify_checksum() != 1 or hdul[0].verify_datasum() != 1:
                    raise RuntimeError('checksum failed')
                h = hdul[0].header
                if not bool(h.get('ASTRMCOR', False)):
                    raise RuntimeError('ASTRMCOR is not true')
                w = WCS(h)
                if not w.has_celestial:
                    raise RuntimeError('no celestial WCS')
                cd = w.pixel_scale_matrix
                scale = float(np.sqrt(abs(np.linalg.det(cd))) * 3600.0)
                if not (0.25 < scale < 1.1):
                    raise RuntimeError(f'unexpected scale {scale:.3f}')
                scales.append(scale)
                offsets.append(float(r.get('pointing_separation_arcsec', np.nan)))
                methods[r.get('method', 'unknown')] = methods.get(r.get('method', 'unknown'), 0) + 1
        except Exception as e:
            bad.append((str(p), str(e)))
    summary = {'images': len(d['results']), 'report_failures': len(d['failures']),
               'header_failures': len(bad), 'methods': methods,
               'scale_arcsec_pix': [float(np.nanmin(scales)), float(np.nanmedian(scales)), float(np.nanmax(scales))],
               'pointing_offset_arcsec': [float(np.nanmin(offsets)), float(np.nanmedian(offsets)), float(np.nanmax(offsets))]}
    log = OUT / '2_astrometry_validation.log'
    log.write_text(json.dumps(summary, indent=2) + '\n' + '\n'.join(f'{p}: {e}' for p, e in bad) + '\n')

    ref = Table.read(REF)
    rac = next(c for c in ref.colnames if c.lower() in ('ra', 'raj2000'))
    decc = next(c for c in ref.colnames if c.lower() in ('dec', 'dej2000'))
    coords = SkyCoord(np.asarray(ref[rac], float) * u.deg, np.asarray(ref[decc], float) * u.deg)
    choices = [r for r in d['results'] if r.get('method') in ('SCAMP', 'astrometry.net')][:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9)); axes = axes.ravel()
    for ax, r in zip(axes, choices):
        p = Path(r['path'])
        with fits.open(p, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, float); w = WCS(hdul[0].header)
        lo, hi = np.nanpercentile(data, [2, 98])
        ax.imshow(data, origin='lower', cmap='gray', vmin=lo, vmax=hi)
        x, y = w.world_to_pixel(coords)
        m = (x >= 0) & (x < data.shape[1]) & (y >= 0) & (y < data.shape[0])
        ax.scatter(x[m], y[m], s=25, facecolors='none', edgecolors='red', linewidths=.7)
        ax.set_title(f"{r['detector']} {r['filter']} {r['night']}\\n{r.get('method')} scale={r.get('pixel_scale_arcsec', np.nan):.3f} arcsec/pix")
        ax.set_xlim(0, data.shape[1]); ax.set_ylim(0, data.shape[0]); ax.set_xlabel('x'); ax.set_ylabel('y')
    for ax in axes[len(choices):]: ax.axis('off')
    fig.tight_layout(); fig.savefig(OUT / '2_astrometry_qc.png', dpi=150); plt.close(fig)
    print(json.dumps(summary, indent=2)); print(f'QC: {OUT / "2_astrometry_qc.png"}')
    if bad or d['failures']:
        raise SystemExit(1)

if __name__ == '__main__': main()
