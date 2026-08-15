"""Re-solve frames previously handled by SCAMP using astrometry.net only."""
from pathlib import Path
import importlib.util, json
from astropy.io import fits

REPORT = Path('/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/astrometry_all_obsdates_report.json')
PHOT = Path(__file__).with_name('260805_photometry_loao_sn2026kid.py')
spec = importlib.util.spec_from_file_location('phot', PHOT)
phot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phot)

def main():
    report = json.loads(REPORT.read_text())
    targets = [r for r in report['results'] if r.get('method') == 'SCAMP']
    reference = phot.Table.read(report['reference_catalog'], format='ascii.ecsv')
    recovered = []
    for row in targets:
        path = Path(row['path']); telescope = phot.telinfo(path)
        image, result = phot.solve_astrometry_safe(path, telescope, reference, overwrite=True, verbose=False)
        phot.verify_checksum(path)
        if result.get('method') != 'astrometry.net':
            raise RuntimeError(f'non-astrometry.net fallback for {path}: {result}')
        pointing = phot.SkyCoord(image.ra, image.dec, unit='deg')
        metrics = phot.wcs_metrics(path, pointing)
        new = dict(row); new.update(status='solved', method='astrometry.net', **metrics)
        recovered.append(new)
        print(f"astrometry.net OK {path.name} | scale={metrics['pixel_scale_arcsec']:.4f}")
    by_path = {r['path']: r for r in recovered}
    report['results'] = [by_path.get(r['path'], r) for r in report['results']]
    report['summary']['methods'] = {}
    for r in report['results']:
        method = r.get('method', 'unknown')
        report['summary']['methods'][method] = report['summary']['methods'].get(method, 0) + 1
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(f'replaced {len(recovered)} SCAMP rows; astrometry.net only')

if __name__ == '__main__':
    main()
