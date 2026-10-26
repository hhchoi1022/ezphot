#!/usr/bin/env python
"""
make_isolated_home: create an isolated ezphot HOME for tests / repros / verification, so that recipe runs never
write into the archive (~/ezphot/data, shared with BRIDGE).

    python tests/make_isolated_home.py <purpose> [--base /lyman/data1/factory_hhchoi/scratch/ezphot-iso]
    -> creates <base>/<purpose>_<YYYYmmdd_HHMMSS> (mode 700) and prints its path as T
    HOME=$T/home MPLCONFIGDIR=/tmp/ezphot-matplotlib python -m ezphot.recipes.process_images --paths <real fits> ...

$T/home/ezphot/config is a copy of ~/ezphot/config whose data and log dirs point to $T/archive and $T/log.
CATALOG_DIR, the SExtractor/SCAMP/SWarp/PSFEx parameter dirs and ~/.astropy stay real (read; the catalog cache
may gain new files). Inputs given with --paths are only read; products land under $T/archive.
Afterwards `find <real tile dirs> ~/ezphot/data/refdata -newer $T/marker` must print nothing.
$T holds a copy of the gwportal credentials and the products (~60 GB per baseline tile): delete it when done.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

BASE = Path('/lyman/data1/factory_hhchoi/scratch/ezphot-iso')


def make_isolated_home(purpose: str, base: Path = BASE) -> Path:
    real_home = Path.home()
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    base.chmod(0o700)
    root = base / f"{purpose}_{time.strftime('%Y%m%d_%H%M%S')}"
    root.mkdir(mode=0o700)
    cfg = root / 'home' / 'ezphot' / 'config'
    shutil.copytree(real_home / 'ezphot' / 'config', cfg, symlinks=True)

    redirect = {str(real_home / 'ezphot' / 'data'): root / 'archive', str(real_home / 'ezphot' / 'log'): root / 'log'}
    for f in sorted((cfg / 'common').glob('*.config')):
        conf = json.loads(f.read_text())
        changed = False
        for key, val in conf.items():
            if not isinstance(val, str) or key == 'CATALOG_DIR':
                continue
            for src, dst in redirect.items():
                if val == src or val.startswith(src + '/'):
                    conf[key] = str(dst) + val[len(src):]
                    changed = True
        if changed:
            f.write_text(json.dumps(conf, indent=4))

    # refuse to hand out a HOME that still points into the real data/log dirs
    for f in (cfg / 'common').glob('*.config'):
        for key, val in json.loads(f.read_text()).items():
            if key != 'CATALOG_DIR' and isinstance(val, str) and val.startswith(tuple(redirect)):
                raise RuntimeError(f'{f.name}: {key} still points to {val}')

    if (real_home / '.astropy').exists():
        (root / 'home' / '.astropy').symlink_to(real_home / '.astropy')
    (root / 'marker').touch()
    return root


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('purpose', help='short label, e.g. verify, repro, runner')
    ap.add_argument('--base', type=Path, default=BASE)
    a = ap.parse_args(argv)
    root = make_isolated_home(a.purpose, a.base)
    print(f'T={root}')
    print(f'HOME={root}/home')


if __name__ == '__main__':
    main()
