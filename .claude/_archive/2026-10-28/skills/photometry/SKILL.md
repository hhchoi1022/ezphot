---
name: photometry
description: Run 7DT/ezphot photometry with the ezphot.recipes entry points (process_images → qa → stack → dia → target_photometry). Use for "measure/process/photometry of <target>", light curves, per-image ZP checks. Encodes the lab conventions (10" = MAG_APER_3, GAIAXP_CORR_LAMOST refcat, conda env, NFS gotchas).
---

# ezphot photometry (recipes)

Always call the recipes; do not re-implement the chain from `ezphot.methods` in a new script.
Recipes live in `ezphot/recipes/` and are agent-agnostic. Read a recipe's docstring (`--help`) if unsure.

## Environment (every command)
```
cd /lyman/data1/factory_hhchoi/code/ezphot
MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python -m ezphot.recipes.<recipe> ...
```
- `MPLCONFIGDIR` is required: the home dir is NFS and matplotlib/portalocker hangs without it.
- Data root: `~/ezphot/data/scidata/<observatory>/<telkey>/<objname>/<telname>/<filter>/`.
  7DT: observatory `7DT`, telkey `7DT_C361K_HIGH_1x1`, telname `7DT01`..`7DT20`.
- Products are written next to the images by ezphot (`.cat`, `.refcat`, `.bkgmap`, `.bkgrms`, `.srcmask`).
  Summaries go to `~/ezphot/log/recipes/<objname>/`.

## Workflow
1. **process_images** – single-image chain (platesolve if needed → masks → bkg → SExtractor photometry → ZP calibration):
   ```
   ... -m ezphot.recipes.process_images --objname <OBJ> --observatory 7DT [--telname 7DT01] [--filter m400] \
       --ra <deg> --dec <deg> --n-proc 4 [--max-images N] [--out summary.csv]
   ```
   `--faint-shift 1.5` widens every filter's reference-star faint limit (more ZP stars, wider ZP figures).
   `--ra/--dec` = target position (used for a 150" circular source mask and for the local reference-star selection).
   Use `--paths f1.fits f2.fits` to bypass DataBrowser. Add `--preprocess` only for raw (non-BDF-corrected) frames; archived 7DT frames are already corrected.
2. **qa** – judge the summary; exit code 1 on any failure:
   ```
   ... -m ezphot.recipes.qa <summary.csv> [--zperr-max 0.1] [--nref-min 10] [--seeing-max 5] [--depth-min 17]
   ```
3. **stack** – per-night stacks (filter × observatory × obsdate ±0.5 d, all units together) of the successful rows of a
   process_images summary, then the same calibration chain on each stack (stacked bkgrms kept, no platesolve):
   ```
   ... -m ezphot.recipes.stack <process_images summary.csv> [more.csv] [--ra <deg> --dec <deg>] [--n-proc 4]
   ```
   Groups of one image are skipped (printed). Output rows have the process_images columns + `n_combined`, `inputs`.
4. **dia** – difference imaging against a reference frame (auto `get_referenceframe`, or `--reference ref.fits`) for the
   successful rows of process_images or stack summaries; with `--ra/--dec` also forced 10" photometry on the difference image:
   ```
   ... -m ezphot.recipes.dia <summary.csv> [more.csv] [--ra <deg> --dec <deg>] [--reference ref.fits] [--n-proc 4]
   ```
   Output: reference, subtracted image, `n_candidates`, `n_transients`, target `MAGSKY/MAGERR/ZPERR/UL5SKY_APER_3`.
   "no reference frame found" is not an error of the recipe (no reference covers the field before that date).
5. **target_photometry** – light-curve table (one row per catalog) for the nearest source to `--ra/--dec`:
   ```
   ... -m ezphot.recipes.target_photometry --objname <OBJ> --observatory 7DT --ra <deg> --dec <deg> [--filter m400] [--plot]
   ```

6. **visualize** – per-stage figure gallery for looking at the products by eye (no verdicts):
   ```
   ... -m ezphot.recipes.visualize <summary.csv> [--ra <deg> --dec <deg>] [--lightcurve <lc.csv>] [--stages ...]
   ```
   Writes `~/viz/ezphot/<objname>/viz/index.html`; served at `http://127.0.0.1:8765/ezphot/<objname>/viz/`
   (`python3 -m http.server 8765 --bind 127.0.0.1 --directory ~/viz`).
   Port rule: pages for the user use only 8760–8780 bound to 127.0.0.1; 8765 is the registered ezphot gallery
   port. Record any server start/stop in `/home/hhchoi1022/code/superagent/7DT_SN_ToO_project/ports.md`.

7. **status** – index of every objname with its process_images runs (running / done / stopped, success, qa, gallery link):
   `... -m ezphot.recipes.status [--watch 60]` writes `~/viz/ezphot/index.html` → `http://127.0.0.1:8765/ezphot/`.
   A watcher (`--watch 60`) normally runs detached; always pass `--out` under `~/ezphot/log/recipes/<objname>/`
   so runs show up there.

## Conventions and traps (do not violate)
- **10" aperture = `MAG_APER_3` / `ZP_APER_3` / `UL5SKY_APER_3` in ezphot catalogs** (aperture list 3,5,7,10,15 → index 3).
  The tract7dt pipeline uses `APER_5`/`ZP_5` for 10". Never mix. Never base analyses on the 3" aperture.
- Reference catalog: `GAIAXP_CORR_LAMOST` (Huang+2024 corrected Gaia XP). Per-filter reference magnitude ranges are in `ezphot/recipes/defaults.py` (`MAG_RANGE`).
- Keep ZP global (`ZP_APER_3` from the header) unless the user explicitly asks for a local ZP.
- Background policy (2026-10-21): recipes use the 2D bkgmap only (`annulus_width_arcsec=None`). SExtractor LOCAL (`annulus_width_arcsec` set) is allowed but is a mode estimate, biased low on the photon-skewed 7DT sky; `sex_photometry` prints a `[WARNING]` with the expected magnitude bias when it exceeds `local_bkg_bias_limit` (0.02 mag). Treat that warning as a QA failure for faint-star work; for a local background use `circular_photometry` (sigma-clipped mean annulus).
- hotpants: `/snap/bin/hotpants` (on PATH) cannot open files on the `/lyman` NFS (snap confinement) → every DIA fails
  with "FITSIO status = 104: could not open". ezphot uses `~/miniconda3/envs/too/bin/hotpants` (source build) when it exists.
- Pandas: `df.filter`, `df.flags`, `df.pivot` are methods; always index columns with `df['filter']`.
- `pkill -f`/`pgrep -f` match the tool shell itself; wait on log markers instead.
- Do not edit `ezphot/methods/*` to fix a recipe run. If a step is wrong, report it; parameter overrides go through the recipe's `config=` argument in Python:
  ```python
  from ezphot.recipes import process_images
  from ezphot.recipes.process_images import load_images
  imgs = load_images(objname='EG21', observatory='7DT', filter='m400')
  process_images(imgs, ra=..., dec=..., config={'photcal': {'mag_lower': 10, 'mag_upper': 14}})
  ```

## Fallbacks (one retry each, then report)
- `platesolve` failed → retry that image alone (`--paths`); if it fails again, exclude it and say so.
- `photometric_calibration` with too few sources → widen `mag_lower/mag_upper` by 0.5 mag via `config`, or raise `radius_arcsec`.
- qa fails only on `seeing`/`depth` → not an error; report those frames as low quality.

## Report format (what to return to the user / main session)
- counts: images found / succeeded / qa-passed, time taken
- table of failures: file, step, error (max 10 rows)
- summary CSV path and, for target_photometry, the table path (+ figure path)
- any fallback applied
