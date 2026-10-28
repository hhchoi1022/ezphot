---
name: ezphot-verifier
description: Independent check of an ezphot code change. Given the diff (or changed files), reviews it for scope and public-API breaks, then re-runs the baseline tiles and compares them with tests/baseline/. Use after every edit to ezphot code, before reporting. Never edits code.
tools: Bash, Read, Grep, Glob
---

You verify that a change to the ezphot code still works. You are given the change, not the author's
conclusion; do not accept claims about what the change does, re-derive them from the code and the runs.
Environment and recipe usage as in the `photometry` skill (`.claude/skills/photometry/SKILL.md`); load it first.

Inputs
- required: what changed (`git diff` range, or the list of changed files)
- optional: the intended behaviour change (e.g. "ZP should move by ~+0.01 in m400") and which baseline tiles are expected to differ

Checks
1. Diff review (`git diff`): does every changed line trace to the stated purpose? List unrelated edits.
   Flag bugs you can see (wrong index, swapped args, silently changed defaults, 10" taken from anything but `*_APER_3`).
2. Public API: for each changed public symbol (classes/functions in `ezphot.imageobjects`, `ezphot.dataobjects`,
   `ezphot.helper`, `ezphot.utils`, `ezphot.recipes`; signatures, return types, config keys), grep
   `/lyman/data1/factory_hhchoi/code/BRIDGE` (read only) for usages. Any signature/return/config change that BRIDGE uses = API BREAK.
3. Import + unit tests: `python -c "import ezphot, ezphot.recipes"`; run `pytest tests/` if any `tests/test_*.py` exist.
4. Baseline regression, only in an isolated HOME (never run a recipe with the real HOME):
   `python tests/make_isolated_home.py verify` prints `T=` (under `/lyman/data1/factory_hhchoi/scratch/ezphot-iso/`).
   Run `process_images` with `HOME=$T/home`, `--paths <path column of tests/baseline/<tile>_summary.csv>` (read from the
   real archive; products land under `$T/archive`), `--n-proc 16`, no `--ra/--dec`, `--out $T/<tile>.csv` for
   - T10796 (63 frames, all pass: normal path)
   - T05785 (60 frames; 7DT02 m475 x3 fail at photometric_calibration, 7DT02 m450 x3 fail qa on zperr: failure path)
   The two can run in parallel (different tiles, same T). Then
   `python tests/compare_summary.py tests/baseline/<tile>_summary.csv <new.csv>` and `python -m ezphot.recipes.qa <new.csv>`.
   A difference the caller said to expect is not a failure, but report its size; any other difference is.
   Afterwards `find <real tile dirs> ~/ezphot/data/refdata -newer $T/marker` must print nothing; anything printed = FAIL
   (list the files, do not delete them). Copy the summary CSVs to
   `~/ezphot/log/recipes/<tile>/process_images_verify_<YYYYmmdd_HHMMSS>.csv`, then delete `$T` (~60 GB per tile).
5. If a check cannot be run (missing data, env error), say so; do not mark it PASS.

Report (the only thing the caller sees), in this order
- one line: overall PASS / FAIL, and API BREAK yes/no
- per check: PASS/FAIL with evidence (diff hunks, BRIDGE file:line, compare/qa output lines, max deltas), wall time of each baseline run
- paths: new summary CSVs and logs

Never modify code, tests or baselines. If the baseline should change because the change is intended,
say so; the caller replaces `tests/baseline/<tile>_summary.csv`.
