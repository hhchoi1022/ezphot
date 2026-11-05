"""
Progress sidecar of the recipe loops (process_images, stack, dia): <out>.progress.json next to the summary
CSV, rewritten by the parent process at the start and after every finished task (tmp + os.replace, so a
reader never sees half a file). Read by recipes.status. Internal format, may change:

    {"pid", "start", "updated", "total", "done", "ok", "failed", "end"}
    (times in unix seconds; ok = row['success']; end = None until run_tasks returns or raises)
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def progress_path(out) -> Path:
    return Path(f'{out}.progress.json')


def _write(path: Path, state: dict, warned: list):
    state['updated'] = time.time()
    tmp = Path(f'{path}.tmp')
    try:
        tmp.write_text(json.dumps(state))
        os.replace(tmp, path)
    except OSError as e:   # the progress file is only for status: report it (once per run) and keep processing
        if not warned:
            print(f"[progress] cannot write {path}: {e} (further errors not shown)", flush=True)
            warned.append(e)


def run_tasks(worker, tasks: list, n_proc: int, out) -> list:
    """[worker(t) for t in tasks] in input order, in a pool of min(n_proc, len(tasks)) processes when n_proc > 1
    and there is more than one task, as the recipes did with ex.map; <out>.progress.json counts finished tasks."""
    path = progress_path(out)
    state = dict(pid=os.getpid(), start=time.time(), updated=None, total=len(tasks), done=0, ok=0, failed=0, end=None)
    warned = []
    _write(path, state, warned)

    def finished(row):
        state['done'] += 1
        state['ok' if row.get('success') else 'failed'] += 1
        _write(path, state, warned)

    try:
        if n_proc > 1 and len(tasks) > 1:
            rows = [None] * len(tasks)
            with ProcessPoolExecutor(max_workers=min(n_proc, len(tasks))) as ex:
                futures = {ex.submit(worker, t): i for i, t in enumerate(tasks)}
                try:
                    for f in as_completed(futures):
                        rows[futures[f]] = f.result()   # BrokenProcessPool etc. propagate as with ex.map
                        finished(rows[futures[f]])
                except BaseException:
                    for f in futures:                   # as ex.map does: do not start the rest after an error
                        f.cancel()
                    raise
        else:
            rows = []
            for t in tasks:
                rows.append(worker(t))
                finished(rows[-1])
    finally:   # also on an error: a Python caller may live on, and status must not show the run as running
        state['end'] = time.time()
        _write(path, state, warned)
    return rows
