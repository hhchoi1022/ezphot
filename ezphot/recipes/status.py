#!/usr/bin/env python
"""
status: one index page of every objname processed with the recipes, with the state of each
process_images run (running / done / stopped), success and qa counts, and the gallery link.
Read-only on the logs; writes only <root>/ezphot/index.html.

    python -m ezphot.recipes.status [--root DIR] [--watch SEC]

Runs are found as ~/ezphot/log/recipes/<objname>/process_images_<tag>.{log,csv}. A run is
'running' while a live process_images process on this host has --out <that csv>, 'done' when the
csv exists, 'stopped' when only the log is left, and 'unreadable' when its csv/log cannot be read.
Runs whose csv is written elsewhere are not listed; runs without --out (e.g. the Python API) appear
once their csv lands there but are never shown as running. --watch rewrites the page every SEC seconds.
Served by the gallery server: http://127.0.0.1:8765/ezphot/
"""
from __future__ import annotations

import argparse
import html
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from ezphot.recipes.qa import qa
from ezphot.recipes.visualize import VIZ_ROOT

LOG_ROOT = Path.home() / 'ezphot' / 'log' / 'recipes'
STATE_LABEL = {'running': '실행 중', 'done': '완료', 'stopped': '중단', 'unreadable': '읽기 실패'}


def _running_outs() -> dict:
    """{--out path: number of --paths or None} of live process_images processes on this host
    (read from /proc, so no pgrep self-match; relative --out resolved against the process cwd)."""
    outs = dict()
    for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = [a.decode(errors='replace') for a in cmdline.read_bytes().split(b'\0')]
            cwd = (cmdline.parent / 'cwd').resolve()
        except OSError:
            continue
        if 'ezphot.recipes.process_images' not in args:
            continue
        out = next((a.split('=', 1)[1] for a in args if a.startswith('--out=')), None)
        if out is None and '--out' in args[:-1]:
            out = args[args.index('--out') + 1]
        if not out:
            continue
        n_paths = None
        if '--paths' in args:
            rest = args[args.index('--paths') + 1:]
            n_paths = next((i for i, a in enumerate(rest) if a.startswith('--') or a == ''), len(rest))
        outs[str((cwd / Path(out).expanduser()).resolve())] = n_paths
    return outs


def _run(stem: Path, running: dict) -> dict:
    csv, log = Path(f'{stem}.csv'), Path(f'{stem}.log')   # tags may contain dots (faint1.5)
    text = log.read_text(errors='replace') if log.exists() else ''
    m_n = re.search(r'\[process_images\] (\d+) images\n', text)
    m_ok = re.search(r'\[process_images\] (\d+)/(\d+) images succeeded in (\d+)s', text)
    run = {'tag': stem.name[len('process_images_'):], 'csv': str(csv) if csv.exists() else '',
           'n': int(m_n.group(1)) if m_n else None, 'ok': None, 'qa': None,
           'sec': int(m_ok.group(3)) if m_ok else None,
           'mtime': max(p.stat().st_mtime for p in (csv, log) if p.exists())}
    if str(csv.resolve()) in running:
        run['state'] = 'running'
        if run['n'] is None:   # the log is block-buffered; count --paths instead
            run['n'] = running[str(csv.resolve())]
    elif csv.exists():
        run['state'] = 'done'
        df = qa(str(csv))
        run.update(n=len(df), ok=int(df['chk_success'].sum()), qa=int(df['qa_pass'].sum()))
    else:
        run['state'] = 'stopped'
    return run


def _safe_run(stem: Path, running: dict) -> dict:
    """One unreadable csv/log (half-written, empty, wrong columns) must not stop the watcher."""
    try:
        return _run(stem, running)
    except Exception as e:
        mtimes = [p.stat().st_mtime for p in (Path(f'{stem}.csv'), Path(f'{stem}.log')) if p.exists()]
        return {'tag': stem.name[len('process_images_'):], 'csv': f'{type(e).__name__}: {e}', 'state': 'unreadable',
                'n': None, 'ok': None, 'qa': None, 'sec': None, 'mtime': max(mtimes, default=0)}


def collect(log_root: Path = LOG_ROOT, viz_root: Path = VIZ_ROOT) -> list:
    if not log_root.exists():
        return []
    running = _running_outs()
    objects = []
    for objdir in sorted(p for p in log_root.iterdir() if p.is_dir()):
        stems = {p.parent / p.stem for p in objdir.glob('process_images_*') if p.suffix in ('.csv', '.log')}
        runs = sorted((_safe_run(s, running) for s in stems), key=lambda r: r['mtime'], reverse=True)
        if not runs:
            continue
        gallery = viz_root / 'ezphot' / objdir.name / 'viz' / 'index.html'
        objects.append({'objname': objdir.name, 'runs': runs, 'gallery': gallery.exists()})
    return sorted(objects, key=lambda o: o['runs'][0]['mtime'], reverse=True)


def _t(ts):
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts else '–'


def _cells(r: dict) -> str:
    n = '–' if r['n'] is None else r['n']
    ok = '–' if r['ok'] is None else f"{r['ok']}/{n}"
    q = '–' if r['qa'] is None else f"{r['qa']}/{n}"
    bad = r['state'] == 'done' and (r['ok'] < r['n'] or r['qa'] < r['n'])
    sec = '–' if r['sec'] is None else f"{r['sec']}s"
    return (f'<td class="st {r["state"]}">{STATE_LABEL[r["state"]]}</td><td>{n}</td>'
            f'<td class="{"bad" if bad else ""}">{ok}</td><td class="{"bad" if bad else ""}">{q}</td>'
            f'<td>{sec}</td><td>{_t(r["mtime"])}</td>')


def render(objects: list) -> str:
    body = []
    for o in objects:
        last = o['runs'][0]
        link = f'<a href="{html.escape(quote(o["objname"]))}/viz/">갤러리</a>' if o['gallery'] else '<span class="mute">없음</span>'
        runs = ''.join(f'<tr><td>{html.escape(r["tag"])}</td>{_cells(r)}<td class="path">{html.escape(r["csv"])}</td></tr>'
                       for r in o['runs'])
        body.append(f'<tr class="obj"><td><b>{html.escape(o["objname"])}</b> <span class="path">{html.escape(last["tag"])}</span></td>{_cells(last)}<td>{link}</td>'
                    f'<td>{len(o["runs"])}</td></tr>'
                    f'<tr class="sub"><td colspan="9"><details><summary>실행 {len(o["runs"])}건</summary>'
                    f'<table><thead><tr><th>실행</th><th>상태</th><th>이미지</th><th>성공</th><th>qa 통과</th>'
                    f'<th>소요</th><th>최종 갱신</th><th>summary</th></tr></thead><tbody>{runs}</tbody></table></details></td></tr>')
    n_run = sum(r['state'] == 'running' for o in objects for r in o['runs'])
    return PAGE.format(now=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), n_obj=len(objects), n_run=n_run,
                       rows=''.join(body) or '<tr><td colspan="9" class="mute">기록 없음</td></tr>')


PAGE = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ezphot 처리 현황</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--fg:#1b1d21;--mute:#667085;--line:#e3e6ea;--acc:#2f6fde;--bad:#c4372f;--run:#b26b00}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121417;--card:#1b1e23;--fg:#e6e8eb;--mute:#98a1ae;--line:#2c3138;--acc:#7aa7ff;--bad:#ff7b72;--run:#f0b44c}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Apple SD Gothic Neo",sans-serif}}
main{{padding:16px;max-width:1300px;margin:0 auto}}h1{{font-size:18px;margin:0 0 2px}}.mute,.meta{{color:var(--mute)}}.meta{{font-size:12px;margin-bottom:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}}
th,td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{color:var(--mute);font-weight:500}}
td:first-child,th:first-child,td.st,td.path{{text-align:left}}tr.sub>td{{padding:0 8px 6px 24px;border-bottom:1px solid var(--line)}}
tr.obj>td{{border-bottom:0}}details summary{{color:var(--mute);font-size:12px;cursor:pointer}}details table{{margin:4px 0 6px}}
.path{{color:var(--mute);font-size:12px}}.bad{{color:var(--bad)}}.st.running{{color:var(--run);font-weight:600}}.st.stopped{{color:var(--mute)}}.st.unreadable{{color:var(--bad);font-weight:600}}
a{{color:var(--acc)}}
</style></head><body><main>
<h1>ezphot 처리 현황</h1>
<div class="meta">{now} 갱신 · 대상 {n_obj}개 · 실행 중 {n_run}건 · 60초마다 새로고침 · 성공/qa는 기본 기준(zperr≤0.1, n_ref≥10, seeing≤5)</div>
<div class="card"><table><thead><tr><th>objname</th><th>최근 상태</th><th>이미지</th><th>성공</th><th>qa 통과</th>
<th>소요</th><th>최종 갱신</th><th>갤러리</th><th>실행 수</th></tr></thead><tbody>{rows}</tbody></table></div>
</main></body></html>'''


def write_status(root: Path | str = None) -> Path:
    root = Path(root or VIZ_ROOT).expanduser()
    out = root / 'ezphot' / 'index.html'
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(render(collect(viz_root=root)))
    tmp.replace(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', help=f'gallery root served over http (default {VIZ_ROOT})')
    ap.add_argument('--watch', type=float, help='rewrite the page every WATCH seconds')
    a = ap.parse_args(argv)
    if not a.watch:
        print(f"[status] -> {write_status(a.root)}")
        return 0
    while True:
        try:
            write_status(a.root)
        except Exception:   # keep the always-on watcher alive; the error goes to its log
            traceback.print_exc()
        time.sleep(a.watch)


if __name__ == '__main__':
    raise SystemExit(main())
