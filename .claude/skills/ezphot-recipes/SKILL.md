---
name: ezphot-recipes
description: "ezphot 레시피(process_images → stack → dia → target_photometry → qa → visualize/status)로 천체 이미지를 격리 HOME에서 측광·스택·차영상 처리하는 실행 계획 작성법: 입력 FITS 고르기, reference 지정, 명령 구성, 자원·디스크, 산출물 위치, 갤러리 보기. 측광 실행, '○○ 측광해줘', '스택/DIA 돌려줘', '광도곡선 뽑아줘', 실행 계획 작성에 반드시 사용한다. 실행 전 판정은 ezphot-safety, 결과 판정은 photometry-qa 스킬."
---

# ezphot 레시피 실행

레시피는 ezphot 객체 위에 만든 CLI 워크플로다(`ezphot/recipes/`). 사람·에이전트·다른 프로젝트 누구나 같은 방식으로 부를 수 있게 만들어졌다.
기본은 **격리 HOME**에서 돌린다(아래 「실제 아카이브 모드」는 승인된 경우만). 격리 HOME에서는 config의 데이터 경로가 `<iso>/archive`로 바뀌므로 두 가지가 달라진다.
- `--objname`·`--filter` 같은 **DataBrowser 검색이 비어 있다** → 입력은 실제 FITS 경로를 `--paths`로 직접 준다(읽기만).
- **reference 자동 탐색(`get_referenceframe`)이 비어 있다** → DIA는 실제 reference 경로를 `--reference`로 준다(읽기만).

## 체인

```
process_images  (--paths FITS...)            → <out>.csv  (이미지당 1행: success, step, n_ref, SEEING, ZP_APER_3, UL5SKY_APER_3 ...)
   ↓  success 행
stack           (summary.csv ...)            → 밤·필터별 스택 + 같은 측광 체인 → stack summary.csv   (1장 그룹은 스택하지 않음)
   ↓
dia             (summary.csv ... --reference) → 차영상 + 차영상 카탈로그 + (ra,dec) forced photometry
   ↓
target_photometry (--paths .cat ... --ra --dec) → 대상 1개의 epoch·필터별 측광표 (SATDIST 경고 포함)
qa              (summary.csv)                 → 기준(ZPERR, n_ref, SEEING, depth) 판정, 실패 시 exit 1
visualize       (summary.csv)                 → 단계별 그림 HTML 갤러리 (~/viz)
status                                         → 모든 실행의 색인 페이지 (~/viz/ezphot/index.html)
```

각 레시피의 인자는 `python -m ezphot.recipes.<name> --help` 대신 소스 하단 `argparse`(예: `process_images.py:197-208`)를 읽어 확인한다
(`--help`도 ezphot을 import한다).

## 입력 고르기

1. 사용자에게 받은 대상(이름, RA/Dec, 필터, 날짜 범위)으로 **실제 scidata를 읽기 전용으로 탐색**한다:
   `find <SCIDATA_DIR>/<observatory>/<telkey>/<tile 또는 objname>/ -name '*.fits'`. SCIDATA_DIR은 실제 `~/ezphot/config/common/scidata.config`의 경로(읽기만).
2. 목록을 `<iso>/paths.txt`에 쓰고 개수·필터·날짜 분포를 계획서에 요약한다. 너무 많으면 사용자와 범위를 줄인다(`--max-images`는 시험용).
3. 입력 FITS가 있는 상위 디렉터리들을 계획서에 적는다 — 사후 검사(`post_run_check.sh --pre`)의 인자가 된다.
4. DIA면 reference 후보를 실제 refdata에서 찾아(`find <REFDATA_DIR> ...`) 경로를 계획서에 적는다. 여러 개면 사용자에게 고르게 한다.

## 명령 구성 (계획서에 그대로 쓴다)

```bash
bash .claude/skills/ezphot-verification/scripts/post_run_check.sh --pre <iso> <입력 디렉터리...>
HOME=<iso>/home MPLCONFIGDIR=<iso>/mpl $PY -m ezphot.recipes.process_images --paths $(cat <iso>/paths.txt) \
    --ra <RA> --dec <DEC> --n-proc <N> --out <iso>/home/ezphot/log/recipes/<obj>/process_images_<tag>.csv > <iso>/log/process_images.log 2>&1
HOME=<iso>/home MPLCONFIGDIR=<iso>/mpl $PY -m ezphot.recipes.qa <iso>/home/ezphot/log/recipes/<obj>/process_images_<tag>.csv
HOME=<iso>/home MPLCONFIGDIR=<iso>/mpl $PY -m ezphot.recipes.stack <process csv> --ra <RA> --dec <DEC> --n-proc <N> --out <...>/stack_<tag>.csv
HOME=<iso>/home MPLCONFIGDIR=<iso>/mpl $PY -m ezphot.recipes.dia <stack csv> --reference <실제 ref 경로> --ra <RA> --dec <DEC> --out <...>/dia_<tag>.csv
HOME=<iso>/home MPLCONFIGDIR=<iso>/mpl $PY -m ezphot.recipes.target_photometry --paths <iso>/archive/.../*.cat --ra <RA> --dec <DEC> --out <...>/target_<tag>.csv
HOME=<iso>/home MPLCONFIGDIR=<iso>/mpl $PY -m ezphot.recipes.visualize <process csv> --ra <RA> --dec <DEC> [--lightcurve <target csv>]
bash .claude/skills/ezphot-verification/scripts/post_run_check.sh <iso> <입력 디렉터리...>
```
- `--out`을 `<iso>/home/ezphot/log/recipes/<obj>/` 아래로 두면 `status`가 실행을 찾는다(레시피 기본값과 같은 위치).
- 단계마다 다음 단계 전에 qa와 로그를 본다. process_images 실패율이 높으면 멈추고 photometry-qa 판단을 받는다.
- 요청에 필요한 단계만 넣는다(측광만이면 process_images + qa + target_photometry).

## 자원

- `--n-proc` 기본 4(visualize는 8). 머신 부하(`uptime`, 다른 사용자 프로세스)를 보고 정한다.
- 7DT 타일 하나 전체 = 수십 GB(`<iso>/archive`), 수십 분~수 시간. 시작 전에 `df -h <iso>`를 계획서에 적는다.
- 오래 걸리는 단계는 백그라운드로 돌리고 로그를 본다. 같은 `<iso>`에 레시피를 동시에 두 개 돌리지 않는다(summary 등록 동시 쓰기).

## 산출물 위치와 보존

| 산출물 | 위치 |
| --- | --- |
| 처리된 이미지·sidecar·카탈로그 | `<iso>/archive/...` (config 경로 규칙 그대로) |
| summary CSV·로그 | `<iso>/home/ezphot/log/recipes/<obj>/`, `<iso>/log/` |
| 갤러리 | `<iso>/home/viz/` |

측광 실행에서는 산출물 자체가 결과물이다. 실행이 끝나도 `<iso>`를 **지우지 않고**, summary CSV·qa 출력·target 측광표·대표 그림만
`_workspace/runs/<obj>_<tag>/`에 복사한다. `<iso>` 삭제는 사용자가 결정한다(경로와 크기를 보고).

## 갤러리 보기

```bash
ss -ltnp | grep -E ':87([67][0-9]|80)\b'          # 빈 포트 확인 (8760–8780, 고정 8765·8766·8767 제외)
python3 -m http.server <port> --bind 127.0.0.1 --directory <iso>/home/viz   # 백그라운드
```
사용자에게 주소를 알리고, 다 보면 서버를 내린다. `status --watch`는 페이지만 다시 쓴다(서버 아님).
실제 아카이브 모드 결과는 실제 `~/viz`에 생기므로 8765 상시 갤러리(`http://127.0.0.1:8765/ezphot/<objname>/viz/`)에서 바로 보인다 — 새 서버를 띄우지 않는다.

## 실제 아카이브 모드 (승인된 경우만)

사용자가 대상 타일마다 실제 아카이브 쓰기를 승인하면(ezphot-safety 3절), 산출물을 실제 아카이브에 남기는 운영 처리로 돌린다.
다른 프로젝트가 그 산출물을 쓸 수 있게 하려는 것이다. 격리 모드와 다른 점:
- `HOME`을 바꾸지 않는다. DataBrowser 검색(`--objname` 등)과 `get_referenceframe`이 실제 아카이브를 본다 — `--paths`·`--reference`를 써도 된다.
- 산출물: 실제 아카이브(입력 이미지와 같은 규칙의 경로), `~/ezphot/log/recipes/<obj>/`, `~/viz/`.
- 실행 전: `bash .claude/skills/ezphot-verification/scripts/post_run_check.sh --real-pre _workspace/real/<obj>_<tag> <입력 디렉터리...>`.
  같은 타일의 기존 산출물(`.cat`, `.bkgmap`, `coadd_*` 등) 목록을 `ls`로 계획서에 남겨, 무엇이 덮어써지는지 보이게 한다.
- 실행 후: `... post_run_check.sh --real _workspace/real/<obj>_<tag> <입력 디렉터리...>` — `DELETED` 줄이 있으면 실패, 아무것도 지우거나 되돌리지 않고 보고.
- 실패한 단계를 다시 돌릴 때도 같은 승인 범위 안에서만. 범위를 넓히려면 다시 승인받는다.
