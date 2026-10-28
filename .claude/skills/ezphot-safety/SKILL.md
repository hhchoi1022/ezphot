---
name: ezphot-safety
description: "ezphot 코드나 레시피를 실행하기 전에 지켜야 할 안전 규칙과 판정 절차. ezphot은 import만으로 $HOME/ezphot/config를 쓰고, 처리 산출물·sidecar를 설정된 데이터 아카이브에 쓰며, 일부 API는 입력 FITS를 바꾸거나 지운다. 셸 명령 실행, 레시피(process_images·stack·dia·target_photometry·visualize·status) 실행 계획, 외부 바이너리(SExtractor·SCAMP·SWarp·PSFEx·hotpants·solve-field) 호출, 출력 경로 결정, 갤러리 웹 서버 실행 전에 반드시 사용한다. 코드를 읽기만 하는 작업에는 필요 없다."
---

# ezphot 안전 규칙

ezphot의 기본 설정은 실제 데이터 아카이브를 가리킨다. 처리 결과도 입력 옆이 아니라 설정된 아카이브 경로에 쓰므로,
**HOME을 격리하지 않은 실행 = 아카이브에 쓰는 실행**이다. 이 저장소 CLAUDE.md의 「환경」(env, 격리 HOME 위치, 포트)을 함께 따른다.
실행은 두 모드뿐이다: 기본인 **격리 모드**, 그리고 사용자가 대상 타일마다 승인한 측광 실행에만 쓰는 **실제 아카이브 모드**(3절).

## 1. 아카이브에 닿는 지점 (2026-10 코드로 확인)

| 대상 | 무엇이 일어나나 |
| --- | --- |
| `import ezphot` / `ezphot.initialize()` | `$HOME/ezphot/config`를 읽고, 없으면 만든다 |
| `config/common/*.config`의 `*_DIR` | scidata, refdata, calibdata, mcalibdata, obsdata, 로그 디렉터리. 처리 산출물·sidecar·summary 등록이 여기에 쓰인다 |
| `DataBrowser('scidata'|'refdata')` | 위 경로 검색. 격리 HOME에서는 격리 아카이브(비어 있음)를 본다 |
| `SkyCatalog` | `CATALOG_DIR`에 없는 영역이면 Vizier에서 받아 `CATALOG_DIR/<type>/`에 저장하고 `summary.ascii_fixed_width`를 갱신(`skycatalog/skycatalog.py:436-442`) |
| `PhotometryHelper.run_swarp(fill_zero_tonan=True)` | 입력 FITS를 update 모드로 열어 0을 NaN으로 바꾼다 |
| `BaseImage.rename()` | 대상 경로 파일을 먼저 지운다 |
| `write()` | 대개 overwrite |
| `remove()` | 본체와 연결된 sidecar까지 지운다 |
| Astrometry.net 래퍼 | solve-field 실패 시 입력을 지우는 in-place 경로가 있다(CODEBASE_GUIDE "Upstream issues") |
| reference·master 등록 | 공유 summary 파일 갱신(동시 쓰기 주의) |
| `helper/queryhelper.py` | 제3자 계정 자격증명이 소스에 들어 있다 |
| `utils/dbrowser.py` | GW portal(자격증명) 조회 |
| `update_pypl.sh` / `scan_dependency.py` | 버전 bump + PyPI 업로드 / `requirements.txt` 덮어쓰기 |

## 2. 절대 금지

- 격리 HOME 없이 ezphot을 import하는 실행(레시피, 테스트, 드라이버, `python -c "import ezphot"` 포함). 유일한 예외는 3절의 승인된 실제 아카이브 모드 레시피 실행이다.
  개발 검증(테스트·드라이버·baseline)은 어떤 경우에도 실제 아카이브 모드로 돌리지 않는다.
- 실제 아카이브(scidata, refdata, calibdata, mcalibdata, obsdata, 실제 `~/ezphot/log`)에 쓰거나 지우는 것 — 승인된 실제 아카이브 모드의 대상 타일 산출물은 예외. 어떤 모드에서도 지우지는 않는다.
- 원본 FITS에 in-place로 platesolve·SWarp(`fill_zero_tonan=True`)·rename·remove를 거는 것.
- `queryhelper` 자격증명, GW portal(`dbrowser`)을 쓰는 경로 실행. 자격증명 값을 출력·복사하는 것.
- `update_pypl.sh`, `scan_dependency.py` 실행. PyPI 등 외부로 무언가 올리는 것.
- 공유 Python env에 `pip install/uninstall`.
- `git push`, remote 추가, 커밋(사용자가 요청할 때만), `git stash push/pop/apply`·`checkout -- <file>`·`restore`·`reset`·브랜치 전환
  (작업 트리에 원래 있던 미커밋 변경이 사라진다). `git stash create`(작업 트리를 건드리지 않는 스냅샷)는 허용.
- 다른 Claude 세션에 SendMessage. 이번 실행에서 리더가 띄운 에이전트끼리만.

## 3. 조건부 허용

- **격리 HOME 실행**: `ezphot/tests/make_isolated_home.py <purpose>`가 만든 HOME에서. 출력의 `T=` 줄이 격리 루트(`<iso>`), 명령에는 `HOME=<iso>/home MPLCONFIGDIR=<iso>/mpl`.
  격리 HOME은 config의 데이터·로그 경로를 `<iso>/archive`, `<iso>/log`로 바꾼다. **예외: `CATALOG_DIR`은 실제 경로 그대로**다.
- **입력 FITS**: 실제 scidata의 FITS는 `--paths`로 경로만 넘긴다(읽기 전용, 산출물은 `<iso>/archive`). 격리 HOME에서는 DataBrowser 검색이 비므로 `--objname` 검색은 쓰지 않는다.
- **DIA reference**: 실제 refdata의 reference는 `--reference <경로>`로 넘긴다(읽기 전용). 격리 HOME에서는 `get_referenceframe`이 빈 refdata를 본다.
- **sky catalog 캐시**: 새 영역이면 실제 `CATALOG_DIR`에 파일이 생기고 Vizier(익명 공개 조회)에 접속한다. 격리 HOME 설계가 의도한 예외다.
  계획 단계에서 "캐시 갱신 가능"으로 표시하고, 실행 뒤 사후 검사의 `CATALOG` 줄을 보고서에 그대로 옮긴다.
- **외부 바이너리**: 작업 디렉터리가 격리 아카이브 안일 때만. 이름에 공백이 있는 경로(`1.0-m KASI` 등)는 SExtractor·SWarp·solve-field가 잘라 먹는다.
- **실제 아카이브 모드**: `ezphot-photometry`에서만. 조건은 모두 충족해야 한다.
  (1) 사용자가 **대상 타일(관측소·telkey·타일/objname·필터·날짜 범위)마다** 실제 아카이브 쓰기를 명시 승인했고 그 기록이 `_workspace/00_input/request.md`의 "## 승인"에 있다.
  (2) 실행은 레시피 CLI뿐(드라이버·임의 코드 금지), `HOME`을 바꾸지 않는다. (3) 실행 전 `post_run_check.sh --real-pre`로 입력 FITS 목록을 남기고, 실행 후 `--real`로 사라진 파일이 없는지 확인한다.
  (4) 같은 타일에 기존 산출물이 있으면 무엇이 덮어써지는지 계획서에 적는다. 이 모드에서는 산출물이 실제 아카이브·`~/ezphot/log/recipes`·`~/viz`에 남고, 8765 갤러리에 바로 보인다.
- **운영 격리 HOME의 자격증명**: `make_isolated_home.py`는 gwportal 자격증명도 복사한다. 그래서 격리 실행에서도 `dbrowser`(GW portal) 경로는 실제로 외부에 연결된다 — 계획에 넣지 않는다.
- **갤러리 웹 서버**(`visualize`·`status`의 `~/viz`): CLAUDE.md 「환경」의 포트 규칙(8760–8780, 고정 포트 제외, `127.0.0.1`)로만. 실제 HOME의 `~/viz`는 8765 상시 갤러리가 이미 서빙한다.
  격리 모드 결과를 볼 때만 별도 서버를 띄운다,
  예: `python -m http.server <port> --bind 127.0.0.1 --directory <iso>/home/viz`. 띄우기 전 `ss -ltnp`, 다 쓰면 내린다.

## 4. 명령 판정 절차

1. **HOME**: ezphot이 import될 수 있는가? 그렇다면 명령에 `HOME=<iso>/home`이 있는가? 없다면 실제 아카이브 모드 조건(3절 (1)–(4))을 모두 충족하는가?
2. **쓰기 경로**: `--out`, 레시피 기본값(`~/ezphot/log/recipes/<obj>/`, `~/viz`), 산출물 savedir(config `*_DIR`), 드라이버의 `write`/`savedir`를 따라가 `<iso>` 또는 `_workspace/` 안인지.
3. **입력 보존**: 입력 FITS에 update 모드 open, rename, remove, in-place platesolve가 일어나는가?
4. **연결**: 자격증명이 필요한 조회(queryhelper, dbrowser)가 있는가? Vizier 조회 가능성(캐시 갱신)은 표시만.
5. **자원**: 타일 하나 전체 처리 = 수십 GB·수십 분. 디스크 여유와 `--n-proc`를 계획에 적는다.
6. **되돌림**: 삭제·overwrite 대상이 `<iso>` 안인지 `ls`로 확인했는가?

판정은 `허용 / 조건부 허용(조건 명시) / 차단(근거 파일:줄)` 중 하나.

## 5. 사고 대응

실제 아카이브(특히 입력 타일 디렉터리, refdata, config)에 쓰기·삭제 흔적이 보이면 즉시 멈추고, 고치려 하지 말고(삭제도 쓰기다)
무엇이 언제 어디에 쓰였는지만 기록해 사용자에게 보고한다.
