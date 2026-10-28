---
name: ezphot-verification
description: "ezphot 코드 변경을 단계별(정적 → 깨끗한 import → 합성 데이터 단위 드라이버 → baseline 레시피 회귀)로 검증하는 절차, 작업 전 스냅샷과 변경분 diff, 격리 HOME 생성·정리, 실행 전후 아카이브 무결성 검사. 변경 검증, 회귀 확인, baseline 비교, 검증 계획 작성, '엔지니어가 바꾼 것만 보기'에 반드시 사용한다. 측광 실행 자체는 ezphot-recipes, 실행 전 안전 판정은 ezphot-safety가 맡는다."
---

# ezphot 검증

ezphot에는 회귀 테스트 스위트가 없다. 있는 것은 baseline 2개(`tests/baseline/T05785_summary.csv`, `T10796_summary.csv`)와
`tests/compare_summary.py`, 격리 HOME 생성기 `tests/make_isolated_home.py`뿐이다. 그래서 검증은 **싼 것부터, 필요한 만큼만** 올린다.
baseline 회귀는 타일당 수십 GB·수십 분이라 가벼운 단계가 통과하기 전에 돌리지 않는다.

```
S=.claude/skills/ezphot-verification/scripts        # ezphot 저장소 루트 기준
PY=<공유 env의 python 절대경로>                       # CLAUDE.md 「환경」의 env(`too`). `python`만 쓰지 않는다
```
셸 상태가 다음 명령으로 이어지지 않을 수 있으므로 `<iso>`와 `PY`는 매 명령에 리터럴로 쓴다.

## 스냅샷과 변경분

ezphot 작업 트리에는 원래 미커밋 변경이 있다(2026-10-28 기준 수정 6개, 삭제 6개, 미추적 160여 개). 작업 트리 전체 `git diff`는 이것까지 보여 준다.
```bash
bash $S/snapshot.sh _workspace            # 엔지니어를 띄우기 전에 한 번
bash $S/change_diff.sh _workspace         # 엔지니어가 만든 변경만
bash $S/change_diff.sh _workspace --names
```
부분 재실행이면 기존 `00_baseline.txt`를 그대로 쓴다.

## 단계와 선택 기준

| 단계 | 내용 | 실행 조건 | 격리 HOME | 승인 |
| --- | --- | --- | --- | --- |
| **V0 정적** | 바뀐 `.py`만 `ast.parse`; 호출 지점 grep | 항상 | 불필요 | - |
| **V1 import** | 깨끗한 인터프리터에서 바뀐 모듈과 `ezphot.imageobjects/methods/dataobjects/utils/recipes` import | 모든 코드 변경 | 필요 | - |
| **V2 단위** | 감사된 드라이버(`_workspace/verify/*.py`): 작은 배열·합성 헤더·작은 테이블로 바뀐 함수 직접 호출, 외부 바이너리는 `PhotometryHelper.run_command` mock | 함수 동작이 바뀔 때 | 필요 | - |
| **V3 baseline** | baseline 타일 FITS로 `process_images` → `compare_summary.py` | 측광·캘리브레이션·배경·마스크·좌표 수치가 바뀔 수 있을 때(ezphot-api-policy "수치 변경") | 필요 | **매번 사용자 승인** |

DIA·stack을 바꿨으면 V3에 `stack`/`dia` 레시피를 이어 붙인다(baseline이 없으므로 변경 전후 두 번 실행해 비교 — 비용 2배, 승인 시 명시).

## V0 정적
```bash
bash $S/change_diff.sh _workspace --names | grep '\.py$' \
  | xargs -r $PY -c "import ast,sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]; print('ast ok')"
```
바뀐 파일만 검사한다(`ezphot/analysis/`에는 원래 파싱되지 않는 파일이 9개 있다). `py_compile`은 `__pycache__`를 쓰므로 쓰지 않는다.

## 격리 HOME (V1 이상)
```bash
$PY tests/make_isolated_home.py <purpose>          # 출력 "T=<경로>" 줄의 경로가 <iso>. 기본 위치는 스크립트의 `BASE`(`/lyman/data1/factory_hhchoi/scratch/ezphot-iso/`)
ls <iso>                                            # home/ archive/ log/ marker
grep -h '"' <iso>/home/ezphot/config/common/*.config | grep -v CATALOG_DIR \
  | grep -E "$(getent passwd $(id -un) | cut -d: -f6)/ezphot/|/data/ezphot/" && echo "STOP: 실제 경로가 남아 있음"
```
마지막 grep은 아무것도 출력하지 않아야 한다. `CATALOG_DIR`만 의도적으로 실제 경로다(ezphot-safety 3절).

## V1 import
```bash
HOME=<iso>/home MPLCONFIGDIR=<iso>/mpl $PY -c "import ezphot.imageobjects, ezphot.methods, ezphot.dataobjects, ezphot.utils, ezphot.recipes; import <바뀐 모듈>; print('import ok')"
```
`update_new_transmission.py`처럼 import 때 예제를 실행하는 모듈은 넣지 않는다.

## V2 단위
드라이버는 리더가 `_workspace/verify/<name>.py`에 쓰고 ezphot-safety-auditor가 판정한다. 원칙:
- 실제 FITS 대신 `numpy` 배열·`astropy.io.fits` 합성 헤더·작은 `Table`. 실제 FITS가 꼭 필요하면 `<iso>`로 **복사**해서 쓴다.
- 외부 바이너리는 `PhotometryHelper.run_command`를 mock(CODEBASE_GUIDE "Suggested verification strategy").
- 산출물은 `<iso>` 아래로. 기대값을 assert하고 종료 코드로 알린다.
```bash
HOME=<iso>/home MPLCONFIGDIR=<iso>/mpl $PY _workspace/verify/<name>.py
```

## V3 baseline (승인 필요)
```bash
bash $S/post_run_check.sh --pre <iso> <입력 타일 디렉터리>            # 입력 파일 목록 기록 (삭제 감지용)
cut -d, -f1 tests/baseline/T05785_summary.csv | tail -n +2 > <iso>/paths.txt
HOME=<iso>/home MPLCONFIGDIR=<iso>/mpl $PY -m ezphot.recipes.process_images \
  --paths $(cat <iso>/paths.txt) --n-proc 4 --out <iso>/T05785_summary.csv > <iso>/process.log 2>&1
$PY tests/compare_summary.py tests/baseline/T05785_summary.csv <iso>/T05785_summary.csv
```
입력 타일 디렉터리 = `paths.txt`에 나온 FITS들의 상위 디렉터리(필터 디렉터리들). 허용오차는 run-to-run 산포에 맞춘 값이다. **허용오차를 넓혀 통과시키지 않는다.**
의도한 수치 변화면 어느 열이 얼마나, 왜 바뀌었는지 보고하고 baseline 갱신은 사용자에게 넘긴다(baseline 파일을 고치지 않는다).
`compare_summary.py`는 `ezphot.recipes.defaults`를 import하므로 이것도 `HOME=<iso>/home`을 붙이는 게 안전하다.

## 사후 무결성 검사 (V1 이상이면 항상)
```bash
bash $S/post_run_check.sh <iso> [입력 타일 디렉터리 ...]
```
- **엄격**(실제 config, refdata, calibdata, mcalibdata, obsdata, 입력 디렉터리; `--pre`를 했으면 입력 파일 변경·삭제): 접두어 없는 줄·`CHANGED-OR-DELETED` 줄이 있으면(exit 1) **실패**. ezphot-safety 5절, 아무것도 지우지 않는다.
- **CATALOG**: 실제 sky catalog 캐시에 생긴 파일. 실패는 아니지만 보고서에 그대로 옮긴다.
- **INFO**: 실제 `~/ezphot/log`의 새 파일. 다른 프로세스가 쓸 수 있으니 이번 실행과 관련 있는지만 확인.

## 정리
요약 CSV·비교 출력·로그를 `_workspace/04_artifacts/`로 옮긴 뒤 `<iso>`를 지운다. 지우기 전에 `readlink -f <iso>`가 격리 기본 디렉터리 아래인지 확인한다.

## 보고 형식
| 단계 | 명령 | 결과(pass/fail/skip/미실행) | 근거(출력 요약, 로그 경로) |

미실행 단계는 이유(불필요/승인 대기/거절)를 적는다. 실패를 고쳐 다시 돌렸다면 첫 실패도 남긴다.
