# ezphot

천문 FITS 처리·측광 라이브러리. 여러 프로젝트가 가져다 쓰는 독립 라이브러리이므로, 특정 사용처를 근거로 설계하지 않고
공개 API가 바뀌면 `CHANGELOG.md`와 버전으로 알린다. 코드 지도는 `CODEBASE_GUIDE.md`, CLI 워크플로는 `ezphot/recipes/`.

## 환경
- Python: `/home/hhchoi1022/miniconda3/envs/too/bin/python` (운영 env `too`). `python`만 쓰지 않는다.
  이 env는 ezphot·BRIDGE·NGSF-7DT 저장소를 **editable**로 가리킨다 — 코드를 고치면 다음 import부터 운영 프로세스에 반영된다.
- 격리 HOME: `ezphot/tests/make_isolated_home.py <purpose>` → `/lyman/data1/factory_hhchoi/scratch/ezphot-iso/<purpose>_<stamp>`.
  운영판은 gwportal·ATLAS 자격증명(`gwportal.config`·`atlas.config`)도 복사하므로, 쓰고 나면 지운다.
- 웹페이지 포트: **8760–8780**, `127.0.0.1`에만 바인딩(SSH 로컬 포트포워딩으로 접속). 고정 할당 `8765` ezphot 갤러리(상시), `8766` ngsf-qa, `8767` BRIDGE opsmonitor —
  다른 용도로 쓰지 않는다. 띄우기 전 `ss -ltnp | grep -E ':87([67][0-9]|80)\b'`로 확인하고, 보고할 때 포트·URL·내용을 알리고, 일회성 페이지는 다 쓰면 내린다.
  포트 기록부는 pixel-agents가 관리하는 `~/.pixel-agents/ports.md`(읽기만). 상시 포트를 추가·변경·폐지할 때는 superagent를 통해 pixelagents에 요청하고,
  일회성 페이지는 자동으로 감지되므로 등록하지 않는다.
- 측광 실행 모드: 기본은 격리 HOME. 실제 HOME으로 돌려 아카이브에 쓰는 실행은 운영 처리이며 **대상 타일마다 사용자 승인**을 받는다(`ezphot-safety` 3절).

## 안전 (요약 — 판정 절차는 `ezphot-safety` 스킬)
- ezphot은 import만으로 `$HOME/ezphot/config`를 쓰고, 처리 산출물을 config의 데이터 아카이브 경로에 쓴다.
  **ezphot을 import하는 모든 실행(레시피·테스트·드라이버)은 `tests/make_isolated_home.py`로 만든 격리 HOME에서** 한다.
  예외는 사용자가 대상 타일마다 승인한 측광 실행(실제 아카이브 모드)뿐이고, 개발 검증은 언제나 격리 HOME이다.
- 실제 아카이브(scidata, refdata, calibdata, mcalibdata, obsdata, `~/ezphot/log`)에는 승인된 실제 아카이브 모드 외에는 쓰지 않고, 어떤 경우에도 지우지 않는다.
- 원본 FITS에 in-place platesolve·`run_swarp(fill_zero_tonan=True)`·rename·remove를 걸지 않는다.
- `update_pypl.sh`(PyPI 업로드), `scan_dependency.py`(requirements 덮어쓰기)를 실행하지 않는다. `~/ezphot/config`의 자격증명(`gwportal.config`·`atlas.config`)과 0.4.16 이하 `helper/queryhelper.py` 이력의 하드코딩 값을 쓰거나 출력하지 않는다.
- 커밋은 요청이 있을 때만. 작업 트리의 기존 미커밋 변경을 `git stash`·`checkout --`·`restore`·`reset`으로 되돌리지 않는다.

## 프로젝트 간 소통
- 다른 프로젝트(BRIDGE · ezphot · NGSF-7DT · pixelagents)에 보내는 요청·질문·통보는 그 프로젝트 세션(`<project>/lead-*`, `claude-HC_*` 등)에
  직접 SendMessage하지 않고 **superagent**에게 보낸다. superagent가 전달해 온 요청에 대한 답도 superagent에게 보낸다.
- superagent = pixel-agents 오피스가 `~/code/superagent`에서 띄운 `superagent-<id4>` 세션. 예전 7DT_SN_ToO 설정
  (지금은 `superagent/_archive/2026-10-28/7DT_SN_ToO_project`, `ports.md` 포함)과는 다르다.
- 보내기 전에 사용자에게 내용을 확인받는다. 길면 이 저장소의 `_workspace/upstream/<대상>_<slug>.md`에 쓰고 메시지에는 경로만 넣는다.
- `ListAgents`에 `superagent-*`가 없으면 보내지 않고, 사용자에게 superagent를 띄워 달라고 한다.
- 다른 저장소의 코드는 읽어도 되지만 고치지 않는다.

## 요청이 모호할 때

해석에 따라 바뀌는 코드나 결과가 달라지는 요청이면, 추측해서 진행하지 말고 먼저 구체적으로 묻는다.
- **묻는다**: 해석이 둘 이상이고 그 차이가 결과를 바꿀 때 / 코드·문서·기존 기록으로 답을 정할 수 없을 때 / 되돌리기 어렵거나 비용이 큰 일(수치 변화, 공개 동작 변경, 긴 실행)일 때.
- **묻지 않는다**: 코드를 읽어 알 수 있는 사실, 관례적 기본값이 분명한 것. 직접 확인하고, 가정한 것은 보고서의 "가정" 절에 적는다.
- **묻는 방법**: 무엇을 확인했는지 한두 줄로 밝힌 뒤, 한 번에 최대 4개까지, 질문마다 구체적인 선택지 2–4개와 권장안(이유 포함)을 준다.
  "무엇을 원하세요?" 같은 열린 질문 대신 "A(…)와 B(…) 중 어느 쪽인가요?"처럼 묻는다. 받은 답은 `_workspace/00_input/request.md`의 "## 확인한 사항"에 남긴다.
- **작업 도중**: 새 모호함이 생기면 그 시점에 멈추고 같은 방식으로 묻는다. 서브에이전트는 사용자에게 직접 묻지 않고, 질문과 선택지를 리더에게 올린다.
- **이 프로젝트에서 자주 모호한 것**: 지금 무엇이 틀렸고 어떤 결과가 맞는가(기대 동작) · 어느 망원경·필터·날짜·타일인가 ·
  측광 수치가 바뀌어도 되는가, 허용 범위는 · 공개 API를 깨도 되는가(deprecation 여부) · 어느 개구(`*_APER_3` = 10″ 등)·어느 baseline으로 판단하나 ·
  측광 실행이면 좌표, 필터, 날짜 범위, 원하는 산출물(측광·스택·DIA·광도곡선·갤러리), DIA reference.

## 코딩 원칙

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 하네스: ezphot 개발·측광 실행

**목표:** ezphot을 독립 라이브러리로 안전하게 고치고(공개 API는 CHANGELOG로 관리), 레시피로 측광을 격리 HOME에서 돌려 결과를 받는다.

**호출 조건:**
- ezphot 코드의 버그 수정·기능 추가·알고리즘 변경·리팩터링과 그 후속 요청(리뷰만 다시, 검증만 다시 등)에는 `ezphot-change` 스킬을 사용한다.
- 천체 이미지 측광·스택·DIA·광도곡선·갤러리 요청과 그 후속 요청(이어서 stack/dia, QA만 다시 등)에는 `ezphot-photometry` 스킬을 사용한다.
- 코드 위치만 묻는 단순 질문에는 직접 답해도 된다.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
| --- | --- | --- | --- |
| 2026-10-28 | Harness v2로 처음 구성 (에이전트 6, 스킬 9, 오케스트레이터 2) | 전체 | 개발과 측광 실행을 함께 맡기기로 결정 |
| 2026-10-28 | 코드 변경 오케스트레이터 이름 `ezphot-change` → `ezphot-develop` | ezphot-develop, ezphot-photometry, CLAUDE.md | 사용자 요청 |
| 2026-10-28 | 이름을 다시 `ezphot-develop` → `ezphot-change`로 되돌림 | ezphot-change, ezphot-photometry, CLAUDE.md | 사용자 요청 (`ezphot-dev`와 이름이 헷갈림) |
| 2026-10-28 | 「요청이 모호할 때」 규칙, 오케스트레이터 요청 명확화 단계, 엔지니어 질문 규칙 추가 | CLAUDE.md, ezphot-change, ezphot-photometry, ezphot-engineer | 사용자 요청: 모호한 질문은 구체적으로 되묻게 |
| 2026-10-28 | 「환경」 웹페이지 포트에 기록부 규칙 추가: `~/.pixel-agents/ports.md`(pixel-agents 소유), 상시 포트 변경은 superagent 경유, 일회성은 자동 감지 | CLAUDE.md | 사용자 결정: ports.md를 pixel-agents가 관리 |
| 2026-10-28 | 「프로젝트 간 소통」 추가: 다른 프로젝트로 가는 메시지는 모두 superagent(pixel-agents의 `superagent-<id4>`) 경유. BRIDGE·ezphot·NGSF-7DT·pixelagents CLAUDE.md에 같은 절 | CLAUDE.md | 사용자 결정: 프로젝트 간 요청은 superagent로 일원화 |
| 2026-10-28 | 샌드박스(harness-lab)에서 운영 저장소로 이식. 「에이전트 경계」·`ports.md` 등록 규칙 제거, 코딩 원칙 유지, 「환경」을 운영값(`too`, `scratch/ezphot-iso`, 8760–8780)으로. 예전 에이전트·스킬·CLAUDE.md는 `.claude/_archive/2026-10-28/` | 측광 실제 아카이브 모드(타일별 승인) 추가 | 사용자 결정: 샌드박스 폐지, 하네스를 운영으로, 승인 시 실제 아카이브 |
| 2026-10-29 | queryhelper 자격증명 문구를 0.4.17 상태(`ATLAS_TOKEN`·`atlas.config`, 이력의 옛 값)로, 격리 HOME 사본에 atlas.config 추가. `ezphot-verification` STOP grep을 data·log 경로로 좁힘(파라미터 경로는 의도적으로 실제), specific log 키 예외 기록 | CLAUDE.md, ezphot-safety, ezphot-safety-auditor, ezphot-verification | queryhelper 수정 후 낡은 문구, 검증 r1이 STOP grep 오탐으로 중단 |
