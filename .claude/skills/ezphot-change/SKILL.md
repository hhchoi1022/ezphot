---
name: ezphot-change
description: "ezphot 라이브러리 코드 변경 하네스의 오케스트레이터. 영향 분석 → 구현↔리뷰(CHANGELOG·버전 제안 포함) → 안전 감사 → 격리 검증(baseline 회귀) → 측광 QA → 보고를 조율한다. ezphot 버그 수정, 기능 추가, 알고리즘 변경, 리팩터링, 'DIA v2', '배경 추정 바꿔줘', 'aperturephotometry ○○ 고쳐줘', '레시피에 옵션 추가' 요청에 사용한다. 수정, 보완, 다시 실행, 재검증, '리뷰만 다시', '검증만 다시', '이전 결과를 바탕으로' 같은 후속 요청에도 사용한다. 천체 측광 실행 요청은 ezphot-photometry, 코드 위치만 묻는 질문은 직접 답한다."
---

# ezphot 변경 오케스트레이터

ezphot은 여러 프로젝트가 쓰는 독립 측광 라이브러리다. 이 하네스는 세 가지를 보장하도록 짜여 있다.
1. **안전**: 감사 없이 아무것도 실행되지 않는다. ezphot은 HOME을 격리하지 않으면 실제 아카이브에 쓴다.
2. **정직한 공개 API 관리**: 변경마다 분류(내부/호환 추가/수치 변경/파괴적)를 하고 `CHANGELOG.md`로 알린다. 사용처 코드는 찾아다니지 않는다.
3. **조용한 수치 변화 방지**: 측광 수치가 바뀔 수 있는 변경은 baseline으로 확인하고 크기를 기록한다.

리더(메인 에이전트)가 이 스킬을 따른다. 리더도 `ezphot-safety`를 따르며 검증 명령을 직접 실행하지 않는다(예외: 스냅샷·변경분 스크립트, V0).

## 실행 모드: 혼합

| 단계 | 실행 모드 | 선택 이유 |
| --- | --- | --- |
| 1. 영향 분석 | 서브에이전트 위임 | 결과를 한 번 받으면 된다 |
| 2. 구현 ↔ 리뷰 | 지속형 에이전트 협업 | 같은 엔지니어·리뷰어가 맥락을 유지하며 왕복 |
| 3. 안전 감사 | 서브에이전트 위임 | 구현 맥락에 오염되지 않은 독립 판정자 |
| 4. 격리 검증 | 서브에이전트 위임 | 감사된 명령을 순서대로 실행 |
| 5. 측광 QA | 서브에이전트 위임 | V3을 실행했을 때만 |
| 6. 보고 | 리더 직접 | |

`Workflow` 도구는 기본으로 쓰지 않는다. 사용자가 "워크플로로", "철저히 리뷰"를 명시하면 2단계 리뷰를 관점별(단위·공개 API·조용한 실패) 적대적 검증으로 바꿀 수 있다(에이전트 10명 미만).

## 에이전트 구성

| 이름 | `subagent_type` | 모델 | 역할 | 스킬 | 산출물 |
| --- | --- | --- | --- | --- | --- |
| (이름 없음) | `ezphot-mapper` | sonnet | 위치·내부 호출·API 범주 조사 | ezphot-dev, ezphot-api-policy | `01_ezphot-mapper_report.md` |
| `engineer` | `ezphot-engineer` | opus | 구현 + CHANGELOG + 버전 제안 | ezphot-dev, ezphot-api-policy, ezphot-safety | 변경분 + `02_engineer_change.md` |
| `reviewer` | `ezphot-reviewer` | opus | diff 검토 | ezphot-review, ezphot-api-policy | `02_ezphot-reviewer_review_r<N>.md` |
| (이름 없음) | `ezphot-safety-auditor` | opus | 실행 전 감사 | ezphot-safety, ezphot-verification, ezphot-recipes | `03_ezphot-safety-auditor_verdict.md` |
| (이름 없음) | `ezphot-runner` | sonnet | 격리 HOME 검증 실행 | ezphot-verification, ezphot-recipes, ezphot-safety | `04_ezphot-runner_report.md` |
| (이름 없음) | `photometry-qa` | opus | baseline 차이의 측광적 해석 | photometry-qa | `05_photometry-qa_report.md` |

## 작업 디렉터리

`ezphot/_workspace/` (git 무시: `.git/info/exclude`). 파일명 `{phase}_{agent}_{artifact}.{ext}`. `S=.claude/skills/ezphot-verification/scripts`.

## 요청 명확화 (0단계보다 먼저)

1. 요청을 CLAUDE.md 「요청이 모호할 때」 기준과 그 프로젝트의 "자주 모호한 것" 목록으로 점검한다.
   코드·문서·`_workspace` 기록을 읽어 스스로 답할 수 있는 것은 먼저 확인한다(읽기 전용).
2. 남은 모호함은 AskUserQuestion으로 묻는다: 질문 최대 4개, 질문마다 선택지 2–4개, 권장안을 첫 번째에 두고 이유를 적는다.
   질문 앞에 무엇을 확인했는지 한두 줄로 밝힌다.
3. 답과, 묻지 않고 가정한 것을 기억해 두었다가 1단계에서 `_workspace/00_input/request.md`의 "## 확인한 사항", "## 가정"에 적는다.
4. 모호함이 없으면 묻지 않고 바로 0단계로 간다. 1단계 이후(영향 분석 결과 등)에 생기는 질문은 그 단계에서 묻되, 이미 답한 것을 다시 묻지 않는다.
5. 최종 보고서에 "가정" 절을 둔다 — 사용자가 확인하지 않은 가정이 결과에 영향을 줬는지 알 수 있게.

## 0단계: 기존 작업 확인

- `_workspace/`가 없으면 → 처음부터.
- 있고 같은 요청의 부분 재실행이면 → 해당 단계부터, 기존 `00_baseline.txt` 유지.
- 있고 새 요청이면 → `_workspace_<YYYYmmdd_HHMMSS>/`로 옮기고 처음부터.
- 처음부터면 엔지니어를 띄우기 전에 `bash $S/snapshot.sh _workspace`. 작업 트리에는 원래 미커밋 변경이 있으니 묻지 말고 스냅샷만 남긴다.
  이후 "엔지니어 diff"는 항상 `bash $S/change_diff.sh _workspace`.

## 1단계: 영향 분석

**실행 모드:** 서브에이전트 위임

1. 요청 원문을 `_workspace/00_input/request.md`에 저장한다.
2. `Agent(subagent_type: "ezphot-mapper", prompt: "<요청> / 요청 파일 경로 / 보고서를 최종 메시지로 반환")` → `_workspace/01_ezphot-mapper_report.md`.
3. "모르는 것"에 사용자 판단이 필요한 항목(알고리즘 선택, 수치 변화 허용, 파괴적 변경 허용·deprecation 여부)이 있으면 **여기서 묻고** 답을 request.md에 덧붙인다.

## 2단계: 구현 ↔ 리뷰

**실행 모드:** 지속형 에이전트 협업

1. 한 메시지에서 병렬로 띄운다.
   - `Agent(name: "engineer", subagent_type: "ezphot-engineer", prompt: "요청·영향 분석 경로 / 산출: _workspace/02_engineer_change.md, CHANGELOG.md Unreleased / 첫 보고에 사용 가능한 도구 목록")`
   - `Agent(name: "reviewer", subagent_type: "ezphot-reviewer", prompt: "대기. 리더가 요청하면 리뷰 / 첫 보고에 사용 가능한 도구 목록")`
2. 첫 보고에 Edit/Write(엔지니어)·SendMessage가 빠졌으면 리더가 대신하거나 `tools` 제한 없이 다시 띄운다.
3. 엔지니어 완료 → `SendMessage({to: "reviewer"}, "리뷰 r1: request/mapper/change 경로")` → `_workspace/02_ezphot-reviewer_review_r1.md`.
   - `승인` → 4로. `변경 필요` → 지적을 엔지니어에게 전달, r2 재검토. 같은 지적이 남거나 반론이 있으면 리더가 판단하거나 사용자에게 묻는다. **왕복 최대 3회.**
4. **동결**: 엔지니어·리뷰어에게 동결을 알리고(이후 수정은 `_v2` 새 파일, 코드 수정은 리더 지시 없이 하지 않음)
   `bash $S/change_diff.sh _workspace > _workspace/02_frozen.diff; sha256sum _workspace/02_* > _workspace/freeze_02.sha`.

## 3단계: 안전 감사

**실행 모드:** 서브에이전트 위임

1. 리더가 `ezphot-verification` 단계 기준과 영향 분석·리뷰·change.md의 검증 요청을 합쳐 `_workspace/03_plan.md`를 쓴다.
   명령은 실제 실행할 문자열 그대로, 격리 경로만 `<iso>`로. 분류가 "수치 변경"이면 V3 baseline을 반드시 넣는다. 드라이버는 `_workspace/verify/<name>.py`.
2. `Agent(subagent_type: "ezphot-safety-auditor", prompt: "계획·드라이버·동결 diff·change.md 경로 / 판정을 최종 메시지로")` → `_workspace/03_ezphot-safety-auditor_verdict.md`.
   - `차단` 명령은 빼거나 고쳐서 재감사. diff가 만든 위험이 차단이면 2단계로.
   - `조건부: 사용자 승인 필요`(V3)는 시간·디스크 비용과 함께 묻는다. 거절되면 "미실행(승인 거절)" — 이 경우 수치 변경 분류의 CHANGELOG 항목에 "baseline 미확인"을 적게 한다.
3. 실행할 것이 V0뿐이면 4단계를 건너뛰고 리더가 V0만 직접 확인한다.

## 4단계: 격리 검증

**실행 모드:** 서브에이전트 위임

1. `Agent(subagent_type: "ezphot-runner", prompt: "개발 검증 / 계획·판정·드라이버 경로 / 허용 명령만 / 끝나면 <iso> 정리 / 보고서: _workspace/04_ezphot-runner_report.md")`.
2. 보고서를 읽는다.
   - 사후 검사 엄격 출력(접두어 없음, `CHANGED-OR-DELETED`) → **모든 작업 중단**, 사용자에게 즉시 보고.
   - fail → 실패 로그를 엔지니어에게 전달해 2단계 재진입(동결 해제 명시) → 3단계 재감사부터.
   - pass/skip → V3을 실행했으면 5단계, 아니면 6단계.

## 5단계: 측광 QA

**실행 모드:** 서브에이전트 위임 (V3 실행 시에만)

`Agent(subagent_type: "photometry-qa", prompt: "baseline 비교 해석 / 러너 보고서·비교 출력·change.md(의도한 변화) 경로 / 최종 메시지로")` → `_workspace/05_photometry-qa_report.md`.
의도와 다른 변화가 있으면 엔지니어에게 넘겨 2단계로 갈지 사용자와 정한다. 의도한 변화면 그 크기를 CHANGELOG 항목에 반영하게 한다(동결 해제 → 엔지니어 수정 → 리뷰 1회).

## 6단계: 보고

1. `sha256sum -c _workspace/freeze_02.sha`로 동결 산출물이 바뀌지 않았는지 확인(5단계에서 CHANGELOG를 고쳤으면 새 해시로 다시 동결한 뒤).
2. `_workspace/06_report.md`를 쓰고 사용자에게 요약한다:
   - 무엇을 바꿨나(파일:줄, `02_frozen.diff`), 공개 API 분류와 근거
   - `CHANGELOG.md` Unreleased 항목(그대로 인용), 버전 제안
   - 리뷰 회차와 남은 "참고" 지적
   - 검증 단계별 결과, 사후 검사(엄격·CATALOG·INFO)
   - baseline 차이와 QA 요약(실행했으면)
   - 사용자 결정 사항: 커밋, 버전 올리기, baseline 갱신, CODEBASE_GUIDE 갱신
3. 커밋하지 않는다. `_workspace/`는 남긴다.

## 데이터 전달

| 단계 → 단계 | 방법 |
| --- | --- |
| 0 → 전체 | `00_baseline.txt`, `00_untracked.txt` |
| 1 → 2 | `01_ezphot-mapper_report.md` |
| 2 내부 | SendMessage(리더 중계) + `02_*` |
| 2 → 3 | `02_frozen.diff` + `freeze_02.sha` |
| 3 → 4 | `03_plan.md`, `verify/*.py`, `03_ezphot-safety-auditor_verdict.md` |
| 4 → 5 | `04_ezphot-runner_report.md`, `04_artifacts/` |

## 오류 처리

| 상황 | 대응 |
| --- | --- |
| 서브에이전트 실패 | 한 번 재시도. 1·3단계는 필수라 또 실패하면 진행하지 않고 보고 |
| 지속형 에이전트 무응답 | SendMessage로 확인 → 같은 유형을 `engineer-b`처럼 새 이름으로 띄우고 change.md·변경분 경로 전달 |
| 사용량 한도·인증 만료·권한 거부 | 재시도하지 않는다. 부분 산출물을 직접 확인하고 `_workspace/`에 누락 기록, 사용자에게 보고(한도 해제 시각 포함) |
| 리뷰 왕복 3회 초과 | 쟁점을 정리해 사용자 결정 요청 |
| 감사 차단 | 우회 명령을 만들지 않는다. 꼭 필요한 검증이면 사용자에게 알린다 |
| 사후 검사 엄격 출력 | 전면 중단·즉시 보고. 아무것도 지우지 않는다 |
| 에이전트 판단을 리더가 채워야 할 때 | 직접 확인한 사실만 반영. 판단 이유를 추측해 쓰지 않는다 |

## 테스트 시나리오

### 정상 흐름 — 수치 변경
요청: "배경 맵 box_size 기본값을 128에서 64로 바꿔줘."
1. mapper: `recipes/defaults.py`의 `BKGMAP`·`BKGRMS`, `backgroundgenerator.py`, 레시피 summary의 SKYVAL·ZP 영향 → "수치 변경" 예상.
2. engineer 구현 + CHANGELOG `### 수치 변경` 항목(영향 크기 "baseline 확인 후 기입") + patch 제안 → reviewer `승인` → 동결.
3. 계획 V0·V1·V3(T05785 baseline). 감사: V3 조건부(승인). 사용자 승인.
4. runner: `--pre` → process_images → compare_summary 실패(SKYVAL·ZP 이동, 의도된 변화) → 사후 검사 clean, CATALOG 0줄 → `<iso>` 정리.
5. photometry-qa: 이동 방향·크기가 box_size 축소와 맞는지 판단 → 엔지니어가 CHANGELOG에 크기 기입 → 리뷰 1회 → 재동결.
6. 보고: baseline 갱신 여부는 사용자 결정.

### 오류 흐름 — 감사 차단
드라이버가 실제 scidata FITS를 복사하지 않고 `ScienceImage(path).platesolve()`를 바로 부름.
감사: "입력 FITS in-place platesolve, solve-field 실패 시 입력 삭제 경로" → 차단. 리더는 드라이버를 `<iso>`로 복사한 파일을 쓰도록 고쳐 재감사.

### 오류 흐름 — 입력 변경 감지
runner 사후 검사에 `CHANGED-OR-DELETED .../7DT07_..._0002.fits`. runner 즉시 중단·보고, 리더는 모든 단계를 멈추고 경로·명령·시각을 사용자에게 보고. 파일을 건드리지 않는다.
