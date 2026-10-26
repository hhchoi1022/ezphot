# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

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

## 4. Goal-Driven Execution

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

---

## 에이전트 경계 (7DT_SN_ToO 프로젝트)

이 저장소는 **ezphot-developer agent(측광 담당)의 소유**다.

- 하위 에이전트: `ezphot-runner`(측광 실행, 코드 수정 금지)와 `ezphot-qa`(결과 독립 검증,
  파일 수정 금지)는 ezphot-developer가 호출하는 하위 계층이다. 측광 코드를 수정한 뒤에는
  `ezphot-verifier`(코드 변경 독립 검증, 코드 수정 금지)에 diff를 넘겨 diff 리뷰·BRIDGE API 호환·
  기준 타일(T10796, T05785) 재실행 후 `tests/baseline/` 비교를 받은 다음 보고한다.
  의도한 변경으로 기준값이 바뀌면 `tests/baseline/<tile>_summary.csv`는 developer가 교체한다.
  `ezphot-viz`(단계별 그림 갤러리, 산출물 수정 금지)는 사용자가 결과를 눈으로 볼 때 호출한다
  (`http://127.0.0.1:8765/ezphot/<objname>/viz/`, 서버 루트 `~/viz`).
- 웹페이지 포트: superagent `7DT_SN_ToO_project/CLAUDE.md`의 "웹페이지 호스팅 포트" 규칙을 따른다
  (8760–8780 범위, `127.0.0.1` 바인딩). ezphot 갤러리는 `8765` 고정(`ports.md` 등록, 상시).
  첫 화면 `http://127.0.0.1:8765/ezphot/`은 objname별 실행 목록·상태(`ezphot.recipes.status --watch 60`,
  setsid로 상시 실행)이다. 서버나 갱신 프로세스가 죽어 있으면 다시 띄우고 `ports.md`의 PID를 고친다.
  갤러리 서버나 다른 페이지 서버를 띄우거나 내리는 것은 ezphot-developer가 하고, 그때마다
  `/home/hhchoi1022/code/superagent/7DT_SN_ToO_project/ports.md`를 갱신한다(저장소 밖 수정 금지의 유일한 예외).
  띄우기 전에 `ss -ltnp | grep -E ':87([67][0-9]|80)\b'`로 확인하고, 보고할 때 포트·URL·내용을 함께 알린다.
- 시험·재현·검증 실행은 격리 HOME(`tests/make_isolated_home.py`, 데이터 경로가
  `/lyman/data1/factory_hhchoi/scratch/ezphot-iso/` 아래 임시 폴더를 가리키는 설정 사본)에서만 한다.
  기준 타일 T10796/T05785도 verifier는 격리 HOME에서 돌린다. 실제 HOME으로 recipe를 돌려
  아카이브(~/ezphot/data)에 쓰는 실행은 운영 처리이며, 대상 타일마다 사용자 승인을 받는다.
- 소유 범위: 측광 알고리즘·파이프라인 코드와 그 단위 테스트. 이 저장소 밖
  (BRIDGE, NGSF-7DT)은 수정하지 않는다 — 필요한 변경은 superagent를 통해 요청한다.
- **공개 API(측광 진입점 함수, 반환 형식, 설정 스키마)를 바꿀 때는 반드시 보고**할 것:
  BRIDGE가 이 패키지를 import해서 호출하므로 하위 호환이 깨지면 통합이 깨진다.
- BRIDGE에서 넘어온 버그 리포트(재현 조건 포함)를 받으면 이 저장소 안에서 수정하고,
  수정 후 superagent에 보고해 bridge agent가 통합 재검증을 하게 한다.
