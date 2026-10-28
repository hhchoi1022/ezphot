---
name: ezphot-review
description: "ezphot 코드 diff를 정확성·단위·공개 API 영향·조용한 실패·변형/삭제 API 관점에서 검토하는 체크리스트와 판정 형식. 구현 직후 리뷰, 수정본 재검토, '이 변경 괜찮아?', 'diff 봐줘' 요청에 사용한다. 실행 안전성은 ezphot-safety, 산출물의 측광적 타당성은 photometry-qa가 맡으므로 여기서는 코드의 옳고 그름만 본다."
---

# ezphot 변경 리뷰

리뷰의 목적은 "이 diff가 요청을 정확히, 그리고 그것만 하는가"를 확인하는 것이다. ezphot은 회귀 테스트가 얇아서 리뷰가 첫 번째 검증이다.

## 입력
- `_workspace/00_input/request.md`, `_workspace/01_ezphot-mapper_report.md`, `_workspace/02_<engineer>_change.md`
- 실제 diff: `bash .claude/skills/ezphot-verification/scripts/change_diff.sh _workspace` (작업 트리 전체 `git diff`는 기존 미커밋 변경까지 섞여 쓰지 않는다)

엔지니어 보고와 실제 diff가 다르면 diff를 믿고, 차이를 지적한다.

## 체크리스트 (순서대로)

1. **요청 충족**: 경계 입력(빈 카탈로그, 1장 ImageSet, NaN·마스크된 픽셀, 포화, 0 노출, 헤더 키 누락)을 따라가 본다.
2. **범위**: 요청 밖 변경이 섞였는가?
3. **단위·규약**: pixel/arcsec, ADU/e⁻, flux·mag 규약, 0/1-based 좌표, sentinel. 변환이 한 번만, 올바른 방향으로 일어나는가.
4. **공개 API**: ezphot-api-policy의 분류가 맞는가? 파괴적인데 "호환"으로 분류했거나, 수치 변경인데 CHANGELOG에 영향 크기가 없는가?
   deprecation 없이 이름·인자를 지웠는가?
5. **조용한 실패**: 새 `except`, 새 기본값, 새 early return이 실패를 None·빈 카탈로그로 바꾸는가?
6. **변형·삭제**: overwrite, rename, remove, update 모드 open이 새로 생겼거나 대상이 바뀌었는가? 입력 FITS가 보존되는가?
7. **상태 공유**: mutable default, 호출자 객체 변형, 공유 summary 파일 동시 쓰기.
8. **import**: 새 import가 순환을 만들지 않는가(지역 import 패턴).
9. **외부 바이너리**: 인자·경로(공백 포함 경로), 임시 디렉터리 정리, 실패 시 입력 보존.
10. **레시피·문서**: 레시피 summary 열·CLI가 바뀌면 docstring·`qa.py`·`compare_summary.py`도 맞췄는가? CODEBASE_GUIDE의 사실이 틀리게 됐으면 갱신했는가?
11. **독립성**: 외부 프로젝트 이름·경로를 코드에 새로 넣었는가?
12. **테스트 자산**: baseline CSV를 diff로 바꿨다면 그 자체가 차단 사유다(사용자 승인 사항).

## 판정

```
[차단|수정 권고|참고] 파일:줄 — 무엇이 문제인가
  근거: 코드 인용 또는 실행 경로
  제안: 구체적인 수정 방향
```
차단이 하나라도 있으면 전체 판정 `변경 필요`, 없으면 `승인`. 확신 없는 지적은 "확인 필요"로 두고 차단으로 올리지 않는다.

## 재검토
이전 지적의 해소 여부를 먼저 표시하고 새 diff 부분만 다시 본다. 같은 지적이 두 회차 연속 남으면 리더에게 올린다.
