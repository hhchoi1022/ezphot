---
name: ezphot-api-policy
description: "ezphot을 독립 라이브러리로 유지하기 위한 공개 API 정의, 변경 분류(호환/수치 변경/파괴적 변경), 버전 올리기, CHANGELOG 작성, deprecation 규칙. ezphot의 공개 클래스·메서드·인자·반환 형태·카탈로그 컬럼·출력 파일명·기본값을 바꾸거나, 버전·CHANGELOG를 다룰 때 반드시 사용한다. 특정 사용처(다른 프로젝트)의 코드를 근거로 판단하지 않는다."
---

# ezphot API 정책

ezphot은 여러 프로젝트가 가져다 쓰는 라이브러리다. 사용처는 ezphot을 editable 설치나 버전으로 받아 쓰고,
ezphot 쪽에서는 누가 무엇을 쓰는지 다 알 수 없다. 그래서 **사용처를 찾아다니는 대신, 무엇이 공개 계약인지 정하고
그것이 바뀔 때 버전과 CHANGELOG로 알린다.** 사용처는 CHANGELOG를 읽고 스스로 적응한다.

이 하네스는 특정 사용처의 코드를 읽거나 고치지 않는다. 사용처 이름을 근거로 "괜찮다/안 된다"를 판단하지 않는다.

## 공개 계약 (이것이 바뀌면 CHANGELOG에 적는다)

| 범주 | 범위 |
| --- | --- |
| 이름 | `ezphot.imageobjects`, `ezphot.methods`, `ezphot.dataobjects`, `ezphot.utils`, `ezphot.helper`, `ezphot.recipes`가 export하는 클래스·함수 |
| 시그니처 | 위 이름의 인자 이름·순서·기본값 |
| 반환 | 반환 타입·튜플 길이·None 가능성(예: `ImageSet.stack()` → `(image, bkgrms)`) |
| 카탈로그 | 컬럼 이름·단위·sentinel(`MAGSKY_APER_*`, `MAGERR_*`, `ZP_*`, `UL5SKY_*`, `X_IMAGE`/`Y_IMAGE` 기준, 오차 `99`) |
| 산출물 | 출력 파일명 패턴(`coadd_*com.fits`, `.cat`, `.circ.cat`, `sub_*`, `.tract7dtcat`, sidecar 확장자), 저장 위치 규칙 |
| 수치 | 같은 입력에서 ZP·FWHM·배경·kernel·측광값이 바뀌는 알고리즘·기본값 변경 |
| 레시피 | CLI 인자, summary CSV 열, 종료 코드 |
| 설정 | `config/common/*` 키와 의미 |

비공개: `_`로 시작하는 이름, `ezphot/analysis/`, `*copy*.py`, `*_standalone.py`(stale), 테스트 보조 파일.

## 변경 분류

| 분류 | 예 | 버전 (0.x 동안) |
| --- | --- | --- |
| **내부** | 리팩터링, 비공개 함수, 문서 | 올리지 않아도 됨. CHANGELOG 선택 |
| **호환 추가** | 새 인자(기본값 = 기존 동작), 새 함수·컬럼 | patch (0.4.16 → 0.4.17) |
| **수치 변경** | 배경 추정법, kernel 하한, 클리핑 규칙 | patch 이상, CHANGELOG에 **"수치 변경"**과 영향 크기 필수 |
| **파괴적** | 이름 삭제·변경, 인자 의미 변경, 반환 형태·컬럼·파일명 변경 | minor (0.4.x → 0.5.0), deprecation 우선 |

수치 변경은 에러 없이 사용처의 결과를 바꾸므로 가장 조심한다. baseline 비교 결과(어느 열이 얼마나)를 CHANGELOG 항목에 적는다.

## Deprecation

파괴적 변경은 가능하면 한 버전 동안 옛 이름·인자를 남기고 `warnings.warn(..., DeprecationWarning, stacklevel=2)`로 알린 뒤 다음 minor에서 지운다.
즉시 지워야 하면(틀린 결과를 내는 경우) CHANGELOG에 그 이유를 적는다.

## CHANGELOG

저장소 루트 `CHANGELOG.md`(없으면 첫 변경 때 만든다). 맨 위 `## Unreleased`에 항목을 추가하고, 버전을 올릴 때 그 절의 제목을 버전·날짜로 바꾼다.

```markdown
# Changelog

## Unreleased
### 수치 변경
- Subtract: HOTPANTS kernel sigma 하한 1.2 px. T05785 기준 sigma < 1.2 px coadd 약 37%에서 kernel이 바뀐다.
### 파괴적
- ImageSet.stack: 이미지 1장일 때도 `(image, bkgrms)`를 반환 (이전: image).
### 추가
### 수정
```
항목 하나 = 사용처가 알아야 할 사실 하나. 무엇이 바뀌었는지, 사용처가 무엇을 확인해야 하는지를 쓴다. 커밋 해시는 커밋 후 덧붙인다.

## 버전 올리기

버전은 `ezphot/__init__.py`의 `__version__` 한 곳(`setup.py`가 읽는다). 엔지니어는 올리지 않고 **제안만** 한다(분류와 함께).
실제로 올리는 것은 사용자가 커밋할 때 결정한다. `update_pypl.sh`는 bump + PyPI 업로드를 함께 하므로 쓰지 않는다.

## 판정 순서

1. diff가 공개 계약 표의 어느 범주를 건드리는가? 없으면 "내부".
2. 기존 호출이 그대로 동작하는가? 아니면 "파괴적".
3. 같은 입력에서 숫자가 바뀌는가? 그렇다면 "수치 변경" + baseline 근거.
4. CHANGELOG `Unreleased` 항목과 버전 제안을 change.md에 적는다.
