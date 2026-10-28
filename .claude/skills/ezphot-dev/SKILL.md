---
name: ezphot-dev
description: "ezphot 코드를 고치거나 기능을 추가할 때의 작업 규약, 어디부터 읽을지, 모듈별 변경 시작점, 알려진 깨진 경로와 위험 지점. ezphot 구현 작업(버그 수정, 기능 추가, 리팩터링, DIA·측광·배경·스택·캘리브레이션 수정)에 반드시 사용한다. 공개 API 판정은 ezphot-api-policy, 실행·검증은 ezphot-verification 스킬이 맡는다."
---

# ezphot 개발 규약

정본은 `CODEBASE_GUIDE.md`다("One-page mental model", "Where to make a change", "Current risks"). 이 스킬은 변경 작업에 바로 필요한 부분을 추린다.
ezphot은 회귀 테스트가 얇고 넓은 `except`가 많아서, 돌아가는 것처럼 보이는 변경이 산출물을 조용히 바꾸기 쉽다.

## 구조 한 줄

FITS → `ScienceImage`/`ReferenceImage`(헤더로 관측소·필터 추론, config 로드) → `ImageMethod` 파사드 →
`methods/*`(preprocess, mask, bkg, astrometry, photometry, photcal, stack, subtract) → sidecar(info JSON, .cat 등) →
`Catalog`/`CatalogSet` → `LightCurve`/`PhotometricSpectrum`. 배치는 `ImageSet`. CLI 워크플로는 `recipes/`.

## 변경 시작점

| 바꾸려는 것 | 시작 | 함께 볼 것 |
| --- | --- | --- |
| 헤더 키 해석 | `helper/photometryhelper.py` | `observatory_info_hint.yaml`, `BaseImage` |
| 망원경/카메라 | `configuration/common/observatory_info.dat` | `Configuration.register_telescope()` |
| 처리 단계 | `methods/<op>.py` | `ImageMethod`, 레시피 |
| 개구 측광·컬럼 | `methods/aperturephotometry.py` | `Catalog`, photcal, 레시피 summary 열 |
| 측광 보정 | `methods/photometriccalibration.py` | `SkyCatalog`, 색 변환, SATDIST |
| 배경·오차 맵 | `methods/backgroundgenerator.py`, `methods/errormapgenerator.py` | 2026-10 per-mesh 5σ-clipped mean/std로 변경됨(`4a31202`) |
| stack | `methods/stack.py` | `ImageSet.stack` 반환 형태 |
| DIA | `methods/subtract.py` | kernel 하한, 음수 검출 패스 |
| 레시피 공통 파라미터 | `recipes/defaults.py` | 7DT 운영 설정을 맞춘 값(docstring). 바꾸면 수치 변경 |
| sky catalog | `skycatalog/`, `utils/catalogquerier.py` | `CATALOG_DIR` 캐시 |

## 규약

1. **먼저 CODEBASE_GUIDE의 해당 절과 "Current risks"를 읽는다.**
2. **최소 diff.** 요청 밖 리팩터링·포매팅 금지. 주변 코드의 이름·주석 밀도·관용구를 따른다.
3. **단위·규약 유지**: pixel/arcsec/deg, ADU/e⁻, flux 규약, 0/1-based 좌표, `err=99`. 바꾸면 그게 변경의 본체이고 ezphot-api-policy의 "파괴적" 또는 "수치 변경"이다.
   좌표 기준은 최근 변경·revert 이력이 있다(`0557129` → `3a612d2`).
4. **조용한 실패를 늘리지 않는다.** 새 맨 `except:`/`except Exception: pass` 금지.
5. **mutable default 인자 금지**, 호출자·기본 객체를 변형하지 않는다(`35c8a84` 같은 수정 이력).
6. **변형·삭제 API**(`run_swarp(fill_zero_tonan=True)`, `rename`, `remove`, `write`)를 새로 쓰면 대상 경로가 무엇으로 해석되는지 주석이나 change.md에 적는다.
7. **새 import는 깨끗한 인터프리터 기준.** 패키지가 eager export + 지역 import로 순환을 피하고 있다(V1에서 확인).
8. **비대화형 실행**: `input()`을 새로 넣지 않는다(`52ccdb7`).
9. **비공개·stale 경로**: `ezphot/analysis/`(9개는 파싱조차 안 됨), `imageobjects/imageset copy.py`, `utils/imagequerier_standalone.py`는 요청 없으면 고치지 않는다.
10. **외부 프로젝트 이름을 코드·문서에 새로 넣지 않는다.** ezphot은 독립 라이브러리다. 기존 docstring의 언급은 그대로 둔다.
11. **실행 확인은 직접 하지 않고** 검증 단계로 요청한다(ezphot-safety, ezphot-verification). 예외: 바뀐 파일의 `ast.parse`.
12. **커밋하지 않는다.** 버전도 올리지 않고 제안만(ezphot-api-policy).

## 알려진 깨진 경로 (근처를 고치면 고쳤는지/그대로 뒀는지 보고)

- `Stack.stack_swarp()` ↔ `prepare_images()` kwargs 불일치, `ImageQuerier.query()`의 `pixel_scale`
- `ErrormapGenerator.calculate_sourcerms_from_propagation()` numexpr 식 오류(`/fcf mflat_map`)
- `PSFPhotometry.build_epsf_model_psfex()`의 정의되지 않은 `sex_params`
- `Errormap.target_img`의 `tippy.imageobjects` import
- `skycatalog/conversion.py`의 `PANSTARRS1_to_SDSS` 이중 정의
- `ImageFormatter.search_files()`의 `folder` kwarg
- `update_new_transmission.py` import 시 예제 실행
- DIA 음수 검출 패스가 반전 이미지가 아니라 양수 `sub_` 이미지로 SExtractor를 돌려 candidate/transient 카탈로그가 항상 빔(2026-10 확인, "DIA v2"에서 수정 예정)
- `observatory_info.dat`의 LOAO iKon pixel scale(0.794″ → 0.356″가 맞음)

## 구현 보고 형식

`_workspace/02_<engineer>_change.md`:
- 바꾼 파일과 이유(파일:줄)
- ezphot-api-policy 분류(내부/호환 추가/수치 변경/파괴적)와 근거, CHANGELOG `Unreleased` 항목 초안, 버전 제안
- 고친/그대로 둔 알려진 문제
- 검증 요청: V0–V3 중 무엇이 왜 필요한지(수치 변경이면 V3 baseline 필수)
- 범위 밖 발견
- 가정: 묻지 않고 정한 해석과 그 이유(결과에 영향을 줄 수 있으면 리더에게 먼저 질문한다)
