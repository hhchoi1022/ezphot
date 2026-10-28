---
name: photometry-qa
description: "ezphot 처리 산출물(process_images·stack·dia summary CSV, 카탈로그, target_photometry 측광표, visualize 그림)과 baseline 비교 결과가 측광적으로 타당한지 점검하는 체크리스트와 판정 형식. 측광 실행 뒤 결과 판정, '이 ZP 괜찮아?', '광도곡선 이상해', baseline 차이의 의미 해석, 변경 전후 산출물 비교에 사용한다. 코드 정확성은 ezphot-review, 실행 안전성은 ezphot-safety가 맡는다."
---

# 측광 QA

`recipes/qa.py`는 기준값만 본다(ZPERR, n_ref, SEEING, depth). 기준을 통과해도 측광은 틀릴 수 있다:
밝은 별 halo 아래의 배경 과소 차감, 호스트 은하가 지배하는 개구, 잘못된 reference, 프레임 하나의 나쁜 platesolve.
이 점검은 "숫자와 그림이 측광적으로 말이 되는가"를 본다. 산출물은 읽기만 한다. PNG는 Read로 직접 본다.

## 입력
- summary CSV들, `qa` 출력, target 측광표, 갤러리 그림(`<iso>/home/viz/...` 또는 `_workspace/runs/...`)
- baseline 비교면 `compare_summary.py` 출력과 baseline CSV
- 대상 정보(좌표, 예상 밝기, 호스트 여부)

## 체크리스트

### 이미지별 (process_images / stack)
1. `success`·`step`·`error` 분포: 실패가 특정 필터·날짜·망원경에 몰리는가?
2. `n_ref`(ZP 별 개수): 기준보다 적거나, 같은 필터에서 크게 튀는가?
3. `ZP_APER_3`·`ZPERR_APER_3`: 같은 밤·필터·망원경 안의 산포. 0.05 mag 넘게 벗어나는 프레임.
4. `SEEING`·`ELLIP`: 추적 오류·초점 문제 프레임. `SKYVAL`·`SKYSIG`: 박명·달빛.
5. `UL5SKY_APER_3`(depth): 노출·하늘 밝기로 설명되는가?
6. 스택: 입력 수, 1장 그룹 제외 여부, 스택 depth가 입력보다 깊어졌는가.

### 대상 측광 (target_photometry)
7. 매칭 반경 안 행 존재, 여러 개면 어느 것인가.
8. `SATDIST < 1.2` 행: 밝은 별 halo로 배경이 과소 차감 → 대상이 밝게 나온다. DIA나 국소 annulus로 대체 권고.
9. 오차 `99`/mag≈125 sentinel 섞임.
10. 개구(지름 10″ = `*_APER_3`)가 호스트에 지배되는가 — 확장된 호스트 위의 transient면 DIA forced 값과 비교.
11. 시간·필터 순서가 물리적으로 말이 되는가(급변은 프레임 문제를 먼저 의심).

### DIA
12. reference와 science의 seeing·depth 관계, kernel sigma(1.2 px 하한 적용 여부 — 이전 kernel로 만든 산출물과 섞지 않는다).
13. 차영상 잔차 그림: 별 위치의 쌍극 잔차(정렬·PSF 불일치), 포화 별.
14. candidate/transient 카탈로그가 비어 있는 것은 현재 알려진 문제(음수 검출 패스)일 수 있다 — 결과로 해석하지 않는다.

### baseline 비교 (개발 검증)
15. 어느 열이 몇 행에서 얼마나 바뀌었나. 바뀐 방향이 변경 의도와 맞는가(예: 배경 추정 변경 → SKYVAL·ZP 체계적 이동).
16. run-to-run 산포(compare_summary 기본 허용오차)를 넘는 변화만 의미 있게 다룬다.

## 판정과 보고

```
| 항목 | 판정(정상/주의/이상) | 근거(파일, 행, 값) |
```
- 과학적 선택(어느 프레임을 버릴지, 개구 대신 DIA를 쓸지, baseline을 갱신할지)은 결론 내리지 말고 근거와 선택지를 사용자에게 넘긴다.
- 이상의 원인이 코드로 보이면 의심 위치(파일:줄)를 적어 리더에게 넘긴다. 고치지 않는다.
