# Changelog

## 0.5.2 (2026-10-30)
### 추가
- `recipes.visualize`: 키워드 인자 `n_proc: int = 8`(시그니처 맨 끝)과 CLI `--n-proc`(기본 8). 갤러리의 이미지를 장 단위로 `n_proc`개 프로세스에서 그린다
  (`ProcessPoolExecutor`, 다른 레시피와 같은 방식). **기본이 병렬이다**: 이전처럼 한 프로세스로 돌리려면 `n_proc=1`(`--n-proc 1`).
  병렬일 때 CPU를 최대 8코어 쓰고, 전처리(`preprocess`) 그림을 그리는 장은 워커마다 과학 영상·master 전체를 메모리에 올리므로(장당 메모리는 이전과 같고, 9576×6388 영상에서 약 2 GB로 추정 — 측정 전) 동시에 그리는 장 수만큼 메모리 사용이 늘어난다.
  갤러리마다 첫 장은 혼자 먼저 그리고(그 장의 platesolve 그림이 sky catalog 캐시 `CATALOG_DIR`를 채우도록) 나머지를 병렬로 그린다.
  기본 카탈로그(`GAIAXP_CORR_LAMOST`)는 캐시에 없으면 조회·쓰기 없이 그 그림만 실패하므로 워커들은 캐시를 읽기만 한다. 원격 조회하는 카탈로그 종류로 바꾼 경우에는, 시야가 다른 장이 섞여 그 영역이 캐시에 없을 때 여러 워커가 동시에 원격 조회해 캐시(`summary.ascii_fixed_width`)에 쓸 수 있다(이 캐시 쓰기에는 잠금이 없다).
  파일 stem이 같은 장이 있으면(출력 디렉터리가 겹침) 그 갤러리는 순차로 그린다.
  그림·`index.html`·`manifest.json`의 내용과 순서, `[visualize]` 진행 출력의 순서는 입력 순서 그대로이며 바뀌지 않는다(ezphot 내부 함수가 워커에서 찍는 출력은 섞일 수 있다).
  장을 그리다 단계 밖에서 난 예외(FITS 읽기 실패 등)는 이전처럼 갤러리 전체를 멈추지만, 병렬일 때는 이미 시작한 다른 장의 그림 파일이 남는다.
  데몬 프로세스(`multiprocessing.Pool` 워커 등) 안에서 2장 이상으로 `visualize()`를 부르면 하위 프로세스를 만들 수 없어 `AssertionError`가 난다(0.5.1에서는 동작하던 호출) — 그런 곳에서는 `n_proc=1`로 부른다.
- `recipes.visualize`: 과학 영상 전체를 float32로 변환하는 것은 전처리 그림을 실제로 그리는 장(master bias·dark·flat 파일이 있을 때)에서만 한다. 다른 단계(WCS·마스크·배경·측광)는 영상 크기와 컷아웃만 쓰므로
  memory-map된 `fits.getdata` 배열에서 컷아웃이 걸친 행 범위만(mmap 페이지 단위) 읽어 변환한다(압축·BZERO/BSCALE/BLANK 영상은 이전처럼 `fits.getdata`가 전체를 읽는다). 그림은 바뀌지 않는다.

## 0.5.1 (2026-10-30)
### 추가
- `recipes.visualize`: `lightcurve`(`--lightcurve`)를 주면 갤러리 `index.html`에 인터랙티브 '광도곡선'·'SED' 탭이 생긴다(인라인 JS·SVG, 외부 의존 없음).
  `MAGSKY_APER_3`(10″)·`MAGERR_APER_3`를 쓰고, 같은 시간 bin(`floor((mjd − 최초 mjd) / 폭)`, 기본 1일, 페이지에서 폭 변경)·같은 필터(유닛 무관) 점을
  등급 공간 역분산 가중 평균으로 합쳐 보여 준다(오차 max(√(1/Σw), std/√n); 표에 관측소가 여럿이면 이름이 같은 필터도 관측소별로 따로 둔다). 합치기는 페이지 표시용이며 측광표·카탈로그·수치는 바뀌지 않는다.
  필터 파장은 `SYNPHOT_FILTERDIR`의 `<observatory>_<filter>.csv`, 없으면 `<filter>.csv` 투과곡선의 응답 가중 평균(nm), 곡선이 없으면 `m###` 이름 숫자(페이지에 '추정' 표시).
  `MAGERR ≥ 99`·NaN 행은 빼고 그 수를 페이지에 적는다. 페이지 데이터에는 프레임 이름(stem)만 들어가고 경로는 들어가지 않는다.
  `lightcurve`를 주지 않으면 페이지 표시는 이전과 같다. `visualize()` 시그니처·반환, CLI, `lightcurve.png`·`thumb_lightcurve.jpg`·`run_trends.png`, `manifest.json` 키는 그대로다.

## 0.5.0 (2026-10-30)
### 수치 변경
- SCAMP 기준 카탈로그 기본값 변경(기본 동작 변경): `ImageMethod.platesolve`·`Platesolve.solve_scamp`·`PhotometryHelper.run_scamp`가
  기본으로 로컬 `GAIAXP_CORR_LAMOST` 타일 캐시(`CATALOG_DIR`)에서 만든 기준 카탈로그(`-ASTREF_CATALOG FILE`, FITS_LDAC, SCAMP 임시 디렉터리)를 쓴다.
  Gaia DR3 위치(J2016.0)를 이미지 관측 시각(`DATE-OBS`, 없으면 `MJD-OBS`; 여러 장이면 평균)으로 pmra/pmdec 전파하고(고유운동 없는 별은 그대로),
  고유운동 컷 없이 캐시의 G≲17.6 별을 모두 쓴다. 기준 별 위치 오차는 Gaia 오차(2 mas)와 캐시 좌표의 float32 양자화(간격/√12, RA는 ×cos δ)를 제곱합한 별별 값이다.
  카탈로그에는 PM = 0, OBSDATE = 전파한 epoch 열도 들어 있어 SCAMP가 다시 전파해도 위치가 바뀌지 않는다.
  모든 이미지 footprint의 50% 이상을 타일이 덮을 때만 로컬을 쓰고, 아니면(캐시·관측 시각 없음 포함) 이전처럼 `catalog_type`(기본 `GAIA-DR2`)을 SCAMP가 원격 조회한다(혼합하지 않음).
  원격 조회가 실패하면 이전처럼 `ExternalToolError`가 나고, 메시지 앞에 어떤 기준 카탈로그를 왜 썼는지가 붙는다.
  같은 입력에서 SCAMP WCS(PV)가 바뀌므로 좌표·매칭·`n_ref`·ZP가 달라질 수 있다. 영향 크기(T05785 baseline 6장, 7DT07 m675·7DT09 m775, 원격 GAIA-DR2 SCAMP 해 대비): WCS 위치 차 중앙값 0.04–0.05″, 95% 0.09–0.12″, 최대 0.22″(모서리); ZP_APER_3 |Δ| ≤ 0.07 mmag, n_ref ±1. SCAMP 외부 잔차(GD71 1장) 0.04″.
  이전 동작은 `local_reference=False`.
- 이 기본값을 이어받는 `recipes.process_images`(platesolve 단계)와 `Reproject.align(platesolve=True)`의 결과도 바뀐다.
  기본 경로는 장마다 `CATALOG_DIR`의 `summary.ascii_fixed_width`와 겹치는 타일 CSV를 읽는다(원격 조회·캐시 쓰기는 없음, 장당 읽기 시간이 늘어난다).
### 파괴적
- `catalog_type`을 명시해도 로컬 타일이 이미지를 덮으면 로컬 카탈로그가 우선한다. `catalog_type`은 이제 로컬을 쓰지 못할 때의 원격 카탈로그를 뜻한다
  (`None`이면 SCAMP 설정 파일의 `ASTREF_CATALOG`). 원격을 강제하려면 `local_reference=False`, `scamp_params`의 `ASTREF_CATALOG`는 여전히 모든 것에 우선한다.
  deprecation 기간 없이 바로 바꿨다: 기본값 변경은 사용자 결정이고, 기존 기본 경로(SCAMP의 원격 VizieR 조회)가 이 환경에서 막혀 platesolve가 실패하기 때문이다.
### 추가
- `ImageMethod.platesolve`, `Platesolve.solve_scamp`, `PhotometryHelper.run_scamp`에 키워드 인자 `local_reference: bool = True`(각 시그니처의 맨 끝, 기존 위치 인자 순서 유지).

## 0.4.17 (2026-10-29)
### 수정
- 보안: `ezphot.helper.queryhelper.Queryhelper.query_atlas_lightcurve`의 `token` 기본값을 소스에 박힌 토큰에서 `None`으로 바꿨다(동작 변경).
  `token`을 넘기지 않으면 환경 변수 `ATLAS_TOKEN`, 그다음 `~/ezphot/config/atlas.config`(JSON, 키 `ATLAS_TOKEN`)에서 읽고,
  어디에도 없으면 네트워크 요청 전에 `ValueError`(설정 파일 경로와 키 이름 안내)를 낸다.
  이전에 `token` 없이 호출하던 코드는 위 둘 중 하나를 설정해야 한다. 수치 변경 없음.
- 보안: 비공개 메서드 `Queryhelper._request_atlas_token`을 삭제했다. 받아 온 토큰을 반환하지 않아 쓸모가 없었고(호출처 없음),
  소스에 계정 기본값을 담고 있었다. ATLAS 토큰은 서비스에서 직접 발급받아 위 설정으로 넘긴다.
- 보안: 0.4.16 이하의 `helper/queryhelper.py`에 하드코딩되어 있던 ATLAS 사용자 이름·비밀번호·토큰은 공개 git 이력에 남아 있다.
  노출된 것으로 보고 서비스 쪽에서 교체해야 하며, 이전 기본값에 기대던 사용처는 교체된 토큰을 위 설정으로 넘겨야 한다.
