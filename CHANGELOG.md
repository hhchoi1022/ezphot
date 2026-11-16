# Changelog

## 0.8.0 (2026-11-16)
### 파괴적
- dia(`ra`/`dec`를 준 실행): 차영상 헤더에 유한한 `ZP_APER_3`가 없으면 그 행은 이제 `success=False`, `step='apply_zp'`(오류 문구는 forced_photometry와 같음)이고
  값 열은 비어 있다. 0.7.12는 성공에 빈 값이었다. 같은 입력에서 `success`·`step`과 CLI 종료 코드(0 → 1)가 바뀌므로 계약의 실패 표현 변경이다.
  `ra`/`dec` 없이 돌린 dia는 그대로다(그때는 ZP를 보지 않는다).
  deprecation 기간 없이 바꾸는 이유: 사용자 결정(D4)이고, 값이 빈 행을 성공으로 두면 하류가 NaN을 측정값으로 받는다. 옛 동작을 고르는 인자는 두지 않는다.
  - 갤러리: `visualize --dia`는 그런 프레임의 dia.png를 그리지 않고 상태에 "DIA failed at apply_zp: …"를 적는다(0.7.12는 그림만 그리고 점은 없었다).
  - **BRIDGE 영향**: ZP 없는 차영상 행을 실패로 받는다(종료 코드 1). 성공 행만 쓰는 처리라면 결과는 같고, 실패 집계가 늘 수 있다.
### 수치 변경
- **차영상(`sub_*`) forced 측광**(dia 레시피, forced_photometry 레시피에 차영상을 줄 때): 새 config 섹션 `forced_photometry_sub`
  (`defaults.FORCED_PHOTOMETRY_SUB = dict(annulus_width_arcsec=60, annulus_errors=False, n_empty_apertures=500, host_poisson=True)`)을
  `forced_photometry` 위에 얹는다(같은 키는 이쪽이 이긴다). 차영상이 아닌 영상의 forced 측광, dia의 후보·트랜지언트 카탈로그, 차영상 자체는 바뀌지 않는다.
  - 국소 배경: 타겟마다 고리(안쪽 지름 = 가장 큰 개구 + 1.5 seeing, 바깥 = 안쪽 + 60″; seeing 2.5–4.9″에서 반경 약 9–12″ … 39–42″)의
    5σ 클립 평균 × 개구 면적을 플럭스에서 뺀다. hotpants의 1차 평면 배경이 남긴 차영상 pedestal(T08635 고리 평균 −1.56 … +1.52 ADU/pix)을 없앤다.
    고리는 플럭스에만 쓰고 `SKYSIG`·`UL3`/`UL5`·`FLUXERR`의 배경 항은 고리 std가 아니라 지금처럼 bkgrms 적분이다.
  - 배경 잡음 실측 비 r_k: 같은 영상의 무작위 빈 개구(최대 500, 고정 시드)에서 타겟과 같은 방식으로 잰 값의 산포 ÷ bkgrms 예측.
    `SKYSIG_APER_k`·`UL3/UL5_APER_k`(→ `UL3SKY`/`UL5SKY`)·`FLUXERR_APER_k`의 배경 항에 곱한다(상관 잡음). 빈 개구의 소스 회피는 `conv_`가
    convolve된 reference일 때만 한다(그 밖에는 무효 픽셀 회피와 4σ 클립뿐, 계약 3절). `host_poisson=False`이면 `conv_`를 열지 않으므로
    소스 회피도 없다.
  - 호스트 Poisson: 같은 폴더 `conv_<name>`(convolve된 reference, `CONVOL00=TEMPLATE`)의 개구 플럭스(자기 고리 평균을 뺀 값) ÷ EGAIN을
    `FLUXERR`/`MAGERR` 분산에만 더한다. 한계 등급에는 넣지 않는다. `conv_`가 없거나 convolve된 science이거나 그 타겟에서 유한하지 않으면 넣지 않고
    요약 `host_source`(`missing`/`not template`/`not finite`)와 카탈로그 `HOST_FLUX_APER_k`(NaN)에 남긴다.
  - 정의는 `RECIPES_CONTRACT.md` 3절. 0.7.12 값을 다시 내려면(기본 `forced_photometry` 섹션 기준) `config={'forced_photometry_sub':
    dict(annulus_width_arcsec=None, n_empty_apertures=None, host_poisson=False)}`. `forced_photometry` 섹션에 `annulus_width_arcsec=X`를 직접 주던 호출은
    차영상에서 이제 그 값이 이 섹션 값으로 덮이므로, 0.7.12 값은 `forced_photometry_sub=dict(annulus_width_arcsec=X, annulus_errors=True,
    n_empty_apertures=None, host_poisson=False)`로 낸다.
  - 호스트 항은 T08635에서 `FLUXERR` 분산의 0.07/0.31/0.43(최소/중앙/최대)을 차지한다. `host_source`가 `conv`가 아닌 행(`missing`·`not template`·
    `off`·`not finite`)은 이 항이 없어 오차가 작게 나오므로, 한 광도곡선에 섞으면 오차가 일관되지 않는다.
  - 자원(T08635 실측, 60.9 Mpx 차영상 60장, 4개씩): 워커 최대 RSS는 변하지 않았다(3.4–3.7 GB). 시간은 영상당 약 +1.5 s(+4 %)이고,
    60장 forced가 540 s에서 562 s가 됐다.
  - **BRIDGE 영향**: `sub_*.fits.circ.cat`과 dia 요약에서 바뀌는 열. 숫자는 T08635 DIA 60장·20필터 격리 검증의 실측(새 − 0.7.12, 최소/중앙/최대)이다.
    | 열 | 변화 |
    | --- | --- |
    | `MAGSKY_APER_3` 프레임(`MAG`·`FLUX_APER_3`, 다른 구멍도) | 국소 배경만큼. 청·녹(m400–m725, r; 42장) −0.078/−0.009/+0.071 mag(\|Δ\| 중앙 0.017), 적색(m750–m875; 18장) −0.122/+0.015/+0.478 mag(\|Δ\| 중앙 0.066, 프레임 SNR 1.7–12). 새 프레임 오차 단위로 −1.40/−0.13/+1.16σ |
    | `MAGSKY_APER_3` 3장 결합(플럭스 역분산 가중) | 20필터 −0.100/−0.008/+0.228 mag. m800 −0.100, m825 −0.039, m850 +0.103, m875 +0.228(등급 가중이면 −0.101, −0.031, +0.086, +0.166). 적색은 SNR이 낮아 결합 방식에 따라 0.1 mag 넘게 달라진다 |
    | `BACKGROUND`(새 열, 타겟별 고리 평균) | −1.56/−0.19/+1.52 ADU/pix |
    | `FLUXERR_APER_3` | ×1.188/1.356/1.448(r_3와 호스트 항). 3장 결합 오차 ×1.185/1.360/1.544 |
    | `MAGERR_APER_3` | ×1.134/1.353/1.940(= FLUXERR 비 × FLUX_0.7.12 / FLUX_new; 최대는 m875 f1, SNR 1.7) |
    | `SKYSIG_APER_3` | × noise_ratio = 1.117/1.214/1.376(r 표본 오차 약 3 %, N = 500) |
    | `UL5SKY_APER_3`·`UL3SKY_APER_3`(`UL5`/`UL3_APER_3`) | 0.120/0.211/0.346 mag 얕아짐(= 2.5 log10 r_3, 호스트 항은 넣지 않음) |
    | `X_WORLD`, `Y_WORLD`, `ZP_APER_3`, `ZPERR_APER_3` | 그대로 |
    | 새 열 | `BACKGROUND`(ADU/pix), `NOISE_RATIO_APER[_k]`(r_k), `HOST_FLUX_APER[_k]`(호스트 항에 쓴 `conv_` 플럭스, ADU) |
    검증 전 이 항목에 있던 기대값(지난 QA 기준의 결합값과 FLUXERR·MAGERR 비)은 지웠다. 지난 QA의 3σ 클립 pedestal이 0.13 ADU/pix 낮게 편향돼 있었고,
    새 5σ 클립 고리 평균은 대상 근처 빈 개구 평균과 편향이 없다(+0.000 ± 0.018 ADU/pix).
    정규화 잔차는 값에 보정하지 않고 `norm_*` 열로만 알린다(아래 `### 추가`).
    BRIDGE는 수용 검사에서 위 열의 분포를 다시 보고, 한계 등급(`UL5SKY_APER_3`)이 빈 개구 실측 기준으로 얕아진 것을 반영해야 한다.
### 추가
- `AperturePhotometry.circular_photometry`·`ImageMethod.photometry_forced_circular`에 새 인자(맨 끝, 기본 = 이전 동작): `annulus_errors=True`
  (False면 고리를 배경 수준에만 쓰고 오차·한계 등급은 bkgrms로), `n_empty_apertures=None`(N이면 빈 개구 실측 잡음 비 r_k로 `SKYSIG`/`UL`/`FLUXERR`
  배경 항을 키우고 `NOISE_RATIO_APER_k` 열을 쓴다), `host_img=None`(주면 그 영상 개구 플럭스의 Poisson 잡음을 `FLUXERR`/`MAGERR`에 더하고
  `HOST_FLUX_APER_k` 열을 쓴다; 고리가 없으면 그 영상의 배경이 0이라고 본다).
  뒤의 둘은 bkgrms 오차에만 적용되며, 고리와 함께 쓰면서 `annulus_errors=True`이면 `ValueError`.
- dia 요약 끝에 새 열: `ra`, `dec`(넘긴 좌표), `noise_ratio`, `host_source`(차영상 forced, forced_photometry와 같은 뜻), 정규화 QA
  `norm_resid`(필드 별 10″ 플럭스 비 차영상/science의 중앙값), `norm_resid_n`(쓴 별 수, 정수), `norm_resid_mad`, `ksum`(헤더 `KSUM00`),
  `ksum_exp`(=10^(−0.4(ZP_ref−ZP_sci)), `CONVOL00=TEMPLATE`일 때만), `ksum_ratio`, `norm_flag`(|`norm_resid`| > `defaults.DIA_NORM_RESID_MAX` = 0.035).
  별은 science 영상 옆 `<img>.fits.refcat`(ZP 기준성). 별이 10개 미만이면(refcat·`sci_`가 없을 때 포함) `norm_resid`·`norm_resid_mad`·`norm_flag`는
  빈 칸이고(플래그를 세우지 않음) `norm_resid_n`은 그 별 수다. QA 열은 값과 `success`를 바꾸지 않고, 파일을 쓰거나 지우지 않는다. 기존 열의 이름·순서는 그대로다.
  `norm_resid`는 밝은 refcat 별(적색 필터에서 약 10–14 mag) 기준이라, 적색에서는 어두운 천체보다 0.5–1 % 낮게 나올 수 있다(계약 2절 dia).
  - **BRIDGE 영향**: `norm_flag`가 참인 차영상은 hotpants 플럭스 스케일이 3.5 % 넘게 어긋난 것이다. T08635 검증에서 플래그는 m650 f0,
    m675 f1·f2 세 장이었다(norm_resid 약 −4 %, +6 %, +5.5 %, 지난 QA와 0.004 안). **그 프레임의 `MAGSKY`는 보정하지 않는다.**
    T08635에서 m650 f0은 약 0.08 mag 어둡고 m675 f1·f2는 약 0.11 mag 밝았다(10″ 안 호스트/트랜지언트 1.6–1.8배). 사용처가 그 프레임을 빼거나 보정해야 한다.
- forced_photometry 요약 끝에 `noise_ratio`, `host_source`(타겟별, 차영상만, 그 밖은 빈 칸).
  - **BRIDGE 영향**: forced_photometry 요약(스택·`reference_stars` 포함)에 끝 열 둘이 늘어난다(차영상이 아니면 빈 칸). 열 집합을 엄격히 검사하면 맞춘다.
- dia·forced_photometry의 새 config 섹션 `forced_photometry_sub`(`defaults.FORCED_PHOTOMETRY_SUB`, 차영상에만 `forced_photometry` 위에 얹음, 같은 키는 이쪽이 이긴다)와
  `defaults.DIA_NORM_RESID_MAX`(0.035, config 섹션이 아닌 상수).
- 새 예외 `ezphot.recipes.NothingToDo(ValueError)`: 처리할 것이 없을 때(입력이 빔, 성공 행 없음, 스택할 묶음 없음, 카탈로그 없음). 메시지는 그대로다:
  process_images `no images to process`, stack `no successfully processed images in the summary`·`no group has more than one image`,
  dia `no successful images in the summary`, forced_photometry `no images given`, target_photometry `no catalogs given`(일관성을 위해 함께).
  `except ValueError`는 그대로 잡는다. 계약 1절에 메시지 표.
  - **BRIDGE 영향**: 메시지 앞부분 비교 대신 `except NothingToDo`로 "처리할 것 없음"을 가를 수 있다.
- `ImageMethod.correct_bdf(..., master_max_days=100)`, `process_image`·`process_images(..., master_max_days=100)`, CLI `process_images --master-max-days`:
  `do_preprocess`에서 master bias·dark·flat 세 가지를 찾는 기간(일). 기본 100 = 이전 고정값(결과 같음).
  - **BRIDGE 영향**: 1000일 등 긴 창이 필요하면 이 인자로 레시피 전처리를 쓸 수 있다.
### 변경
- dia: 차영상 forced가 `recipes.forced_photometry.forced_image`(`calibrate='header'`)를 부른다(같은 호출이라 위 수치 변경과 `### 파괴적`을 빼면 값과
  `sub_*.circ.cat`은 같다). 행의 `telname`·`filter`·`obsdate`·`exptime`은 그대로 science 영상 값이다. forced의 ZP 외 실패는 `step='forced_photometry'`.
- dia: 영상을 열거나 헤더 값을 읽지 못하면 실행 전체가 예외로 멈추던 것을, forced_photometry처럼 그 영상 행만 `step='load'` 실패로 남기고 계속한다
  (요약 CSV가 생기고, n_proc > 1에서 나머지 영상이 취소되지 않는다). 정상 입력의 결과는 같다.
  - **BRIDGE 영향**: 열 수 없는 영상은 예외가 아니라 `success=False, step='load'` 행으로 온다. CLI 종료 코드는 그대로 1이다. 예외로 이 실패를 잡던 Python 호출은
    `success`를 본다.
- dia: `ra`/`dec`는 타겟 하나다. 숫자 또는 원소 하나짜리 리스트·튜플·배열을 받고, 그 밖(원소 둘 이상, 숫자가 아님)은 처리 전에 `ValueError`다
  (0.7.12는 리스트를 받아 첫 타겟 값을 썼다). `dia_image`는 그대로 예외를 내지 않는다. 계약 밖 입력이라(0.7.12 계약·주석 타입은 float,
  CLI는 `type=float`) 파괴적으로 분류하지 않는다.
- forced_photometry: 차영상의 `bkg_source`가 고리를 쓰면 `annulus`(0.7.12: `none`).
### 수정
- `visualize --dia`: forced 위치를 `sub_*.circ.cat`의 첫 행이 아니라 dia 요약 `ra`/`dec`에서 1″ 안의 행에서 가져온다. 다른 좌표의 forced_photometry가
  그 파일을 덮어써서 맞는 행이 없거나 요약의 `ra`/`dec`가 비어 있으면 그 프레임의 DIA 그림·점을 그리지 않고 상태에 그 이유를 적는다.
  이유는 프레임 폴더 `dia.json`의 `nopos`(내부 형식)에도 남아 `--dia` 없이 갱신해도 같은 메모가 나온다(`nopos`가 없는 옛 `dia.json`은 이전 문구).
  `ra`/`dec` 열이 없는 옛 요약은 이전처럼 첫 행. 옛 요약과 새 요약은 따로 넘긴다(한 DataFrame으로 합치면 옛 행은 `ra`가 빈 칸이 되어 위치가 없다).
- `circular_photometry`·`photometry_forced_circular` docstring: `annulus_width_arcsec`는 고리 바깥 지름 = 안쪽 지름 + 이 값(반경 폭은 절반)이다.

## 0.7.12 (2026-11-15)
### 추가
- 새 레시피 `recipes.forced_photometry`: 단일·스택·차 영상에서 주어진 좌표(스칼라 또는 배열)에 forced 원형 측광을 하고 ZP를 붙인다.
  Python `forced_photometry(images, ra, dec, calibrate='header', config=None, n_proc=4, out=None, verbose=True)`(영상×타겟 한 행의 DataFrame,
  CSV 저장), 영상 하나 단위 `forced_image(target_img, ra, dec, calibrate='header', config=None, clear=True)`(타겟별 행 목록),
  CLI `python -m ezphot.recipes.forced_photometry <image.fits | summary.csv> ... --ra ... --dec ... [--calibrate header|reference_stars] [--n-proc] [--out]`.
  `ezphot.recipes`가 두 함수를 export한다. 이 레시피를 더하는 것만으로 기존 레시피의 동작과 값은 바뀌지 않는다
  (`photometric_calibration`의 변경은 아래 `### 수정`, target_photometry 검색은 `### 변경`).
  - 입력: FITS 경로와 process_images/stack 요약 CSV(`success` 행의 `path`)를 섞어서(경로는 resolve해 중복을 뺀다). 한 영상의 모든 타겟을 한 번의 forced 호출로 잰다.
  - `calibrate='header'`(기본): `photometry_forced_circular` → `apply_zp`(영상 헤더 ZP). dia가 차영상에서 하는 것과 같은 호출이고, 차영상(`sub_`)에서는
    배경도 dia처럼 넣지 않으므로 같은 차영상·좌표면 dia의 forced 값과 같다.
  - `calibrate='reference_stars'`: `<img>.fits.refcat`의 기준성 위치와 타겟을 한 번에 forced로 재고 그 카탈로그로 `photometric_calibration`
    (defaults `PHOTCAL`, 기준성 선택 등급 `MAG_APER_3`). 레시피가 고정하는 키(config로 바꿀 수 없음): `update_header=False`, `save_image=False`, `save=True`,
    `mag_lower`/`mag_upper`(`defaults.MAG_RANGE`), 영역 컷 없음(`ra_deg=dec_deg=None`, `inner_fraction=1.0` — refcat은 이미 process_images의 영역 컷을 거쳤다).
    forced 섹션은 `save=True` 고정. refcat이 없으면 그 영상은 `step='refcat'` 실패.
    forced 카탈로그에 없는 열로 하는 별 판별·FWHM·플래그·SATDIST 컷은 빠진다(`RECIPES_CONTRACT.md` 5절).
  - 배경: 차영상은 없음, 그 밖은 영상 옆 `.bkgmap`/`.bkgrms` 사이드카(헤더 ZP를 정한 측광의 배경). 없으면 영상에서 메모리로 계산(소스 마스크 없음)하고
    요약 `bkg_source`/`bkgrms_source`에 `sidecar`/`image`/`none`으로 적는다.
  - 산출물: 영상과 같은 폴더의 `<img>.fits.circ.cat`(+ `.info`, ASCII 표), `reference_stars`면 `<img>.fits.circ.refcat`(+ `.info`)도. 그림은 만들지 않는다.
    입력 FITS는 다시 쓰지 않는다. 같은 영상에 이미 있는 `.circ.cat`(+ `.info`, 다른 도구가 만든 것 포함)은 덮어쓴다. 영상 폴더에 남아 있던 `subbkg_<img>*` 파일은
    지운다(배경을 뺄 때의 임시 객체 정리, process_images 측광과 같은 동작). `reference_stars` 뒤에 `header`로 다시 돌리면 이전 `.circ.refcat`이 낡은 채 남는다
    (지우지 않는다). 카탈로그 자체에는 어느 방식의 ZP인지 기록이 없다(요약 `calibrate` 열). 입력이 심볼릭 링크면 따라간 실제 파일의 폴더에 쓰고
    그 폴더의 `subbkg_<img>*`를 지운다. 요약 `path`는 resolve된 절대 경로다.
  - 차영상(`sub_`)에 dia와 다른 좌표로 돌리면 dia가 쓴 `sub_*.fits.circ.cat`을 덮어써서, `visualize --dia`가 그리는 forced 위치(그 파일의 첫 행)가 바뀐다
    (값은 dia 요약에서 오므로 위치와 값이 어긋날 수 있다).
  - 요약 열: `path`, `target`, `ra`, `dec`, `telname`, `filter`, `obsdate`, `exptime`, `calibrate`, `success`, `step`, `error`, `n_ref`, `bkg_source`, `bkgrms_source`,
    `catalog`, `MAGSKY_APER_3`, `MAGERR_APER_3`, `ZP_APER_3`, `ZPERR_APER_3`, `UL5SKY_APER_3`. `out=` 기본 `~/ezphot/log/recipes/<objname>/forced_photometry_<stamp>.csv`,
    실행 중 `<out>.progress.json`. 성공 판정은 카탈로그의 `ZP_APER_3`가 있고 유한한가(`reference_stars`는 `n_ref > 0`도)다. 아니면 그 영상 행은
    `step='apply_zp'`(헤더에 `ZP_APER_3` 없음, dia는 이 경우 성공에 빈 값) 또는 `step='photometric_calibration'`(기준성 0개 등) 실패다.
    영상을 열 수 없으면(파일 없음, 망원경 판별, 헤더 값 해석) 비대화 실행에서는 멈추지 않고 그 영상 행을 `step='load'` 실패로 남긴다
    (터미널·Jupyter의 순차 실행이면 ezphot이 망원경 등록 입력을 먼저 묻는다). `n_ref`는 정수 열(실패 행은 빈 칸)이다.
  - `recipes.status`가 실행 중·중단된 forced_photometry 실행도 보여 준다(stack·dia와 같은 방식).
  - **BRIDGE 영향**: forced 카탈로그를 직접 만들던 경로를 이 레시피로 바꿀 수 있다. 카탈로그 읽기 계약(파일 이름·위치·형식, `X_WORLD`/`Y_WORLD`/
    `MAGSKY`/`MAGERR`/`ZPERR`/`UL5SKY_APER_3`, info `obsdate`·`filter`·`observatory`·`telname`, 행 순서가 아니라 좌표 매칭)은 `RECIPES_CONTRACT.md` 4절.
- `photometric_calibration`(`methods.PhotometricCalibration`, `ImageMethod` 래퍼)에 새 인자 `save_image=True`(맨 끝). 기본값은 이전과 같이 끝에서 영상(FITS와 `.info`)을
  savepath에 쓴다. `False`면 영상 파일을 쓰지 않는다(`update_header=True`여도 헤더는 메모리에서만 바뀐다). 이전에는 `update_header=False`여도 항상 썼다.
- 새 문서 `RECIPES_CONTRACT.md`(저장소 루트): process_images·stack·dia·forced_photometry·target_photometry·qa의 시그니처, 요약 열, `out=` 규칙, 실패 표현,
  구멍 번호의 뜻(`APER_3` = 10″, 인덱스 0은 접미사 없음, seeing 배수 구멍은 뒤에 붙고 영상마다 다름), forced 카탈로그 읽기 계약, 이 계약을 바꿀 때의 규칙.
  - **BRIDGE 영향**: 앞으로 이 문서의 항목(요약 열, 구멍 목록, forced 카탈로그 형식 등)을 바꾸면 이 CHANGELOG에 분류와 사용처 영향을 적는다.
### 변경
- `recipes.target_photometry`: DataBrowser 검색(`load_catalogs`, CLI에서 `--paths` 없이)이 파생 접두어에 더해 forced 카탈로그 `*.circ.cat`도 버린다.
  forced_photometry를 단일 영상에 돌리면 접두어 없는 `<img>.fits.circ.cat`이 생겨 기본 패턴 `*.cat`이 같은 에포크를 두 번 잡기 때문이다.
  접두어 없는 `.circ.cat`이 없는 폴더에서는 결과가 이전과 같다(지금까지 레시피가 만든 `.circ.cat`은 `coadd_`·`sub_` 접두어라 이미 걸러졌다).
  `--pattern`(`pattern=`)이 `.circ.cat`으로 끝나면 forced 카탈로그를 버리지 않는다. `--paths`나 `catalogs`로 직접 준 카탈로그는 그대로 쓴다.
### 수정
- `photometric_calibration`: 카탈로그에 `SKYSIG` 열이 **없을 때**(forced `.circ.cat` 등) `UL3SKY_APER[_k]`/`UL5SKY_APER[_k]`가 빠지던 것을 고침.
  `apply_zp`와 같은 순서로, 구멍마다 카탈로그의 `UL3_APER[_k]`/`UL5_APER[_k]` 열이 있으면 그 값 + ZP(행별), 없으면 영상 info `SKYSIG`로
  `ZP − 2.5 log10(n · SKYSIG · √mean(NPIX))`(n = 3, 5). 카탈로그(와 refcat)에만 쓰고, 헤더 갱신용 `update_kwargs`(반환 네 번째 값)에는 넣지 않는다 —
  헤더의 `UL3SKY`/`UL5SKY`·`SKYSIG`는 그 영상의 SExtractor 보정 값으로 남는다. `SKYSIG` 열이 있는 카탈로그(process_images·stack의 SExtractor `.cat`)는
  값·헤더 모두 이전과 같다. info에 `SKYSIG`가 없는 영상(헤더에 없음, CalibrationImage)이고 `UL*_APER` 열도 없으면 이전처럼 한계 등급 열이 없다.
  - **BRIDGE 영향**: 스택 forced 카탈로그(`photometric_calibration` 경로)에 이제 `UL5SKY_APER_3`가 생긴다(전에는 없었다). 새로 생기는 값이고
    기존 값은 바뀌지 않는다. 같은 카탈로그를 `apply_zp`로 보정했을 때와 같은 식(`UL5_APER_3` + ZP)이다.

## 0.7.11 (2026-11-14)
### 추가
- `recipes.visualize` 갤러리에 **DIA**(dia 레시피 결과): 새 인자 `visualize(..., dia=None)`(맨 끝, CSV 경로·DataFrame 또는 그 목록),
  CLI `--dia a.csv [b.csv ...]`(dia 레시피 summary CSV). 넣지 않고 프레임 폴더에 `dia.json`도 없으면 페이지 데이터(`const D`)·산출 파일·
  manifest·`.viz_done.json`은 이전과 같다(페이지 JS가 늘어 `index.html` 바이트만 다르다). 측광 값과 레시피 출력은 바뀌지 않는다.
  - 매칭: 행의 `path`가 갤러리 프레임의 path와 같으면 그 프레임, 아니면 파일명이 같은 유일한 프레임(처리 영상 사본에서 돌린 DIA).
    어느 프레임에도 맞지 않는 행(스택 DIA 등), 갤러리에 같은 파일명이 여럿인 행, 한 프레임에 맞는 행이 둘 이상인 경우는 쓰지 않고
    개수를 로그와 개요 부제(`DIA n프레임, … 건너뜀`)에 적는다. 한 프레임에 행이 여럿이면 그 프레임 상태 칸에도 적는다.
  - 새 탭 "차영상(DIA)": 프레임마다 `dia.png`(+ `thumb_dia.jpg`). science(`sci_`)·reference(`ref_`)·convolved(`conv_`)·차영상(`sub_`) 4칸,
    forced 위치(`sub_<name>.circ.cat`의 `X_WORLD`/`Y_WORLD`) 중심 한 변 약 60″, 10″ 지름 개구 원(빨강, 차영상 헤더 WCS).
    칸마다 ZScale, 차영상은 0 중심 대칭. convolved 칸 제목은 차영상 헤더 `CONVOL00`(TEMPLATE: convolved reference, IMAGE: convolved science).
    실제 PNG로 쓴다(링크가 아님). 영상은 읽기 전용 memmap의 `.section`으로 잘라낼 부분만 읽고 ezphot 영상 객체를 만들지 않는다.
    탭 위에 그 프레임의 DIA 값(`MAGSKY_APER_3` ± `MAGERR_APER_3`, `UL5SKY_APER_3`, `ZPERR_APER_3`, reference 파일명, convolve 방향, 고정 좌표, CSV 파일명).
  - 광도곡선·SED 탭: **DIA 계열**(빈 마름모 ◇, "DIA" 체크박스, 기본 켜짐). 값은 dia summary의 `MAGSKY_APER_3`·`MAGERR_APER_3` 그대로다
    — 차영상 flux의 등급(총 flux 등급이 아님)이고 국소 배경 보정 전이다. 시각은 summary `obsdate`의 MJD. 프레임마다 점 하나이고
    GP 적합·합친 점·Δmag 기준(필터별 평균)·SED bin 평균에 넣지 않으며, Δmag 모드에서는 그리지 않는다. MAGERR ≥ 99 또는 NaN인 행과
    forced 위치가 없는 행(`.circ.cat` 없음, dia를 `--ra/--dec` 없이 돌린 경우)은 점이 없다. 필터 칩·필터별 패널은 이름이 같은 필터에 적용된다.
    DIA 점만 있는 시간 bin은 SED bin 목록에 "DIA 점만"으로 나온다. 체크박스를 끄면 DIA가 없는 보기(y 범위 포함)와 같다.
    페이지의 좌표 입력과 무관한 고정 좌표 계열이다(좌표를 적용하기 전인 카탈로그 매칭 모드에서는 탭에 안내문만 나온다).
    광도곡선·SED 탭은 DIA 점만 있는 갤러리에도 생긴다.
  - 새 산출물: 프레임 폴더 `<stem>/dia.png`, `<stem>/thumb_dia.jpg`, `<stem>/dia.json`(그 프레임의 DIA 값·파일명, 경로 없음),
    `.viz_done.json`의 `dia` 레코드, `manifest.json` `images[<path>]`의 `'dia'` 키(그림이 없으면 빈 목록). 페이지 데이터에 `dia`, `images[].dia`,
    `labels.dia`가 생긴다(내부 형식).
  - 나중 갱신: 프레임 폴더의 `dia.json` 덕분에 `--dia` 없이 다시 돌려도(모든 모드) 마지막 DIA를 보여 준다. DIA 산출물과 CSV를 지운 뒤에도 그렇다.
    이미 만든 갤러리는 `visualize <summary> --skip-unchanged --dia <csv> ...`로 DIA만 추가된다. 다른 단계 그림은 입력이 같으면 다시 그리지 않는다.
    sources.json·fov.jpg·index.html, 그리고 `--lightcurve`가 있으면 GP 적합은 지금처럼 매번 다시 한다(`--no-gp`로 뺄 수 있다).
    실패한 행, 차영상 파일이 없는 행, 모호한 행은 그 프레임의 기존 DIA를 바꾸지 않고 상태 칸에 이유를 적는다.
    쓸 수 있는 행인데 `dia.png`를 그리다 실패하면(산출물 일부 없음, forced 위치가 차영상 밖 등) 그 프레임 갤러리 폴더의 옛 `dia.png`·`thumb_dia.jpg`를
    지운다(갤러리 출력만). 그래서 이후 실행에서 옛 그림이 새 값과 함께 나오지 않고 상태 칸에 "dia figure missing"이 나온다.
    `--pages-only`(`pages_only=True`)는 `--dia`를 받지 않는다(오류).
  - 읽는 것: dia 레시피 summary 열 `path`, `success`, `subtracted`, `obsdate`, `filter`(필수, 없으면 `ValueError`)와 `step`, `error`, `telname`,
    `reference`, `MAGSKY/MAGERR/ZPERR/UL5SKY_APER_3`, DIA 산출물 이름 `sci_/ref_/conv_/sub_<name>`, `sub_<name>.circ.cat`, 차영상 헤더 `CONVOL00`.
    이 열이나 이름이 바뀌면 이 기능도 바꿔야 한다.
- `recipes.target_photometry`: 같은 관측의 중복 카탈로그를 거르는 선택 옵션 **dedupe**(기본 꺼짐, 끄면 카탈로그 선택과 표·CSV는 이전과 같다. 그림은 아래 `### 수정`의 `LightCurve.plot` 항목을 따른다).
  Python `target_photometry(..., dedupe=False)`(새 인자, 맨 끝), CLI `--dedupe`.
  - 켜면 카탈로그 info의 (observatory, telname, filter, obsdate)가 모두 같은 묶음마다 UL5가 가장 큰(가장 깊은) 카탈로그 하나만 남긴다.
    UL5는 그 카탈로그 `UL5SKY_APER_3` 열의 유한값 중앙값(10″ 개구 5σ 한계, 출력 표의 같은 열로 확인할 수 있다)이고, 열이 없거나 전부 NaN이면
    카탈로그 info의 `depth`다(info의 `depth`가 비어 있는 카탈로그도 있다). 둘 다 없으면 가장 얕은 것으로 치고, 같으면 입력 순서가 앞인 것을 남긴다.
    거르기 전에 모든 카탈로그의 표를 읽는다(그 뒤 측광에서도 읽는 표다).
    네 값 가운데 하나라도 없는 카탈로그는 거르지 않는다. obsdate는 문자열이 완전히 같아야 같은 묶음이다.
  - LightCurve를 만들기 전에 카탈로그 단계에서 거르므로 표 행과 `--plot` 그림이 모두 남긴 카탈로그로 만들어진다.
    입력이 `CatalogSet`이면 그 `target_catalogs`(선택된 카탈로그)에서 거른다. `verbose`면 남긴 수/전체 수와 뺀 파일명(최대 10개)을 한 줄 출력한다.
  - 용도: 같은 노출 시각(`DATE-OBS`)의 coadd가 두 번 만들어진 경우(예: 파일명 끝 `_0005`/`_0006`). 서로 다른 노출이 같은 obsdate를
    가질 수 있는 데이터에서는 정보를 잃으므로 켜지 않는다.
  - CLI의 DataBrowser 검색(`--paths` 없이)은 지금처럼 `coadd_` 등 파생 접두 카탈로그를 버린다. coadd 카탈로그에는 `--paths`나 Python 호출로 넘길 때 적용된다.
### 변경
- `recipes.visualize` 광도곡선·SED 탭의 **GP 곡선**: jitter를 적합하고, GP에서 크게 벗어난 점을 반복 클리핑으로 뺀다. 갤러리 페이지만 바뀐다.
  측광 값, target_photometry 표, summary CSV, 다른 산출물은 그대로다.
  - 적합: `gp.fit(jitter=True, nstart=10)`(이전: `jitter=False`). gpimg 기본 시작점·경계(flux 척도의 0.01배에서 시작, 1e-6–1배 안)이고,
    각 측정 분산은 오차² + j²다.
  - 클리핑: 남은 점(처음에는 전부)으로 적합 → 모든 중대역 점(빠진 점 포함)의 시각·밴드 파장에서 잡음 없는 flux를 예측(smoother `rts`) →
    z = (flux − 평균) / √(오차² + j² + 예측 분산) → |z| ≤ 5인 점만 남겨 다시 적합. 남는 점이 바뀌지 않거나 적합이 6번 돌면 멈추고,
    곡선은 마지막 적합의 것이다. 어떤 단계에서 모든 점이 빠지게 되면 그 직전 적합을 쓴다. 기준은 모듈 상수 `GP_CLIP_SIGMA = 5`,
    `GP_CLIP_MAXITER = 6`이고 CLI 옵션은 없다. 점이 모두 빠진 필터도 GP 필터 목록에 남고, 그 파장의 곡선은 다른 밴드에서 예측된다
    (광도곡선 탭에는 그 필터의 선을 그리지 않는다). 적합이 최대 6번 돌므로 GP 시간이 그만큼 길어질 수 있다.
  - 표시: GP 체크박스가 켜져 있는 동안(**기본으로 켜져 있다**), 빠진 점은 같은 기호로 흐리게 그린다(불투명도 0.3, 개별 점과 합친 점을 함께 볼 때 0.15, 오차 막대 포함).
    또 합친 점(bin 평균), 필터별 가중 평균(Δmag 기준), SED bin 평균에서 뺀다. "개별 점"을 끄고 합친 점만 볼 때도 그 점들을 그린다.
    - 빠진 점만 있는 SED bin은 bin 목록에 "GP에서 뺀 점만 (합친 점 없음)"으로 남는다(bin 시각 = 그 점들의 평균 시각).
    - 한 필터의 점이 모두 빠지면 그 필터의 Δmag 기준은 전체 점 평균이다.
    - 탭 설명·GP 캡션에 빠진 점 수, |z| 기준, 적합 횟수, jitter 값, 점이 모두 빠진 필터를 적고(캡션의 "jitter 없음"은 없어진다),
      빠진 점의 툴팁에 "GP 적합에서 뺀 점"을 적는다.
  - GP 체크박스를 끄거나 GP가 없으면(`--no-gp`, gpimg 없음·실패, 좌표 입력 매칭) 모든 점을 0.7.10과 똑같이 그리고 합친다.
    체크박스가 기본으로 켜져 있으므로, 빠진 점이 있는 표는 **기본 보기**의 합친 점·Δmag 기준·SED bin이 0.7.10과 다르다.
    0.7.10과 같은 보기는 GP를 끈다. 체크박스를 켜고 끄면 합친 점·Δmag 기준·SED bin이 그에 따라 다시 계산된다.
  - 예: LDS749B(gpimg 0.2.0, 갤러리 페이지 점으로 측정). 중대역 점 393개 가운데 2개(2026-10-30 이중상 프레임 두 장)가 빠지고 적합이 수렴한다:
    A 3530 μJy, l_t 12.0 d, l_λ 0.094, jitter 49.6 μJy. 0.7.10 페이지의 적합(jitter 없음, 모든 점)은 수렴하지 못했고, 길이 척도 둘 다
    하한(l_t 0.5 d, l_λ 0.03)에 붙어 있었다. 같은 측정에서 빠진 점은 GD71 0개, WD2149+021 0개, FEIGE110 1개, T22956 5개(1,953개 중)였다.
  - 페이지 데이터 `lcdata.gp`에 `x`(빠진 점의 `pts` 인덱스), `nx`, `clip`(`sigma`, `fits`, `stable`), `fit.jit`(μJy, 유한하지 않으면 null) 키가 생기고,
    `n`은 마지막 적합의 점 수다(내부 형식).
  - 이미 만든 갤러리는 `visualize --pages-only --lightcurve <표>`로 다시 만들면 반영된다.
- `recipes.visualize` 광도곡선·SED 탭의 **시야 이미지**: 좌표를 적용하면 그 위치로 옮기고 확대한다. 갤러리 페이지만 바뀐다
  (페이지 데이터, sources.json, fov.jpg, CLI, 다른 탭은 그대로).
  - 언제: "좌표 적용"("카탈로그 매칭으로 보기") 버튼, 좌표 칸의 Enter, 대상 이름 버튼, 탭을 처음 열 때(`--ra/--dec`가 있을 때).
    `--lightcurve` 표 모드에서도(`--ra/--dec`가 있을 때) 대상 이름 버튼과 처음 열 때 대상 좌표로 옮긴다(광도곡선은 표 그대로).
    시야 이미지를 클릭해 검출을 고를 때는 지금처럼 보기를 움직이지 않는다.
  - 배율: 전체 시야(처음, "전체 시야" 버튼 뒤, 휠로 끝까지 축소한 상태)면 폭 4′로 확대하고(휠과 같은 한계: 이미지 폭의 1/40 – 전체 폭),
    이미 확대해 두었으면 그 폭 그대로 중심만 옮긴다. 보기는 끌기·휠처럼 이미지 안에 머물러서 가장자리 근처의 좌표는 가운데에 오지 않는다.
  - 위치: 좌표에 가장 가까운 대표 프레임 검출(SNR > 3) 10개에 최소제곱 affine 변환(좌표를 접점으로 한 gnomonic ξ, η → 시야 이미지 px)을
    맞춰 구한다. 4′의 px 크기도 이 변환의 배율에서 정한다.
  - 표시: 적용한 좌표 위치에 빨간 원(클릭으로 고른 검출과 같은 원)을 그린다(이전: 클릭으로 고른 검출에만). RA·Dec 칸을 고치면 지금처럼
    원이 사라지고, 보기는 그대로다.
  - 시야 밖: 변환한 위치가 이미지 밖이거나, 가장 가까운 검출이 10′보다 멀거나, 가까운 검출이 3개 미만이거나 거의 한 줄이어서
    변환을 맞출 수 없으면 보기를 그대로 두고 빨간 원을 숨기며, 시야 이미지 카드에
    "좌표가 이 프레임의 시야 밖입니다"를 쓴다. 대표 프레임 하나 기준이라 다른 유닛·필터의 프레임에는 그 좌표가 있을 수 있고,
    광도곡선 매칭은 지금처럼 모든 프레임에서 한다.
  - 이미 만든 갤러리는 `visualize --pages-only`(같은 `--ra/--dec`, `--lightcurve`)로 다시 만들면 반영된다.
### 수정
- `PhotometryHelper.run_swarp(fill_zero_tonan=True)`(기본값)가 **입력 FITS를 더 이상 고치지 않는다**(이전: 입력 파일을 열어 0을 NaN으로 바꿔 제자리에 덮어썼다).
  0 → NaN은 출력 폴더(`target_outpath`의 부모) 아래 임시 하위 디렉터리 `.swarp_input_*`에 같은 파일 이름으로 만든 사본에서 하고,
  SWarp는 그 사본을 읽으며, 사본은 SWarp가 끝나면(실패해도) 지운다. 입력 옆에 SWarp 외부 헤더(`<name><HEADER_SUFFIX>`, 기본 `.head`)가 있으면 함께 복사한다.
  시그니처·반환·출력 파일 이름과 위치는 그대로다. SWarp가 읽는 픽셀 값은 이전과 같다. **수치 변화 없음**: 수정 전 코드와 같은 입력 사본으로
  비교해, 합성 영상 `run_swarp` 6사례(결합, 재표본만, `fill_zero_tonan=False`, 외부 헤더 있음·없음, BSCALE/BZERO 정수형)의 출력 배열(헤더 차이는
  `DATE`뿐), `Reproject.reproject`(에러맵 포함) 출력, T08635 7DT01 m400 단일 프레임 DIA의 `sub_/sci_/ref_/conv_` 배열과
  forced `MAGSKY/MAGERR/ZPERR/UL5SKY_APER_3`, 같은 밤 3장 stack의 coadd·`.bkgrms`·`.bkgmap`·summary가 모두 같았다.
  실패 사례(BZERO 32768 uint16 입력의 `ValueError`, 없는 weight 파일로 SWarp가 실패한 `ExternalToolError`)는 옛·새 코드가 같은 예외를 냈다
  (uint16 입력 한계는 이 변경 전부터 있던 것).
  새 코드에서는 입력 파일이 바뀌지 않았고(이전 코드는 그 DIA science의 0 픽셀 9401개를 NaN으로 바꿨다) refdata에 쓴 파일이 없었다.
  - 이 함수를 거치는 경로 모두 해당한다: `Reproject.reproject`/`ImageMethod.reproject`(기본 `fill_zero_tonan=True`, 에러맵용 두 번째 SWarp 호출 포함),
    `Subtract.find_transients`/`ImageMethod.DIA`와 `recipes.dia`(science·reference 재투영), `ImageSet.prepare_stack`/`Stack.prepare_images`의 재투영과
    `Stack.stack_swarp`(SWarp 결합), `recipes.stack`.
  - 사용처가 확인할 동작 변경: 위 처리 뒤에도 입력 science·reference·에러맵 파일의 0 픽셀이 NaN으로 바뀌지 않는다. 처리 뒤 입력 파일이
    NaN으로 바뀌어 있다는 데 기대던 코드는 고쳐야 한다. 이전 버전이 이미 0 → NaN으로 바꾼 입력 파일은 되돌리지 않는다. 그래서 같은 입력을 다시
    처리한 결과(예: process_images 재실행)는 이전 버전에서 입력이 NaN으로 바뀐 뒤 돌린 결과와 다를 수 있다.
  - 실행 중에는 출력 폴더에 그 호출의 입력 크기만큼 사본이 생긴다. 프로세스가 강제 종료되면(`SIGKILL` 등) `.swarp_input_*`가
    남을 수 있다(점으로 시작해 `DataBrowser` 검색에는 잡히지 않는다).
  - 파일 이름이 같은 두 입력을 한 번에 넘기면 사본을 만들 때 `OSError`가 난다(이름이 같으면 SWarp 재표본 파일 이름도 겹친다).
    `fill_zero_tonan=False`이면 이전처럼 입력을 그대로 SWarp에 넘긴다. SWarp XML(`XML_NAME`)에 적히는 입력 경로는 임시 사본 경로가 된다.
- `Subtract.find_transients`(`ImageMethod.DIA`, `recipes.dia`): reference 임시 사본(`<uuid>.fits`와 그 `.info`)을 `reference_img.savedir`
  (헤더 필드가 다 있으면 헤더로 만든 `REFDATA_DIR` 레이아웃 폴더이고, `--reference`로 준 파일의 실제 폴더와 다를 수 있다) 대신 target 산출물 폴더(재투영된 reference `coadd_<uuid>.fits`가 원래 쓰이던 곳)에 쓰고, 지금처럼 재투영 뒤 지운다.
  DIA 산출물의 이름·위치·수치는 그대로다. 중간에 실패해 남는 임시 사본도 이제 target 폴더에 남는다.
- `LightCurve.plot`: 등급 키일 때(`flux_key`에 `MAG`가 들어갈 때) `fluxerr_key` 값이 99 이상(오차 센티널)인 행을 그리지 않는다(점·오차 막대·선·비검출 표시 모두).
  플럭스 키로 그릴 때는 이전과 같다.
  이전에는 그 행이 검출로 그려져(오차 99) y 범위·오차 막대가 크게 벌어질 수 있었다. 반환하는 표와 `LightCurve.data`는 그대로(그 행 포함)이고,
  오차가 마스크된 행은 이전처럼 비검출로 그린다. `recipes.visualize` 갤러리와 같은 규칙이며 `recipes.target_photometry --plot` 그림에도 적용된다.
  `PhotometricSpectrum.plot`은 바꾸지 않았다(같은 문제가 남아 있다).
### 내부
- `tests/make_isolated_home.py`: 격리 HOME의 `config/specific/<telkey>/*.config`에 있는 실제 `~/ezphot/data`·`~/ezphot/log` 경로
  (`SEX`/`SCAMP`/`SWARP`/`PSFEX`의 `*_LOGDIR`·`*_HISTORYDIR`)도 `$T/log` 아래로 바꾸고, 남은 실제 경로 검사(`RuntimeError`)를 specific까지 넓혔다
  (메시지의 파일 이름은 config 기준 상대 경로). 실제 config 파일을 가리키는 `*_CONFIG`·`*_DIR`은 그대로다. 라이브러리 동작은 바뀌지 않는다.
- `recipes/defaults.py` 머리말 docstring: 레시피 기본값은 ezphot을 단독으로 쓸 때의 값이고 다른 파이프라인 설정과 맞춰 두지 않는다는 뜻으로 고쳤다. 값은 그대로다.

## 0.7.10 (2026-11-07)
### 추가
- 같은 노출의 master dark가 없을 때 더 긴 노출의 dark를 쓰는 선택 옵션(기본 꺼짐, 끄면 선택 결과는 0.7.9와 같다).
  - `Preprocess.get_masterframe(..., dark_exptime_fallback=False)`: 새 인자(맨 끝). `True`이고 `imagetyp='DARK'`, `exptime`이 주어지면
    master group을 노출과 무관하게 고른다(관측소·telkey·telname이 맞는 DARK 행 가운데 시간상 가장 가까운 `group_id >= 0` 행의 group).
    그 group 안에서 프레임 노출 이상(`np.isclose` 포함)인 dark를 후보로, 시간상 가장 가까운 밤을 고른다
    (밤 = 시간차가 가장 작은 후보의 `obsdate`에서 0.5일 안에 있는 후보들). 그 밤 안에서 같은 노출 행이 있으면 그것을,
    없으면 프레임보다 긴 노출 가운데 가장 짧은 노출의 행을 쓴다. 그래서 group이 길어도 먼 밤의 같은 노출보다 가까운 밤의 긴 노출이 먼저다.
    **프레임보다 짧은 dark는 쓰지 않는다.** 그 group에 후보가 없으면 `None`이다(다른 group으로 가지 않는다).
    `group_id >= 0` 행이 없으면(시간 분기) `max_days` 안의 행에 같은 규칙을 적용한다. 반환 표는 고른 밤·노출의 행만 `time_difference` 순으로 담는다.
  - 다른 노출을 고르면 `verbose`일 때 `DARK <x> s used for <y> s frame (scaled)` 한 줄을 출력한다. 환산은 기존 `correct_dark`/`correct_bdf`의
    `ccdproc.subtract_dark(scale=True, exposure_time='EXPTIME')`가 한다. 헤더 키는 새로 없고, 쓰인 dark는 기존 `DARKPATH`로 확인한다.
  - 켜면 같은 노출 dark가 다른 group이나 같은 group의 먼 밤에 있어도 가까운 밤의 긴 노출 dark가 선택된다. 켰을 때 고른 group이
    꺼졌을 때와 같고 그 group에서 가장 가까운 후보 밤에 같은 노출이 있으면 선택되는 dark(`[0]`)는 대개 같다(반환 표는 그 밤의 행만 담는다;
    두 밤의 거의 중간인 프레임은 예외일 수 있다). group 경계 근처 프레임에서는 같은 노출 dark도
    꺼졌을 때와 다른(더 먼 날짜일 수 있는) group에서 고를 수 있다.
  - `ImageMethod.get_masterframe(imagetyp, max_days=10, dark_exptime_fallback=False)`, `ImageMethod.correct_bdf(..., dark_exptime_fallback=False)`:
    같은 인자(맨 끝). `correct_bdf`는 `dark_image`가 없을 때의 DARK 조회에만 넘긴다. `ImageMethod.correct_dark`에는 없다.
  - `recipes.process_images`: CLI `--dark-fallback`(`--preprocess`와 함께), Python `process_images(..., dark_exptime_fallback=False)`,
    `process_image(..., dark_exptime_fallback=False)`(둘 다 맨 끝). summary CSV 열은 그대로다.
  - 켜도 맞는 dark가 없으면 dark가 없을 때와 같이 실패한다(`correct_bdf`/레시피 `step=correct_bdf` 오류).

## 0.7.9 (2026-11-06)
### 추가
- `recipes.visualize` 광도곡선·SED 탭: **축 드래그로 x·y 범위 조절**. 축 바깥의 눈금 숫자 줄(x: 축 아래, y: 축 왼쪽)을 끈다.
  - 양 끝 1/3을 끌면 그쪽 한계만 움직이고(반대쪽 한계 고정), 가운데 1/3을 끌면 폭을 유지한 채 범위째 이동한다. 잡은 자리의 값이 포인터를 따라간다. y는 지금처럼 작은 등급이 위다. 한계가 뒤집히거나 폭이 0이 되지 않게 막고, 범위는 그 축의 자동 범위 ± 자동 범위 폭의 100배 안에 머문다(드래그를 반복해도 넓어지지 않음). 축 방향으로 5 px 넘게 움직이기 전에는 바뀌지 않는다(클릭·떨림 무시).
  - 대상: 등급·Δmag 광도곡선과 SED는 x·y, 필터별 패널은 x만(y는 패널마다 자동).
  - 광도곡선 x는 기간 컨트롤과 같은 상태라서 x를 끌면 기간 칸과 SED bin 범위도 바뀐다. SED x(파장)는 기간과 무관하다.
  - 축을 더블클릭하면 그 축만 자동 범위로 돌아간다(광도곡선 x는 "전체 기간"과 같음). 컨트롤 "축 자동"은 광도곡선 y(SED에서는 x·y)를 자동으로 돌린다.
  - 수동 y 범위는 필터 칩·bin 폭·필터/bin 오프셋·GP·필터별 패널 왕복에서 유지되고, 등급↔Δmag 전환·좌표 적용(데이터가 바뀜)에서 자동으로 돌아간다.
  - 수동 y 범위일 때는 범위 밖 점·오차 막대가 축에서 잘린다. 자동 범위 계산(점 ± 오차)과 수동 범위가 없을 때의 그림은 0.7.8과 같다.
  - 끄는 동안 그림을 바로 다시 그린다(화면 프레임당 한 번).
### 변경
- `recipes.visualize` 광도곡선 탭: x 눈금 숫자 줄 위에서 가로로 끌면 이제 기간 선택(드래그 확대)이 아니라 x축 조작이다. 그림 영역 안에서 끄는 드래그 확대는 그대로다. y 눈금 숫자 줄에서 시작한 드래그도 드래그 확대가 아니다(큰 그림: y축 조작, 필터별 패널: 동작 없음).
- 이미 만든 갤러리에 넣으려면 `visualize --pages-only`로 페이지를 다시 만든다. 페이지 데이터·산출 파일·측광 값은 바뀌지 않는다.

## 0.7.8 (2026-11-05)
### 추가
- `recipes.visualize` 광도곡선·SED 탭에 **GP 곡선**을 그린다(gpimg 2차원 GP, 평균 ±1σ).
  - `--lightcurve`(target_photometry 표)로 만든 탭에만 있다. 좌표 입력(검출 카탈로그 매칭)으로 그린 곡선에는 없다.
  - 적합 입력은 페이지 점과 같은 행 가운데 중대역 `m###` 필터(이름이 `m` + 숫자 셋, `w` 접미 없음)만이다. `m###w`·광대역은 넣지 않는다.
    파장은 페이지 점의 x(투과곡선 응답 가중 평균 파장) × 10 Å, flux = 10^(−0.4(`MAGSKY_APER_3` − 23.9)) μJy, 오차는 `MAGERR_APER_3`(바닥 1e-4 mag)를 flux로 옮긴 값이다. 계통 오차·jitter는 더하지 않는다.
  - 모형: gpimg 기본(시간 Matérn-3/2 × ln λ 제곱지수), jitter 없음, 시작점 10개 중 최대우도, smoother `rts`. 갤러리마다 visualize 실행 때 한 번 적합한다(`--n-proc`와 무관).
  - 격자: 시간은 적합한 점의 첫–마지막 시각(외삽 없음)을 0.25일 간격으로 나누고, 격자 값이 120,000개(약 1 MB)를 넘으면 간격을 늘린다. 파장은 밴드 파장과 그 사이 균등 60점(밴드 범위 안). 결과는 등급(평균, 평균 ± 1σ의 등급; 평균 − 1σ ≤ 0이면 null)으로 `index.html`에 인라인으로 들어가므로 페이지가 최대 약 1 MB 커진다.
  - 광도곡선 탭: 필터마다 평균 선과 ±1σ 띠를 그 필터 점의 첫–마지막 시각 안에만 그린다(등급·Δmag·필터별 패널, 필터 오프셋, 필터 칩, 확대·기간과 일관). 그래서 그 기간이 시간 격자 간격보다 짧은 필터(예: 한 밤만 관측된 필터, 또는 긴 데이터에서 격자 간격이 넓어졌을 때)는 선이 없을 수 있다. SED 탭: 각 bin 점들의 평균 시각에서 시간 선형 보간한 파장별 곡선(한 bin이면 ±1σ 띠, 여러 bin이면 점선만, bin 오프셋 적용). 컨트롤 "GP"(기본 켬)로 숨길 수 있다. 탭 설명에 적합 정보(l_t, l_λ, 수렴 여부, 넣지 않은 필터)가 붙는다.
  - **gpimg는 선택 의존성**이다(`requirements.txt`에 없음). import되지 않으면 GP 없이 페이지를 만들고 탭에 "GP 곡선 없음: gpimg is not importable …"을 적는다. gpimg가 데이터를 거부하거나 적합에 실패해도 점은 그대로 보이고 오류 문장(경로는 `<path>`)이 탭에 붙는다. 예: 밴드 수가 gpimg 빌드 한도 `GPIMG_MAXB`를 넘을 때. PyPI 빌드는 6밴드라 7DT 중대역 전체(20밴드)를 쓰려면 한도를 올린 빌드가 필요하다. ezphot은 밴드 수를 검사하지 않는다. 각 경우 로그에 `[visualize] GP curves ...` 한 줄이 나온다.
  - `--pages-only`·`--skip-unchanged`에서도 적합한다. 끄려면 `--no-gp` / `visualize(..., gp=False)`(새 인자, 맨 끝, 기본 True).
  - 페이지 데이터 `lcdata`에 `gp` 키가 생긴다(`t`, `wl`, `f`, `row`, `m`, `lo`, `hi`, `n`, `step`, `fit`, `version`, 또는 `error`, `skip`). 내부 형식이라 바뀔 수 있다. 기존 키는 그대로다.
- `recipes.visualize` 광도곡선·SED 탭 공통 **기간** 컨트롤: 시작·끝 날짜(UTC, 끝 날 포함), "전체 기간", 데이터 기간보다 짧을 때 "마지막 7/30/100일".
  - 날짜는 달력에서 고르면 바로, 키보드로 치면 Enter를 누르거나 칸을 벗어날 때 적용된다.
  - 광도곡선에서는 x 범위이고 드래그 확대와 같은 상태다(드래그하면 날짜 칸도 바뀐다).
  - SED에서는 bin 점들의 평균 시각이 기간 안인 bin만 목록·겹쳐 보기에 들어간다.
  - 표시만 바꾼다. GP 적합(전체 점)과 합친 점 계산은 그대로다. 탭을 오가도 유지되고, 좌표를 새로 적용하면(데이터가 바뀌면) 전체 기간으로 돌아간다.
- `recipes.visualize` SED 탭: "bin 오프셋"의 간격을 옆 숫자 칸(등급, 기본 `PhotometricSpectrum.OFFSET`)으로 바꿀 수 있다. 0·음수도 되고, 숫자가 아니면 이전 값을 유지한다.
### 변경
- `recipes.visualize` 광도곡선 탭의 "범위 리셋" 버튼이 기간 컨트롤의 "전체 기간"으로 바뀐다. 광도곡선에서 확대한 범위가 이제 SED 탭의 bin에도 적용된다(이전: SED는 확대와 무관하게 모든 bin).
- `recipes.visualize` SED 탭의 bin 선택 목록이 "파장이 있는 필터의 점이 있고 기간 안인 bin"으로 좁아진다(이전: 모든 bin, 파장 없는 필터의 점만 있는 bin 포함).
- 이미 만든 갤러리에 GP·기간 컨트롤을 넣으려면 `visualize --pages-only --lightcurve <표>`로 페이지를 다시 만든다(GP는 gpimg가 import되는 환경에서만).

## 0.7.7 (2026-11-05)
### 변경
- `recipes.process_images`·`recipes.stack`·`recipes.dia`: `out`을 주지 않으면 기본 경로 `~/ezphot/log/recipes/<objname>/<recipe>_<YYYYmmdd_HHMMSS>.csv`를 처리 **전**에 정한다. 그래서 파일명의 시각이 끝난 시각에서 **시작 시각**으로 바뀌고, 디렉터리도 처리 전에 만든다(빈 입력으로 `ValueError`가 나는 경우는 전처럼 만들지 않는다). 파일명 패턴은 그대로다. 파일 이름(시각)으로 최신 실행을 고르는 코드는 확인할 것: 오래 걸린 실행은, 그보다 늦게 시작해 먼저 끝난 실행보다 이름의 시각이 앞선다.
- `recipes.visualize` 갤러리의 필터 색을 `LightCurve.FILTER_COLOR`와 같게 한다(중대역은 중대역 전체 파장 범위로 정규화한 plasma, 광대역 u/g/r/i/z/U/B/V/R/I 등은 고정색). 대상은 페이지 광도곡선·SED 탭의 점·선·필터 칩과 정적 그림 `lightcurve.png`다. 정규화 범위가 고정이라 갤러리가 달라도 같은 필터는 같은 색이다(이전: 페이지는 그 페이지 필터들의 파장 범위로 정규화한 보라→빨강 hsl, `lightcurve.png`는 matplotlib 순환색 `C0`, `C1`, …).
  - `FILTER_COLOR`에 있는 필터는 다크·라이트 모드에서 같은 색이다. 없는 필터는 이전 색이다(페이지: 모드별 hsl 파장 보간, 파장이 없으면 회색; `lightcurve.png`: 순환색).
  - 마커 모양·필터별 등급 오프셋은 그대로다.
  - 밝은 바탕에서는 노란 계열(m800–m875, `i`), 어두운 바탕에서는 짙은 남색 계열(m375w–m400)이 잘 보이지 않을 수 있다.
  - 이미 만든 갤러리는 `visualize --pages-only`로 페이지만 다시 만들면 반영된다(`sources.json`도 이때 다시 쓰이고, `lightcurve.png`는 처음처럼 `--lightcurve`를 주면 다시 그려진다).
- `recipes.visualize` 광도곡선·SED 탭의 시야 이미지(`fov.jpg`, 클릭 입력) **대표 프레임 기준**을 바꾼다(이전: SNR > 3 검출이 가장 많은 프레임).
  - 후보는 성공 행 가운데 검출(`.cat`)이 있는 프레임이다. 점수는 0.5 × p(−`SEEING`) + 0.5 × p(`UL5SKY_APER_3`)이고, p는 후보 안 백분위 순위(0–1, 같은 값은 평균 순위)다. 값이 없으면 그 항은 0점이다. 점수가 가장 높은 프레임을 고르고, 동률이면 갤러리 순서 앞이다. 필터는 구분하지 않는다. 후보가 없으면(성공 행에 검출이 없으면) 시야 이미지가 없다.
  - `--ra`/`--dec`가 있으면 점수 순으로 헤더 WCS를 읽어, 대상이 칩 안(0-based 픽셀 −0.5 ≤ x < NAXIS1 − 0.5, y도 같음)인 첫 프레임을 고른다. 그런 프레임이 없으면 점수 1위다. 천구 WCS가 없거나 헤더를 읽지 못한 프레임은 건너뛰고 로그에 수와 첫 오류를 적는다. 하나도 판정하지 못하면 점수 1위이고 페이지에 "칩 판정 불가"로 표시한다. WCS 역변환이 수렴하지 않으면(`NoConvergence`) 칩 밖으로 본다. 이 때문에 후보 수만큼 헤더를 더 읽을 수 있다.
  - 갤러리마다 시야 이미지와 클릭 입력에 쓰이는 프레임이 바뀔 수 있다. 매칭·측광 값은 바뀌지 않는다.
  - 로그의 `click input` 줄에 고른 이유(SEEING, 깊이, 점수와 순위, 대상이 칩 안인지)가 붙는다. 페이지 데이터 `fov`에 `sel`(`target`·`outside`·`unknown`·`score`) 키가 생기고, 시야 이미지 툴팁 문구가 새 기준을 설명한다.
  - 이미 만든 갤러리는 `visualize --pages-only`로 다시 만들면 반영된다.
### 수정
- `MaskGenerator.mask_circle`(와 이를 부르는 `ScienceImage.calculate_circularmask`, 레시피 `process_images`의 circularmask 단계): 원형 개구의 bounding box가 이미지와 겹치지 않으면(대상이 칩 밖) `ValueError: Target outside image: (<x>, <y>) [<unit>] -> pixel (<px>, <py>) (0-based), radius <r> px, image shape <shape> (ny, nx)`를 낸다(`unit='coord'`면 `(<x>, <y>)`는 RA/Dec(deg)). 이전에는 같은 경우 `AttributeError: 'NoneType' object has no attribute 'copy'`(data가 있는 외부 mask를 넘기면 `... 'astype'`)였다.
  - 가장자리·모서리 바로 밖이라 bbox만 걸치면 이전처럼 빈 원형 마스크로 진행한다(예외 없음).
  - 실패 여부는 그대로다(실패 → 실패). 예외 타입만 `AttributeError` → `ValueError`로 바뀌므로, 이 경우를 `AttributeError`로 잡던 호출처는 `ValueError`로 바꿔야 한다.
  - `process_images` summary CSV의 `error` 열 문구가 위 메시지로 바뀐다(`step`은 그대로 `circularmask`). 일부만 겹치거나 칩 안이면 마스크와 수치는 이전과 같다.
### 추가
- `recipes.process_images`·`recipes.stack`·`recipes.dia`: 실행 중에 summary CSV 옆에 진행 파일 `<out>.progress.json`이 생긴다(예: `process_images_<tag>.csv.progress.json`). Python API에서 `out`을 준 경우에도 그 옆에 생긴다.
  - 부모 프로세스가 시작할 때와 작업 하나(stack은 스택 하나)가 끝날 때마다 원자적으로 다시 쓴다(tmp + `os.replace`). 실행이 끝나도 지우지 않는다.
  - 내용은 `recipes.status`가 읽는 내부 형식이라 바뀔 수 있다. 스키마는 약속하지 않는다.
  - summary CSV의 열·값·행 순서(입력 순서), 반환, 종료 코드는 그대로다. 워커 프로세스가 죽을 때의 예외(`BrokenProcessPool`) 전파도 그대로다.
  - 진행 파일을 쓰지 못하면(`OSError`) 실행마다 처음 한 번 `[progress] cannot write ...`를 출력하고 처리를 계속한다.
- `recipes.status`(처리 목록 페이지): 실행 중인 실행의 진행률을 보여 준다.
  - `~/ezphot/log/recipes/<objname>/`의 진행 파일로도 실행을 찾는다. 그래서 `--out`만 이 디렉터리로 주고 로그는 다른 곳에 쓰는 실행, `--out` 없이 기본 경로로 쓰는 실행(Python API 포함)도 실행 중에 보인다. `--out`이 이 디렉터리 밖인 실행은 지금처럼 목록에 없다.
  - 실행 중 행: 상태 칸 "실행 중 N/M"(실패가 있으면 "(실패 k)"), 이미지 칸 M, 소요 칸 "1h05m 경과 · 약 20m 남음" 형식이다(끝난 실행은 지금처럼 초). 남은 시간은 경과/N × (M−N)이고, N = 0이면 경과만 나온다.
  - "실행 중"으로 보는 조건은 셋이다. (1) 진행 파일의 pid가 이 호스트에서 살아 있다. 실행 시작 전에 시작된 프로세스여야 하므로 재사용된 pid는 걸러진다. (2) 레시피의 처리 루프가 아직 끝나지 않았다(반환도 예외도 아님). (3) csv가 실행 시작 뒤에 쓰이지 않았다. 지금처럼 `--out`이 맞는 레시피 프로세스가 있어도 실행 중이다.
  - csv가 없고 프로세스가 없거나 처리 루프가 예외로 끝났으면 "중단 N/M"이다. 예를 들어 Python API에서 예외가 난 뒤에도 프로세스가 살아 있는 경우가 그렇다. 정상으로 끝난 Python API 실행은 루프가 끝난 뒤 csv를 쓰기까지의 짧은 동안 "중단"으로 보일 수 있다.
  - stack·dia 실행은 실행 중·중단일 때만 행으로 나온다(실행 열 `stack_<tag>`/`dia_<tag>`). 끝나서 csv가 생기면 지금처럼 목록에서 빠진다. 끝난 실행의 표시·qa·"오래됨"은 그대로 process_images csv 기준이다.
  - `collect()`가 돌려주는 실행 dict에 `recipe`, `done`, `failed`, `elapsed`, `remaining` 키가 생긴다(진행 파일이 없으면 `None`). 기존 키는 그대로다.
  - 이미 떠 있는 실행(0.7.6 이하)은 진행 파일을 쓰지 않으므로 전처럼 보인다. 이미 떠 있는 `status --watch`는 다시 시작해야 반영된다.
- `recipes.visualize`: 페이지 데이터(`lcdata`)와 `sources.json`(카탈로그 매칭 모드)의 `filters` 항목에 `c` 키(hex 색, `FILTER_COLOR`에 없으면 `null`)와 `o` 키(`LightCurve.FILTER_OFFSET` 등급, 없으면 0)가 생긴다. 페이지 데이터 맨 위에 `sed_offset`(`PhotometricSpectrum.OFFSET`)이 생긴다. 기존 키는 그대로다.
- `recipes.visualize` 페이지 광도곡선 탭: **"필터 오프셋"** 켜기/끄기(기본 끔 = 이전과 같음). "등급" 모드에서만 보이고 적용된다.
  - 켜면 필터마다 `LightCurve.FILTER_OFFSET`(없는 필터 0)을 그림의 등급에 더한다. 필터 칩 이름에 `LightCurve.plot` 범례처럼 `m625+2.0`으로 붙고, 툴팁에 오프셋 줄이 생기며, y축 이름에 `(+ offset)`이 붙는다. 툴팁의 등급 값은 오프셋 전 값이다.
  - Δmag 모드(필터별 평균 기준 차이)와 필터별 패널에는 적용하지 않는다(필터마다 따로 보거나 평균을 빼므로 의미가 없다).
- `recipes.visualize` 페이지 SED 탭: 여러 시간 bin을 겹쳐 볼 때 **"bin 오프셋"** 켜기/끄기(기본 끔). 켜면 k번째 bin(0부터)에 k × `PhotometricSpectrum.OFFSET`(2 등급)을 더해 세로로 벌린다. 그림 아래에 bin별 오프셋 범례(`PhotometricSpectrum.plot`처럼 `(+2.0)`), 툴팁에 오프셋 줄, y축 이름에 `(+ offset)`이 붙는다. bin 하나만 볼 때는 적용되지 않는다.
  - 정적 `lightcurve.png`는 바뀌지 않는다. 이미 만든 갤러리는 `visualize --pages-only`로 다시 만들면 생긴다.

## 0.7.6 (2026-11-04)
### 수치 변경
- `recipes.qa`: 기본 `qa_pass`에 색지수 검사 `|K_COLOR_APER_3_g-r| <= 0.2`가 들어간다(`color_max=0.2` 기본 켜짐). `K_COLOR_APER_3_g-r`는 photcal이 헤더에 남기는 10″ ZP 잔차의 g−r 기울기(mag/mag)이며, 모든 필터에 같은 기준을 쓴다.
  - 값이 없는 이미지(색 적합 실패, (g, r) 없음, 헤더 키·파일 없음)는 색지수 검사를 **통과**로 본다. 다른 검사(NaN이면 실패)와 반대다. 실패 행(`success` False)은 읽지 않는다.
  - 같은 summary에서 qa 결과와 CLI 종료 코드가 바뀔 수 있다. 실측(0.7.5 qa와 행별 비교):
    - T22956 pilot·p625, FEIGE110, baseline T05785·T10796에서는 통과→실패가 0이다. 색지수에 걸린 행(FEIGE110 m375w 12장, K +0.224…+0.351; T05785 m450 3장, K 1.46…1.49)이 0.7.5에서도 이미 실패였기 때문이다.
    - 실제 log 트리 전체(summary 50개, 4,413행)의 qa 통과 수는 4036 → 4019다. 바뀐 실행은 T22956 `c1_20261104_132033` 507 → 506, WD0227+050 `20261031_n1002` 104 → 102, WD0308-565 `20261031_n1002` 439 → 431, WD2039-682 `20261031_n1028` 255 → 254, WD2149+021 `20261031_n1028` 222 → 217뿐이다.
  - 처리 목록 페이지(`recipes.status`)의 "qa 통과" 열도 이 기준을 따른다. 페이지 위 기준 문구에 색지수 기준을 넣었다. 이미 떠 있는 `status --watch`는 다시 시작해야 반영된다.
    - 0.7.5 이하 summary는 헤더를 읽으므로, 프로세스의 첫 갱신이 한 번 느리다. 실제 log 트리에서 첫 갱신은 114 s였고, 그 뒤 갱신은 캐시로 0.06–0.3 s였다(0.7.5: 0.96 s).
  - 이전 판정으로 되돌리려면 `qa(..., color_max=None)` / `--color-max none`. 이때 `K_COLOR_APER_3_g-r`·`chk_color` 열을 만들지 않으며 0.7.5와 같다.
### 추가
- `recipes.process_images`(와 같은 체인을 쓰는 `recipes.stack`) summary CSV에 `K_COLOR_APER_3_g-r` 열이 생긴다(`defaults.SUMMARY_HEADER_KEYS` 끝). 헤더에 키가 없으면 빈 값이다. 기존 열·순서는 그대로다.
- `recipes.qa`: `qa(..., color_max=0.2)`(인자는 `depth_min` 뒤), CLI `--color-max <float|none>`(기본 0.2).
  - 반환 frame에 `K_COLOR_APER_3_g-r`(값)과 `chk_color` 열이 생긴다. 값은 summary 열에서 읽고, 열이 없거나 값이 비면(0.7.5 이하 summary) 성공 행에 한해 이미지 헤더(`fits.getheader(path)`)에서 읽는다.
  - CLI 표에 `K_COLOR_APER_3_g-r` 열(`qa_pass` 앞), 요약 줄에 기준과 값 없는 성공 행 수가 둘로 나뉘어 붙고(`; M without key, U unreadable, counted as pass` — M = summary 칸이 비고 헤더에 키가 없는 행, U = FITS를 읽지 못한 행(`OSError`: 파일 없음·FITS 아님); 둘 다 통과로 센다), FAIL 줄에 `color`가 나온다.
- `recipes.status`: done 실행 csv의 qa 결과를 프로세스 안에 (csv 경로, mtime_ns, 크기) 기준으로 남겨, csv가 바뀌지 않으면 `--watch` 갱신마다 qa(헤더 읽기)를 다시 하지 않는다.

## 0.7.5 (2026-11-04)
### 추가
- `recipes.visualize` 페이지, 광도곡선·SED 탭: 좌표 칸의 "표로 돌아가기" 버튼을 없애고 **대상 이름 버튼**(갤러리 이름, 예: `GD71`)을 둔다.
  - 버튼은 시야 이미지 카드의 "전체 시야" 옆에 있다(좌표 칸 줄 수를 늘리지 않기 위해). 시야 이미지가 없거나 만들지 못한 갤러리에서는 좌표 칸의 적용 버튼 옆에 있다.
  - 대상 좌표(`--ra`/`--dec`)가 있으면 늘 보인다. 누르면 RA·Dec·반경 칸을 대상 좌표와 기본 반경(5″)으로 되돌리고, 표(`--lightcurve`)가 있으면 target_photometry 표 보기로, 없으면 그 좌표·반경으로 카탈로그 매칭을 한다.
  - 표는 있고 `--ra`/`--dec`가 없으면 표 보기로 돌아가는 버튼으로만 보인다(칸은 그대로).
  - 대상을 보고 있으면(표 보기, 또는 대상 좌표 0.1″ 이내·기본 반경의 카탈로그 매칭) 선택됨으로 강조한다. 긴 이름은 버튼 안에서 "…"로 줄이고 전체 이름은 툴팁에 있다.
  - 이미 만든 갤러리는 `visualize --pages-only`로 페이지만 다시 만들면 생긴다(그림은 그대로).
  - `--pages-only`로 다시 만든 갤러리는 처리 목록 페이지의 0.7.3 "오래됨" 배지가 사라진다(시각으로 비교하기 때문).

## 0.7.4 (2026-11-04)
### 추가
- `recipes.process_images` CLI `--no-figures`: process_images가 직접 저장하는 진단 그림 PNG(sourcemask, bkgmap, bkgrms, photometry의 `.subbkg.png`, photcal의 `.refcat`·`.zp_mag`·`.zp_color`·`.zp_2d.png`)를 저장하지 않는다. 옵션을 주지 않으면 0.7.3과 같다(그림 저장).
  - 비용: 격리 HOME에서 FEIGE110 24장(30 s 노출, `--preprocess`, n_proc 8)을 그림 켜고 끄며 잰 결과, 그림 저장은 장당 약 45 s로 process_images 시간의 17%였다. ZP_APER_3·ZPERR_APER_3·SEEING·UL5SKY_APER_3·n_ref·SKYVAL은 켠 실행과 끈 실행에서 같았다.
  - Python에서 부르는 쪽은 이미 `process_images(..., config={s: {'save_fig': False} for s in ('sourcemask', 'bkgmap', 'bkgrms', 'photometry', 'photcal')})`로 같은 일을 할 수 있다(시그니처 불변).
  - 그림을 끄면 visualize 갤러리의 배경·ZP Calibration 칸이 빈다(다른 단계는 visualize가 직접 그린다). 단, 이전 실행의 PNG가 남아 있으면(이 옵션은 지우지 않는다) 그 옛 그림이 보이며, 지금의 ZP와 맞지 않을 수 있다.
  - 예전에 이 그림이 있던 갤러리 폴더에 `visualize --pages-only`를 돌리면 그 단계에 'figure missing'이 뜰 수 있다.

## 0.7.3 (2026-11-03)
### 추가
- `recipes.status`(처리 목록 페이지 `<root>/ezphot/index.html`): 대상의 갤러리 칸 "갤러리" 링크 옆에 **"오래됨"** 배지를 붙인다. 붙는 조건은, 그 대상의 완료(done) 실행 summary csv 가운데 가장 새 것의 mtime이 갤러리 `index.html`의 mtime보다 뒤일 때다(같으면 붙이지 않는다).
  - 툴팁에 두 시각(초까지)과 "갤러리 페이지 생성 시각 기준이며 target_photometry 변경은 반영 안 함"을 적는다.
  - 실행 중·csv 없는 중단·읽기 실패 실행은 판정에 쓰지 않는다. 갤러리가 없으면 이전처럼 "없음"이다.
  - 시각 기준이라 `visualize --pages-only`·`--skip-unchanged`로 페이지만 다시 만들어도 배지가 사라진다(그림이 새 summary로 그려졌는지는 보지 않는다).
- `recipes.status.collect()`가 돌려주는 대상 dict에 키 `gallery_mtime`, `summary_mtime`, `stale`이, 실행 dict에 `csv_mtime`(csv가 없거나 읽기 실패면 None)이 추가된다. 기존 키와 값(`gallery`는 index.html이 있는지)은 그대로다.
- 이미 떠 있는 `status --watch` 프로세스는 옛 코드를 import한 상태라 다시 시작해야 반영된다.

## 0.7.2 (2026-11-03)
### 추가
- `recipes.visualize`: `visualize(..., skip_unchanged=False, pages_only=False)`, CLI `--skip-unchanged`, `--pages-only`. 둘은 함께 쓸 수 없다(`ValueError`, CLI는 argparse 오류).
  기본(둘 다 끔)은 0.7.1처럼 모든 단계 그림을 다시 그린다.
  - `--skip-unchanged`: 프레임 단계마다 입력 파일이 이전 실행 때와 같으면 그 단계 그림을 다시 그리지 않고 있는 그림으로 페이지를 만든다.
    "같다"는 파일 mtime(ns)과 크기가 같다는 뜻이다. 입력 파일: 전처리 = 이미지·`.info`·master 3장, WCS = 이미지·`.cat`·`.info`, 마스크 = 이미지·`.srcmask`·`.invalidmask`, 배경·ZP = 링크하는 ezphot 그림.
    다시 그리는 경우는 넷이다: 입력 파일의 mtime이나 크기가 기록과 다를 때, 그 단계를 그린 기록이 없을 때, 그림 파일이 하나라도 없을 때, 기록의 그림 버전(`FIG_REV`)이 다를 때.
    다시 그리지 않은 단계는 썸네일도 있으면 그대로 쓴다(읽을 수 없으면 다시 만든다).
    **sky catalog 아카이브가 바뀐 것은 보지 않는다**(WCS 그림의 기준성이 바뀌었으면 기본 실행으로 다시 그린다).
  - `--pages-only`: 단계 그림을 하나도 그리지 않고 프레임 폴더에 있는 그림으로 페이지를 다시 만든다. 레이아웃·캡션만 바뀐 버전으로 갤러리를 다시 만들 때 쓴다.
    그 단계의 그림이 하나라도 없으면 그 단계를 비우고 상태에 `| <단계> figure missing`을 붙인다(개요 표에서 실패로 보인다; 그림이 원래 없던 단계는 missing으로 보일 수 있다).
    썸네일은 그 단계의 기록 입력이 지금과 같을 때만 그대로 쓰고, 아니면 있는 그림에서 다시 만든다.
    기록이 없는 0.7.1 이전 폴더는 입력 파일로 예상 그림을 정한다.
  - 두 옵션 모두 프레임 목록·`sources.json`·`fov.jpg`·실행 추이 그림·`manifest.json`·`index.html`은 매번 다시 만든다.
- 산출물: 프레임 폴더(`<stem>/`)마다 `.viz_done.json`이 생긴다. 내용은 `rev`(=`FIG_REV`)와 `stages: {단계: {t0, inputs, names}}`이다.
  `inputs`는 역할별 [mtime_ns, 크기]이고 경로는 넣지 않는다. 갤러리와 함께 서빙되기 때문이다.
  - 그리는 실행(기본, `--skip-unchanged`)이 매번 원자적으로(`os.replace`) 다시 쓴다. 그래야 다음 `--skip-unchanged`가 쓸 수 있다. `--pages-only`는 쓰지 않는다.
  - `manifest.json`에는 넣지 않았다(manifest 형식 불변).
### 내부 (속도; 그림·페이지·수치 불변)
- 같은 입력에서 그림 PNG·썸네일·`manifest.json`·`sources.json`·`index.html`(token 제외)은 0.7.1과 바이트가 같아야 한다(검증 항목).
- sky catalog 타일(`SkyCatalogUtility._load_catalog_worker`)을 프로세스 안에서 objname·종류·버전·시야 크기로 캐시한다. `SkyCatalogUtility`는 프로세스당 하나다. `visualize()`를 부를 때마다 비운다.
  병렬 실행의 워커는 fork로 첫 프레임이 채운 캐시를 이어받는다. `ezphot.skycatalog`는 바꾸지 않았다. `visualize()`가 끝나면(오류로 끝나도) 부른 프로세스의 캐시도 비운다.
- 8×8 블록 축소: NaN이 없으면 `np.mean`을 쓰고, 결과에 NaN이 있으면 이전처럼 `np.nanmean`으로 다시 계산한다. 전처리 'corrected' 칸과 마스크 첫 칸이 같은 축소(행 띠 단위)를 한 번만 계산한다.
- master bias·dark·flat(및 flat / mean(flat), dark EXPTIME)을 (경로, mtime_ns) 키로 워커 안에 한 세트만 남겨 다음 프레임에 쓴다. 실행 중에는 워커당 상주 메모리가 master 3장(약 0.7 GB, 7DT 9576×6388 float32)만큼 는다.
  워커는 실행이 끝나면 사라지고, `visualize()`를 부른 프로세스의 세트는 반환할 때 비운다(라이브러리로 부르는 쪽에 남지 않는다).
- 썸네일: 축소 방식(1400 px, JPEG q85, 흰 배경)은 그대로 두고, 메모리에서 인코딩해 임시 파일에 쓴 뒤 `os.replace`로 바꿔 넣는다. 기본 실행은 0.7.1처럼 썸네일을 매번 다시 만든다(재사용은 위 두 옵션에서만).
- 파일 시스템 접근 줄이기:
  - ezphot 그림 링크가 이미 같은 대상을 가리키면 다시 만들지 않는다.
  - 마스크 파일 존재는 한 번만 확인한다.
  - `.cat`은 프레임당 한 번만 읽는다(WCS 그림과 좌표 입력이 공유).
  - `.info`는 objname을 정할 때 한 번 읽은 것을 그림에도 쓴다.
- 개발자 안내: visualize가 그리는 그림이나 썸네일의 내용을 바꾸는 변경은 `visualize.FIG_REV`를 1 올린다. 그래야 이전 기록이 `--skip-unchanged`·`--pages-only`에서 재사용되지 않는다.

## 0.7.1 (2026-11-03)
### 변경
- `recipes.visualize` 페이지 표시, ZP Calibration 탭: 0.7.0의 단계별 서술 중 ZP 부분("그림 4장이면 2×2, 한 화면 안에서 가장 크게. 높이 제한 때문에 남는 폭이 있으면 두 열의 그림이 가운데로 모인다")을 대체한다.
  - 그림 4장은 2×2 두 줄이다. 그리드 폭은 최대 1000 px이고 가운데에 놓인다. 칸은 모두 같은 4:3 상자이고, 상자는 칸 폭을 채운다.
  - 그림은 상자 한가운데에 비율을 지켜 놓인다. 행·열마다 가장자리와 캡션 위치가 맞는다. 캡션은 가운데 정렬이다.
  - 그림 비율이 서로 달라(refcat 1.33, zp_mag·zp_color 1.13, zp_2d 1.29) 4:3보다 세로가 긴 그림은 상자 높이에 맞춰져 좌우가 조금 빈다.
  - **ZP 탭은 한 화면 높이 맞춤을 하지 않는다**: 그림을 크게 보이는 대신 창이 낮으면 세로로 스크롤한다. 다른 단계 탭(전처리·WCS·마스크·배경)의 한 화면 맞춤은 그대로다.
  - 폭 760 px 이하 한 열에서는 상자 비율을 고정하지 않고 그림을 폭 가득 보인다.
  - WCS 2열(`wcs_offsets` · `wcs_cutouts`)은 0.7.0처럼 가운데 경계로 모인다.
- `recipes.visualize` 페이지 표시, 광도곡선·SED 탭의 전체 시야 이미지: 검출 원(초록)을 더 얇고 옅게 했다. 선 굵기 화면 0.75 px(이전 1 px), 불투명도 0.5(이전 0.8), 반지름 화면 4 px(이전 5 px)다.
  확대해도 선 굵기와 반지름은 화면 기준으로 같다. 고른 검출의 빨간 원(굵기 2 px, 반지름 9 px), 원 끄고 켜기, 클릭 반경(화면 20 px)은 그대로다.
- 그림 파일, `manifest.json`, `sources.json`, `fov.jpg`, `index.html`의 페이지 데이터, `visualize()` 시그니처·CLI, 측광 수치는 바뀌지 않는다. 바뀌는 것은 `index.html`의 페이지 템플릿(CSS, 그리드 클래스, 검출 원 반지름)뿐이다.
  **이미 만든 갤러리는 다시 그려야 반영된다**(`index.html`에 템플릿이 들어 있다).

## 0.7.0 (2026-11-02)
### 파괴적
- `recipes.visualize`: **`target_cutout.png`(타깃 주변 60″ 컷아웃, 10″ 개구)를 더 만들지 않는다.** `--ra`/`--dec`(`visualize(ra=, dec=)`)를 주어도 같다.
  그래서 `manifest.json`의 프레임별 `photometry` 목록에서 `"<stem>/target_cutout.png"`가 빠지고 ezphot ZP 그림(`.refcat.png`·`.zp_mag.png`·`.zp_color.png`·`.zp_2d.png` 링크)만 남는다.
  `index.html`의 그림 목록도 같다. `manifest.json`의 키는 그대로다. 목록 안의 `target_cutout.png`를 읽던 코드는 고쳐야 한다. 사용자가 빼기를 요청한 그림이라 deprecation 기간은 두지 않았다.
  같은 갤러리를 다시 그리면 이전 실행의 `target_cutout.png`·`thumb_target_cutout.jpg`가 프레임 폴더에 남지만 페이지·manifest는 가리키지 않는다.
- `--ra`/`--dec`의 의미가 줄어든다: 이제 광도곡선·SED 탭의 기본 좌표(`--lightcurve`가 없으면 탭을 처음 열 때 그 좌표로 카탈로그 매칭)로만 쓰인다(`.cat`이 없어 좌표 입력이 없는 갤러리에서는 효과 없음). 인자 이름·시그니처·CLI는 그대로다.
### 변경
- `recipes.visualize`: `wcs_offsets.png`(파일 이름 그대로)의 배치를 0.6.0에서 다시 바꿨다. 0.6.0은 세 패널을 세로로 쌓은 5×10 in 그림이었다.
  이제는 9.5×5 in 2열 그림이다: 왼쪽 열에 시야 분포(오프셋 화살표), 오른쪽 열 위에 중심 거리 의존성, 아래에 히스토그램. 가로/세로가 1.8 미만이라 페이지 데이터의 `wide`는 `false` 그대로다.
  시야 분포의 색 막대는 그림 아래 가로로 옮겼고, 0.5″ 화살표 기준은 눈금 글자와 색 막대 사이에 둔다. 제목은 `detected -> reference (x2000), N=…`로 줄였다. 계산은 같다.
- 탭 이름: `photometry` 단계의 탭 라벨(`STAGE_LABELS`, 페이지 데이터 `labels.photometry`)이 "측광"에서 "ZP Calibration"이 된다. 단계 키 `photometry`, URL 해시(`#photometry/<n>`), manifest 키는 그대로다.
- 페이지 표시(0.6.0 「변경」의 단계별 열 서술을 대체한다; 그림 파일은 그대로):
  - WCS: 왼쪽 `wcs_offsets` · 오른쪽 `wcs_cutouts` 2열. 열 비율을 새 그림 비율에 맞춰 약 1.6 : 1로 둔다.
  - 배경: `bkgmap` 위, `bkgrms` 아래 1열 2행(0.6.0: 2열).
  - ZP Calibration: 그림 4장이면 2×2, 한 화면 안에서 가장 크게. 높이 제한 때문에 남는 폭이 있으면 두 열의 그림이 가운데로 모인다(2열인 WCS도 같다). 3장이면 2열 2행, 2장 이하면 한 행.
  - 전처리·마스크 1열, 폭 760 px 이하에서 한 열로 접는 것, 한 화면 맞춤은 0.6.0과 같다.
  - 광도곡선·SED 탭: 좌우 2열. 왼쪽 열(폭 최소 320 px, 약 38%)에 좌표 입력과 전체 시야 이미지, 오른쪽 열에 광도곡선/SED 그림. 폭 760 px 이하에서는 한 열(위: 좌표·이미지, 아래: 그림)이다.
    왼쪽 열 전체(좌표 입력, 이미지, 설명)가 스크롤 없이 한 화면에 들어오도록 이미지 크기를 창 높이에 맞춘다(0.6.0의 "창 높이의 60% 이하"를 대체한다). 이미지 상자는 늘 이미지 비율 그대로라 클릭 위치 계산은 같다.
    이를 위해 왼쪽 열의 좌표 안내문, 시야 이미지 설명(`대표 프레임 N. 필터|유닛|시각 · 검출 n개 · 휠 확대 · 끌기 이동 · 클릭 선택`), 좌표 오류 문구를 한 줄로 줄이고(넘치면 말줄임), 전체 문구는 마우스를 올리면 보이게 했다. RA·Dec 칸 폭도 줄여 한 줄에 놓았다.
    이미지는 최소 160 px 높이를 지키며, 그보다 작아져야 하는 낮은 창(창 안쪽 높이 약 560 px 미만, 표 모드는 약 600 px 미만)에서만 왼쪽 열이 넘친다.
    오른쪽 그림은 폭이 좁아진 만큼 글자도 작게 보인다.
  - 전체 시야 이미지의 휠 한 칸 배율: ×1.25 / ×0.8에서 ×1.1 / ×(1/1.1)로 바꿨다(더 천천히). 최대 40배는 같다.
- `visualize()` 시그니처·반환, CLI, `sources.json`, `fov.jpg`, 측광 수치는 바뀌지 않는다.

## 0.6.0 (2026-11-02)
### 파괴적
- `recipes.visualize`: 마스크 단계의 갤러리 그림 파일이 바뀐다. 프레임 폴더(`<stem>/`)에 `invalidmask.png`를 더 쓰지 않고 ezphot의 `.srcmask.png`(Previous Mask 패널 포함) 링크도 만들지 않는다.
  대신 visualize가 그리는 `mask.png` 한 장(3패널: ① 원본 이미지 8×8 평균 ② background source mask = `.srcmask` ③ invalid object mask = `.invalidmask`)을 쓴다.
  그래서 **`manifest.json`의 프레임별 `mask` 목록 값**이 `["<stem>/invalidmask.png", "<stem>/<file>.srcmask.png"]` 형태에서 `["<stem>/mask.png"]`로 바뀐다. `manifest.json`의 키(`run`·`lightcurve`·`images`, 이미지 경로, 단계 이름)는 그대로라
  0.5.5의 "`manifest.json` 키 불변"은 유지되지만, 목록 안의 파일 이름을 고정해 읽는 코드는 `mask.png`로 바꿔야 한다. 사용자가 갤러리에서 빼기를 요청한 그림이라 옛 파일을 함께 쓰는 deprecation 기간은 두지 않았다.
  같은 갤러리를 다시 그리면 이전 실행의 `invalidmask.png`·`.srcmask.png` 링크가 프레임 폴더에 남지만 페이지·manifest는 가리키지 않는다.
  - 마스크 파일 하나가 없으면 그 패널에 "none (no .srcmask file)" 등을 표시한다. 둘 다 없으면 이전처럼 마스크 그림이 없다. 마스크는 읽기 전용으로 열고, 블록은 픽셀 하나라도 마스크되면 칠한다. 마스크 비율은 8의 배수로 자른 영역 기준이다.
  - 원본 이미지 패널 때문에 마스크 단계가 과학 이미지 전체를 8행 띠로 한 번 읽는다(이전에는 컷아웃만). 프레임당 읽는 양이 이미지 크기만큼 늘어난다.
### 변경
- `recipes.visualize`: `wcs_offsets.png`(파일 이름 그대로)의 세 패널(시야 분포, 중심 거리 의존성, 히스토그램)을 가로 한 줄(16×4.6 in)에서 세로로 쌓은 작은 그림(5×10 in)으로 바꿨다. 제목이 두 줄이 된다. 내용·계산은 같다.
- 페이지 배치: 탭 순서가 개요 → 전처리 → WCS → 마스크 → 배경 → 측광 → 광도곡선 → SED가 된다(이전: 광도곡선·SED가 개요 바로 뒤).
  단계 탭의 그림은 창 높이에 맞춰 줄여 스크롤 없이 한 화면에 들어간다(이미지 이름 줄 아래 남는 높이 기준, 최소 240 px; 그림 설명은 한 줄로 자르고 전체 문구는 마우스를 올리면 보인다). 전처리·마스크 1열, WCS는 왼쪽 `wcs_offsets` · 오른쪽 `wcs_cutouts`, 배경 2열,
  측광은 그림 3장 이상이면 2행(5장: 3열 target_cutout, refcat, zp_mag / zp_color, zp_2d; target_cutout 없는 4장: 2열), 2장 이하면 한 행. 폭 760 px 이하에서는 한 열로 접고 높이 제한을 푼다. 그림을 누르면 원본 PNG가 크게 열리는 것은 같다.
  그림 파일(전처리·배경·측광)과 썸네일은 그대로이고 페이지에서 표시 크기만 바뀐다.
### 추가
- `recipes.visualize`: 광도곡선·SED 탭의 좌표 입력 아래에 **대표 프레임 한 장의 전체 시야 이미지**를 넣었다. 이미지를 클릭하면 그 프레임에서 가장 가까운 검출의 RA·Dec(`X_WORLD`·`Y_WORLD`)가 좌표 칸에 채워지고
  0.5.5의 카탈로그 매칭(프레임마다 반경 안 최근접)이 바로 적용된다. 표 모드(`--lightcurve`)에서 클릭해도 "카탈로그 매칭으로 보기"처럼 매칭 모드로 바뀐다. `--ra`/`--dec` 없이 만든 갤러리에서도 동작한다.
  - 대표 프레임 = 0.5.5 `sources.json`에 들어가는 SNR > 3 검출이 가장 많은 프레임(동률이면 갤러리 순서에서 앞의 것). 페이지에 그 프레임 이름(번호와 개요 표의 이름)과 검출 수를 적는다.
    한 장이라 다른 유닛·필터 프레임의 시야와 다를 수 있다.
  - 이미지: summary `path`(처리된 이미지, 읽기 전용 memory map)를 f × f 블록 nanmean으로 줄인 JPEG `fov.jpg`(갤러리 폴더, f = ceil(NAXIS1 / 1200), 폭 NAXIS1 // f, 약 1200 px)를
    ZScale(선형, NaN은 검정)로 쓴다. Y가 큰 쪽이 위(다른 그림의 `origin='lower'`와 같음). raw 파일은 열지 않는다.
  - 그 프레임의 SNR > 3 검출을 초록 원으로 겹쳐 그리고(체크박스로 끄고 켬), 휠로 확대·축소(최대 40배, 업스케일; 세로 휠 값이 0인 가로 스크롤은 무시), 끌기로 이동, "전체 시야"로 되돌린다.
    이미지 높이는 창 높이의 60% 이하로 줄여 표시한다(가로세로 비율 유지).
    5 px 넘게 움직이면 끌기, 아니면 클릭이다. 클릭 위치에서 화면 20 px 안에 검출이 없으면 좌표를 바꾸지 않고 "근처에 검출 없음"을 띄운다. 고른 검출은 빨간 원으로 표시하고, 좌표 칸에 직접 입력하면 지운다.
  - 픽셀 대응은 WCS를 쓰지 않고 그 프레임 `.cat`의 `X_IMAGE`·`Y_IMAGE`(1-based)로 한다: 이미지 가로 위치 = (X_IMAGE − 0.5) / f. 채우는 RA·Dec는 매칭에 쓰는 값과 같아서 대표 프레임은 매칭 거리 0″로 잡힌다.
    `X_IMAGE`·`Y_IMAGE` 중 하나라도 이미지 범위(0.5 – NAXIS + 0.5) 밖이거나 이미지를 만들다 실패하면 이미지 대신 이유를 탭에 띄우고, 좌표 입력은 그대로 쓸 수 있다.
  - `file://`로 연 페이지에서도 이미지·클릭·좌표 채우기는 되지만, 매칭은 0.5.5와 같이 http 서버가 필요하다.
- 산출물: 갤러리 폴더에 `fov.jpg`(수백 KB)가 추가되고, `index.html`의 페이지 데이터에 `fov` 키(이미지 크기, 대표 프레임 번호, 그 프레임 검출의 이미지 위치·RA·Dec; 경로 없음)가 추가된다.
  이 기능으로는 `manifest.json`·`sources.json`의 형식과 내용, `visualize()` 시그니처·반환, CLI, 측광 수치가 바뀌지 않는다(`manifest.json`의 마스크 목록은 위 「파괴적」).
  같은 objname을 다시 그렸는데 이번에는 `fov.jpg`를 만들지 않으면(`.cat` 없음, 실패) 이전 `fov.jpg`가 폴더에 남지만 페이지는 읽지 않는다.
- 시간·메모리: 모든 프레임을 그린 뒤 대표 프레임 한 장을 메인 프로세스에서 한 번 더 읽는다(행 띠 단위라 추가 메모리는 축소 이미지 정도. 압축·BZERO/BSCALE 이미지는 전체를 디코드).

## 0.5.5 (2026-11-01)
### 추가
- `recipes.visualize`: 광도곡선·SED 탭에 좌표 입력(RA, Dec, 매칭 반경 기본 5″ = `target_photometry --radius` 기본값)을 넣었다.
  visualize가 각 프레임의 `.cat`(summary `path`의 `.cat`, `MAGSKY_APER_3` 열이 있는 것. stack summary는 `coadd_` 카탈로그, dia summary는 `path`(차감 대상 이미지)의 카탈로그를 쓴다. 차영상(`subtracted`, `sub_`) 카탈로그는 쓰지 않는다)에서
  **SNR > 3**(`MAGERR_APER_3 < 1.0857/3`) 이고 좌표·등급이 유한한 검출의 `X_WORLD`·`Y_WORLD`·`MAGSKY_APER_3`·`MAGERR_APER_3`·`SATDIST`(열이 없으면 null)를 모아
  갤러리 폴더에 `sources.json` 하나(열 지향 JSON, 경로 없음, 프레임은 페이지 이미지 목록 순서 번호로 연결, 프레임 시각 = summary `obsdate`의 MJD)로 쓴다.
  페이지는 탭을 처음 열 때(`--ra`/`--dec`가 있고 `--lightcurve`가 없을 때) 또는 좌표를 적용할 때 이 파일을 받아, **프레임마다 반경 안 가장 가까운 검출 하나**로 점을 만들고
  기존 시간 bin·합치기·SED·툴팁을 그대로 쓴다(툴팁에 매칭 거리·SATDIST 추가). 반경 안 검출이 2개 이상인 프레임 수를 페이지에 적는다.
  좌표는 도(십진수, 지수 표기 포함) 또는 `hh:mm:ss` / `±dd:mm:ss`. RA 칸 값이 RA 하나로 읽히지 않고 2개 또는 6개 값으로 나뉘면 "RA Dec"로 보고 두 칸을 그 값으로 바꾼다(Dec 칸이 차 있어도).
- 기본 상태: `--lightcurve`가 있으면 이전처럼 표 기반 광도곡선·SED로 시작하고(좌표 칸에는 `--ra`/`--dec`), "카탈로그 매칭으로 보기"·"표로 돌아가기"로 전환한다.
  `--lightcurve` 없이 `--ra`/`--dec`가 있으면 그 좌표의 카탈로그 매칭으로, 둘 다 없으면 좌표 입력을 기다린다.
- 이 모드의 값은 `target_photometry` 표와 다를 수 있다: `target_photometry`는 여러 카탈로그의 반경 안 검출을 군집으로 묶어 고르고, 여기서는 프레임마다 따로 최근접을 고른다(혼잡한 곳에서 다른 광원이 잡힐 수 있음).
  SNR ≤ 3인 프레임의 측정은 들어 있지 않아 어두운 대상은 일부 프레임(또는 전부)이 빠진다. `ZPERR`는 오차에 합성하지 않는다(표 모드와 같음).
  `sources.json`은 `file://`로 연 페이지에서는 읽을 수 없어(브라우저가 막음) 안내 문구가 나온다 — http 서버로 연다.
  페이지는 `sources.json`을 캐시 검증(`no-cache`)으로 받고, `index.html`과 같은 실행에서 쓴 파일인지 토큰으로 확인한다. 다르면(갤러리를 다시 만든 뒤 새로고침하지 않은 경우 등) 다시 만들고 새로고침하라는 문구를 띄운다.
- 0.5.1 항목의 "`lightcurve`를 주지 않으면 페이지 표시는 이전과 같다"는 이제 맞지 않는다: `MAGSKY_APER_3`가 있는 `.cat`이 하나라도 있으면 `lightcurve` 없이도 광도곡선·SED 탭이 생긴다.
  `.cat`이 없는 갤러리는 이전과 같다. `index.html`의 페이지 데이터에 `srcs` 키가 추가되고, `lcdata`(표 모드)는 바뀌지 않는다.
- 크기·시간: `sources.json`은 프레임당 SNR > 3 검출 수에 비례한다(예: FEIGE110 576장, 303,045행에서 12.7 MB, 압축 안 함). 각 프레임의 `.cat`을 그 장을 그리는 프로세스에서 한 번 더 읽는다.
  `.cat`을 읽다 실패하면 그 프레임 상태에 `detections for coordinate input failed: …`가 붙고 개요 표에서 그 프레임이 실패(빨강)로 표시된다. 갤러리는 계속된다.
- `visualize()` 시그니처·반환, CLI, `manifest.json` 키, 기존 그림 파일, 측광 수치는 바뀌지 않는다. 같은 objname을 다시 그렸는데 이번에는 `.cat`이 없으면 이전 `sources.json`이 폴더에 남지만 페이지는 읽지 않는다.

## 0.5.4 (2026-10-31)
### 수정
- `Preprocess.correct_bdf`·`correct_bd`·`correct_bias`·`correct_dark`·`correct_flat`(그리고 이를 부르는 `ImageMethod.correct_bdf` 등, `recipes.process_images --preprocess`):
  보정본 헤더(그리고 `.info`)의 `BIASPATH`·`DARKPATH`·`FLATPATH`가 실제로 읽은 master 파일의 경로(master 이미지 객체의 `path`, resolve된 경로)가 된다.
  이전에는 `savepath.savepath`(ezphot이 그 master를 저장한다면 쓸 `CALIBDATA_MASTERDIR`/`CALIBDATA_DIR` 레이아웃 위치)를 적어서,
  master를 레이아웃 밖(다른 디렉터리·심볼릭 링크 너머)에서 읽으면 존재하지 않는 경로가 기록됐다(틀린 값이라 deprecation 없이 바로 바꿨다).
  그래서 `recipes.visualize`의 전처리(preprocess) 그림이 master를 찾지 못해 빠지던 보정본도 새로 처리하면 그림이 나온다.
  ezphot이 만들어 `write()`로 저장한 master는 이전과 같은 값이다. 저장하지 않은 master의 경로는 객체가 만들어질 때 받은 경로다(stack 결과는 아직 없는 `.com.fits`, `correct_*` 결과는 입력 파일).
  보정에 쓰는 master는 저장했거나 파일에서 읽은 것을 넘긴다.
  `ErrormapGenerator`가 오차 맵 헤더에 쓰는 같은 키는 이미 `path`를 써서, 이제 두 값이 같은 기준이다.
- 수치 변경 없음(보정 계산은 그대로이고 헤더·`.info`의 경로 문자열만 바뀐다). **이미 처리된 파일의 헤더·`.info`는 바뀌지 않는다** — 그 파일들의 전처리 그림이 필요하면 다시 처리한다.
  이 값을 읽어 master를 다시 찾는 코드가 있으면(ezphot 안에는 `recipes.visualize`뿐) 새 값 기준으로 동작한다.
- `recipes.visualize`: summary에 `filter`·`telname`이 비어 있는 행이 있으면 상단 부제를 만들다 `TypeError`로 갤러리 전체가 멈추던 것(pandas 3에서 `astype(str)`가 결측을 문자열로 바꾸지 않음)을 고쳤다. 결측은 부제 나열에서 뺀다.
### 변경
- `recipes.visualize`: 갤러리의 프레임 표시 이름이 `필터|유닛|관측시각: raw 파일 이름`이 된다(예: `m400|7DT01|2026-10-27 01:23:45: <raw 파일명>.fits`).
  raw 파일 이름은 summary의 `raw_path`(0.5.3) basename이고, 그 열이 없거나 비어 있으면 `path`의 basename이다. 관측시각은 summary `obsdate`(초 단위, `T` 대신 공백).
  이전 이름은 `<대상>_<번호>`였다. 이 이름은 개요 표 첫 열과 프레임 탭 드롭다운에 쓰이고, 드롭다운 뒤에 붙던 `· 필터 · obsdate`는 이름에 들어가므로 뺐다.
- `recipes.visualize`: 갤러리 안의 프레임 순서가 summary 행 순서에서 **중대역 필터(`m###`·`m###w`) 먼저, 그다음 광대역(그 밖의 필터)**, 각 그룹 안에서 필터 파장순,
  같은 필터 안에서 `telname`, 그다음 `obsdate` 순이 된다(같은 키는 summary 순서 유지, `obsdate`가 없거나 읽을 수 없는 행은 같은 필터·유닛 끝). 필터 파장은 `SYNPHOT_FILTERDIR`의 `<filter>.csv` 응답 가중 평균, 없으면 `m###` 이름 숫자이고,
  파장을 모르는 필터는 그룹 끝에 이름순이다. 0.5.2 항목의 "그림·`index.html`·`manifest.json`의 내용과 순서, `[visualize]` 진행 출력의 순서는 입력 순서 그대로"는
  이제 "정렬된 순서 그대로"로 바뀐다: `index.html`의 이미지 목록, `manifest.json`의 `images` 키 순서, 진행 출력 순서, 병렬일 때 혼자 먼저 그리는 첫 장이 정렬 순서를 따른다.
  `max_images`(`--max-images N`)는 이전처럼 정렬 전 summary의 처음 N행이다. 이미지별 출력 폴더(`<stem>/`)·그림 파일·`manifest.json` 키 집합, 상단 부제의 필터·유닛 나열은 그대로다.
  페이지 주소의 프레임 번호(`#<단계>/<N>`)는 정렬 순서 기준이라 이전에 저장한 주소는 다른 프레임을 가리킬 수 있다.
- `recipes.visualize`: 프레임 탭의 이미지 넘기기 막대(← 드롭다운 →)가 스크롤해도 상단 헤더 바로 아래에 고정된다(광도곡선·SED 탭의 컨트롤은 그대로). 드롭다운 최대 폭 640 → 960px.
- `visualize()` 시그니처·반환, CLI, 출력 파일명·위치, 수치는 바뀌지 않는다.

## 0.5.3 (2026-10-30)
### 수정
- `recipes.process_images`: summary CSV의 `path`가 체인이 마지막에 쥔 이미지(`correct_bdf`·`platesolve`·`photometric_calibration`이 돌려준 이미지)의 경로가 된다.
  이전에는 항상 입력 경로였고, `--preprocess`(`do_preprocess=True`)이면 보정본과 그 sidecar(`.cat`·`.refcat`·`.info` 등)는 `SCIDATA_DIR`에 따로 쓰이는데
  `path`는 raw 입력을 가리켜서 하위 레시피가 산출물을 찾지 못했다(틀린 값이라 deprecation 없이 바로 바꿨다).
  단계가 실패하면 `path`는 그때까지의 마지막 경로다(`correct_bdf`가 실패하면 입력 경로).
- 그래서 `--preprocess` 요약을 받는 `stack`·`dia`는 raw 대신 보정본을, `target_photometry`·`visualize`는 보정본과 그 sidecar를 쓰고, `qa`는 보정본 기준 행을 판정한다.
- `--preprocess` 없이도 입력이 아카이브 레이아웃(`savepath`) 밖에 있으면 `platesolve`나 `photometric_calibration`의 `write()`가 이미지를 `savepath`에 쓰므로
  `path`가 입력과 달라질 수 있다(이전에는 입력 경로로 남았다).
- `path`는 savepath 형태(config `SCIDATA_DIR` 기준, resolve하지 않음)다. `photometric_calibration`이 끝에서 이미지를 다시 쓰므로 photcal까지 간 모든 행이 그렇다.
  `SCIDATA_DIR`이 심볼릭 링크를 지나면 같은 파일이라도 `path` 문자열이 이전 값·`raw_path`(resolve된 경로)와 다를 수 있다(예: `/home/...` 대 `/lyman/...`).
  경로를 비교할 때는 `Path(p).resolve()`로 맞춘다.
- 이전 summary와 새 summary를 섞어 `stack`·`dia`에 넘기지 않는다. `path`를 문자열로 중복 제거하므로 같은 프레임이 두 번 처리될 수 있다.
- 수치 변경 없음(처리 자체는 그대로이고 summary의 경로 값만 바뀐다). 이전에 만든 summary CSV는 그대로이므로, `--preprocess`로 만든 이전 요약을 하위 레시피에 쓰려면 다시 처리한다.
### 추가
- `recipes.process_images`: summary CSV에 `raw_path` 열(`path` 바로 뒤). 항상 `process_image`가 받은 입력 이미지의 경로(이미지 객체가 resolve한 경로)다.
  `stack` summary 행은 스택을 `process_image`에 넘기므로 `raw_path`가 스택 경로이고, 스택 자체가 실패한 행은 비어 있다.

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
