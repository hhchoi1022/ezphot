# ezphot 레시피 API 계약

`ezphot.recipes`의 처리 레시피(process_images, stack, dia, forced_photometry)와 결과 레시피(target_photometry, qa)를
다른 코드에서 부를 때 기대해도 되는 것. 이 문서는 같은 커밋의 코드 동작을 적는다. 여기 적힌 것이 바뀌면 아래 「이 계약을 바꿀 때」를 따른다.
`visualize`·`status`는 화면용이라 이 계약에 넣지 않는다(입력으로 읽는 요약 열은 각 레시피의 열과 같다).

## 1. 공통 규칙

- **행 단위 실패**: 처리 레시피는 영상(또는 스택 묶음) 하나의 처리 실패로 멈추지 않는다. 요약 행마다
  `success`(bool), `step`(실패한 단계 이름, 성공하면 `'done'`), `error`(`"<예외 타입>: <메시지>"`, 성공하면 빈 칸).
  실패한 행의 값 열(측광 값·헤더 값)은 비어 있다. 단계 정보 열(dia의 `reference`·`subtracted`·`n_candidates`·`n_transients`,
  forced_photometry의 `bkg_source`·`bkgrms_source`·`catalog`·`n_ref`)은 실패하기 전에 정해진 만큼 채워져 있을 수 있다.
  실패한 영상의 산출물(카탈로그 등)은 없거나 중간 상태일 수 있으므로 `success`가 참인 행만 쓴다.
- **영상을 열지 못할 때**(파일 없음, 파일 이름에서 망원경을 정하지 못함, 헤더 값 `DATE-OBS`·`EXPTIME` 등을 해석하지 못함): 비대화 실행
  (n_proc > 1의 워커, 또는 터미널·Jupyter가 아닌 프로세스)에서 forced_photometry는 그 영상의 타겟 행을, dia는 그 영상 행을 `step='load'` 실패로 남기고 계속한다.
  대화형 순차 실행(영상 하나나 n_proc = 1을 터미널·Jupyter에서)이면 ezphot이 망원경 등록 입력을 먼저 묻는다(답하지 않으면 `step='load'`).
  헤더를 아예 못 읽는 7DT 경로의 영상은 열기는 되고 뒤 단계에서 실패한다. dia의 `load` 행은 `path`(요약의 경로)와 `ra`/`dec`만 채워져 있다.
  stack은 묶음을 나누기 전에 모든 입력 영상을 열므로 처리 시작 전에 예외가 난다. process_images는 호출자가 연 영상 객체를 받는다.
  `out=None`이면 첫 영상을 열어 objname을 정하므로, 첫 영상을 열 수 없으면 dia·stack·forced_photometry 모두 처리 시작 전에 예외가 난다.
- **함수 단위 실패**: 처리할 것이 없으면 `NothingToDo`(`ValueError`의 하위 클래스, `ezphot.recipes.NothingToDo`; 요약 CSV를 쓰기 전에 난다),
  `config`에 모르는 섹션이 있으면 `KeyError`, 잘못된 인자 값이면 `ValueError`. `NothingToDo`의 메시지(바꾸면 계약 변경):

  | 레시피 | 메시지 | 경우 |
  | --- | --- | --- |
  | process_images | `no images to process` | 영상 목록이 빔 |
  | stack | `no successfully processed images in the summary` | 요약에 성공 행이 없음 |
  | stack | `no group has more than one image` | 묶음이 모두 영상 하나뿐(스택할 것 없음) |
  | dia | `no successful images in the summary` | 요약에 성공 행이 없음 |
  | forced_photometry | `no images given` | 영상 경로·요약 성공 행이 없음 |
  | target_photometry | `no catalogs given` | 카탈로그가 없음 |

  `except ValueError`는 그대로 `NothingToDo`도 잡는다. 이전 버전(0.7.12 이하)은 같은 메시지의 `ValueError`였다.
  stack은 자기 섹션(`stack_group`, `stack_prepare`, `stack`) 밖의 섹션을 process_images로 넘기므로 모르는 섹션의 `KeyError`가 첫 묶음 처리 때 난다.
- **`config=`**: `{섹션: {키: 값}}`. 섹션마다 `ezphot.recipes.defaults`의 같은 이름(대문자) 사전의 사본에 덮어쓴다. 기본값은 ezphot 단독 사용 기준이다.
  레시피가 호출 직전에 정하는 키는 config로 바꿀 수 없고 조용히 무시된다: process_images·stack의 photcal `mag_lower`/`mag_upper`(`defaults.MAG_RANGE` +
  `mag_faint_shift`)·`ra_deg`/`dec_deg`(= `ra`/`dec`), forced_photometry는 2절의 고정 키.
- **`out=`**(처리 레시피): 기본 `~/ezphot/log/recipes/<objname>/<레시피>_<YYYYmmdd_HHMMSS>.csv`.
  `<objname>`은 첫 입력 영상의 objname(없으면 `unknown`). 실행을 시작할 때 정하고 폴더를 만든다. 실행 중에는 그 옆에
  `<out>.progress.json`(내부 형식, `recipes.status`가 읽음)을 쓰고, 요약 CSV는 끝에 한 번 쓴다(pandas, index 없음).
- **`n_proc`**: 영상(stack은 묶음) 단위 프로세스 수. 행 순서는 작업 순서다 — 영상 입력 순서, stack은 `divide_images`가 만든 묶음 순서
  (입력 순서에서 바로 알 수 없으므로 `inputs` 열로 맞춘다).
- **CLI 종료 코드**: 처리 레시피는 모든 행이 성공하면 0, 아니면 1. target_photometry는 표가 나오면 0, 아니면 1. qa는 모두 통과하면 0, 아니면 1.
- **산출물 위치**: process_images·stack·dia는 ezphot 설정의 아카이브 규칙(영상의 `savedir`, `SCIDATA_DIR/<observatory>/<telkey>/<objname>/<telname>/<filter>/`).
  forced_photometry는 입력 영상과 같은 폴더다. 입력이 심볼릭 링크면 따라간 실제 파일의 폴더에 쓰고 그 폴더의 `subbkg_<img>*`를 지운다.
  요약 `path`는 resolve된 절대 경로다(링크 경로가 아니다).

## 2. 레시피별

### process_images
```python
process_images(images, ra=None, dec=None, do_preprocess=False, do_platesolve=True, config=None,
               mag_faint_shift=0.0, n_proc=4, out=None, verbose=True, dark_exptime_fallback=False,
               master_max_days=100) -> pandas.DataFrame
process_image(target_img, ra=None, dec=None, do_preprocess=False, do_platesolve=True, config=None,
              mag_faint_shift=0.0, clear=True, bkgrms=None, dark_exptime_fallback=False, master_max_days=100) -> dict
```
- `images`: `ImageSet` 또는 `ScienceImage` 목록. config 섹션: `circularmask`, `invalidmask`, `sourcemask`, `bkgmap`, `bkgrms`, `photometry`, `photcal`.
- `master_max_days`(CLI `--master-max-days`): `do_preprocess`일 때 `correct_bdf`가 master bias·dark·flat 세 가지를 찾는 기간(일, `get_masterframe`의 `max_days`).
  기본 100은 이전의 고정값과 같다.
- 열: `path`(마지막으로 다룬 영상 = 이후 레시피가 읽는 영상), `raw_path`(입력), `telname`, `filter`, `obsdate`, `exptime`, `success`, `step`, `error`,
  `n_ref`(ZP 기준성 수), 헤더 값 `SEEING`, `ELLIP`, `SKYVAL`, `SKYSIG`, `ZP_APER_3`, `ZPERR_APER_3`, `UL5SKY_APER_3`, `K_COLOR_APER_3_g-r`.
- 빈 입력: `NothingToDo("no images to process")`(1절).

### stack
```python
stack(summaries, ra=None, dec=None, config=None, mag_faint_shift=0.0, n_proc=4, out=None, verbose=True) -> pandas.DataFrame
stack_group(paths, ra=None, dec=None, config=None, mag_faint_shift=0.0) -> dict
```
- `summaries`: process_images 요약 CSV 경로(하나 또는 목록), `success` 행의 `path`를 쓴다. 영상 하나뿐인 묶음은 스택하지 않는다(행 없음).
- 열: 스택 영상에 대한 process_images 열(`path`, `raw_path` = 스택 영상) + `n_combined`, `inputs`(`;`로 이은 입력 경로).
  스택 단계에서 실패한 행은 `path`가 비고 `step='stack'`.
- 빈 입력: `NothingToDo`(성공 행 없음, 또는 영상이 둘 이상인 묶음 없음; 메시지는 1절).

### dia
```python
dia(summaries, ra=None, dec=None, reference=None, config=None, n_proc=4, out=None, verbose=True) -> pandas.DataFrame
dia_image(target_img, ra=None, dec=None, reference=None, config=None, clear=True) -> dict
```
- `summaries`: process_images 또는 stack 요약 CSV. config 섹션: `dia`, `forced_photometry`, `forced_photometry_sub`.
- `ra`, `dec`(deg): 타겟 하나. 숫자 또는 원소 하나짜리 리스트·튜플·배열을 받고, 그 밖(원소 둘 이상, 숫자가 아님)이면 `dia()`가 처리 전에 `ValueError`를 낸다.
  둘 다 주어야 forced 측광을 한다. `dia_image`는 이 검사 없이 받은 값을 쓰고 예외를 내지 않는다.
- 열: `path`(science 영상), `telname`, `filter`, `obsdate`, `exptime`, `success`, `step`, `error`, `reference`, `subtracted`(차영상 `sub_<name>`),
  `n_candidates`, `n_transients`, `ra`/`dec`를 주면 차영상 forced 값 `MAGSKY_APER_3`, `MAGERR_APER_3`, `ZPERR_APER_3`, `UL5SKY_APER_3`,
  그 뒤에 `ra`, `dec`(넘긴 좌표, 없으면 빈 칸), `noise_ratio`, `host_source`(forced_photometry와 같은 뜻, forced를 하지 않으면 빈 칸),
  정규화 QA 열 `norm_resid`, `norm_resid_n`, `norm_resid_mad`, `ksum`, `ksum_exp`, `ksum_ratio`, `norm_flag`(아래).
  `telname`·`filter`·`obsdate`·`exptime`은 science 영상 헤더 값이다(차영상 헤더의 `MJD-OBS`는 쓰지 않는다).
- 차영상 forced는 forced_photometry의 `forced_image(<sub_ 영상>, ra, dec, 'header', {forced_photometry, forced_photometry_sub 섹션})`와 같다:
  카탈로그 `sub_<name>.fits.circ.cat`(4절), ZP는 차영상 헤더(science 영상 헤더에서 이어짐)의 `ZP_APER_k`, 국소 배경·오차는 forced_photometry의 차영상 규칙(3절).
  `ra`/`dec`를 주었는데 차영상 헤더에 유한한 `ZP_APER_3`가 없으면 그 행은 `success=False`, `step='apply_zp'`이고 값 열은 비어 있다(0.7.12까지는 성공에
  빈 값; `ra`/`dec` 없이 돌리면 ZP를 보지 않아 그대로 성공).
  forced의 다른 실패는 `step='forced_photometry'`(그 오류 문구 그대로). 차영상 전체 카탈로그의 `apply_zp`가 실패해도 `step='apply_zp'`다.
- 정규화 QA(값과 `success`에 영향 없음, 파일을 쓰거나 지우지 않음): hotpants가 reference를 science 밝기에 맞춘 정도.
  - `norm_resid`: 필드 별의 10″ 개구 플럭스 비 차영상/science의 중앙값(맞으면 0). 별 = science 영상 옆 `<img>.fits.refcat`(process_images/stack의 ZP 기준성)의
    `X_WORLD`/`Y_WORLD`. 차영상 `sub_<name>`과 같은 격자의 `sci_<name>`(DIA가 쓴 science 재투영)에서 별마다 15–25″ 3σ 클립 평균을 빼고 10″ 지름 개구 합을 잰다.
    반폭 25″ + 2 px 상자가 두 영상 안에 있고 그 안의 픽셀이 모두 유한하고 유효하며(|값| > 1e-20: `sci_`의 0과 `sub_`의 hotpants 무효 채움값 1e-30을
    뺀다 — hotpants가 무효로 둔 포화 영역(DIA `iu`/`tu` 초과)에 걸친 별도 여기서 빠진다) science 플럭스가 양수인 별만 쓴다. `norm_resid_n` = 쓴 별 수(정수),
    `norm_resid_mad` = 1.4826 × MAD.
  - 별은 밝은 ZP 기준성이다(적색 필터에서 약 10–14 mag). 그래서 적색에서는 `norm_resid`가 어두운 천체(타겟·호스트)에 맞는 값보다 0.5–1 % 낮게 나올 수 있다
    (T08635 검증: 적색 중앙 −1.05 %, QA 별 등급 범위로 제한하면 −0.28 %). 3.5 % 임계 판정에는 영향이 작다.
  - 별이 10개 미만이면(refcat·`sci_`가 없을 때 포함) `norm_resid`·`norm_resid_mad`·`norm_flag`는 빈 칸이다(플래그를 세우지 않음, `norm_resid_n`은 그 수).
    QA가 예외로 끝나면(오류 한 줄을 출력) `norm_resid`·`norm_resid_n`·`norm_resid_mad`·`norm_flag`는 빈 칸이다(`ksum` 열은 예외 전에 채운 값이 남을 수 있다).
    DIA 전에 실패한 행은 QA 열이 모두 빈 칸이다.
  - `norm_flag` = |`norm_resid`| > `defaults.DIA_NORM_RESID_MAX`(0.035). 참이면 그 차영상의 플럭스 스케일이 어긋나 호스트 위 값에
    (스케일 오차 × 호스트 플럭스)만큼의 계통이 섞일 수 있다. **그 프레임의 `MAGSKY`는 보정하지 않는다** — 사용처가 그 프레임을 빼거나 보정한다.
    T08635(호스트/트랜지언트 10″ 안 1.6–1.8배)에서 플래그 세 장 가운데 m650 f0(norm_resid 약 −4 %)은 약 0.08 mag 어둡고,
    m675 f1·f2(약 +6 %, +5.5 %)는 약 0.11 mag 밝았다.
  - `ksum`: 차영상 헤더 `KSUM00`(hotpants 커널 합). `ksum_exp` = 10^(−0.4 (ZP_ref − ZP_sci))(`ZP_APER_3`, reference 헤더 / 차영상 헤더)와
    `ksum_ratio` = `ksum`/`ksum_exp`는 헤더 `CONVOL00`이 `TEMPLATE`(reference를 convolve)일 때만 채운다. 정상이어도 개구 손실 차로 1에서 몇 % 벗어나므로
    플래그에는 쓰지 않는다.

### forced_photometry
```python
forced_photometry(images, ra, dec, calibrate='header', config=None, n_proc=4, out=None, verbose=True) -> pandas.DataFrame
forced_image(target_img, ra, dec, calibrate='header', config=None, clear=True) -> list[dict]
```
- `images`: FITS 경로와 process_images/stack 요약 CSV(`*.csv`, `success` 행의 `path`)를 섞어 하나 또는 목록으로. 단일·스택·차 영상 어느 것이든 된다.
  dia 요약 CSV를 주면 `path`(science 영상)를 쓴다 — 차영상은 경로로 준다.
- `ra`, `dec`: 스칼라 또는 같은 길이의 배열(deg). 한 영상의 모든 타겟을 한 번의 forced 호출로 잰다(같은 영상에 다시 부르면 `.circ.cat`을 덮어쓴다).
- `calibrate`: `'header'`(기본) — 영상 헤더 ZP로 `apply_zp`. `'reference_stars'` — 영상의 기준성을 함께 forced로 재고 그 카탈로그로 `photometric_calibration`(5절).
- config 섹션: `forced_photometry`, `forced_photometry_sub`(차영상에만, 아래 배경), `photcal`. defaults 위에 얹는 레시피 기본(config로 바꿀 수 있음): `forced_photometry`·`photcal` 모두 `visualize=False`, `save_fig=False`
  (그림 없음), photcal `magnitude_key='MAG_APER_3'`, `magnitudeerr_key='MAGERR_APER_3'`(defaults의 `MAG_AUTO`는 forced 카탈로그에 없다).
  레시피가 고정하는 키(config로 바꿀 수 없음): forced_photometry `save=True`(False면 `.circ.cat`을 지운다), `reference_stars`의 photcal
  `update_header=False`, `save_image=False`, `save=True`, `mag_lower`/`mag_upper`(`defaults.MAG_RANGE`의 필터별 값), 영역 컷 없음(`ra_deg=dec_deg=None`, `inner_fraction=1.0`).
- `forced_image`는 처리 오류를 행으로 돌려주고, 인자 오류(config 섹션, calibrate, ra/dec 길이)는 예외로 낸다. 넘겨받은 `target_img`의 `savedir`를
  영상 폴더로 바꾸고 되돌리지 않으며, `clear=True`(기본)면 끝에 영상 데이터를 메모리에서 비운다.
- 배경: 차영상(`sub_` 접두어)은 배경 맵이 없다(bkgrms는 forced 측광이 영상에서 계산). 차영상에는 `forced_photometry` 섹션 위에
  `forced_photometry_sub` 섹션(`defaults.FORCED_PHOTOMETRY_SUB`)을 얹는다(같은 키는 이쪽이 이긴다). 기본:
  - `annulus_width_arcsec=60`, `annulus_errors=False`: 타겟마다 국소 고리의 5σ 클립 평균 × 개구 면적을 플럭스에서 뺀다(`bkg_source='annulus'`, 카탈로그 `BACKGROUND`).
    고리 안쪽 지름 = 가장 큰 개구 + 1.5 × seeing, 바깥 지름 = 안쪽 + 60″(seeing 2.5–4.9″에서 반경 약 9–12″ … 39–42″). 고리는 플럭스에만 쓰고
    `SKYSIG`·`UL3`/`UL5`·`FLUXERR`의 배경 항은 고리 std가 아니라 bkgrms 적분이다(3절).
  - `n_empty_apertures=500`: 3절의 잡음 비 `r_k`를 곱한다. 요약 `noise_ratio` = 10″ 개구의 r(빈 개구가 모자라면 빈 칸, 보정 없음).
  - `host_poisson=True`: 같은 폴더의 `conv_<name>`(hotpants 출력)이 convolve된 reference일 때(헤더 `CONVOL00`이 `TEMPLATE`) 그 개구 플럭스의
    Poisson 잡음을 `FLUXERR`/`MAGERR`에만 더한다. 요약 `host_source`(타겟별): `conv`(더함), `missing`(`conv_` 없음), `not template`(`conv_`가 convolve된
    science), `off`(`host_poisson=False`), `not finite`(그 타겟의 `conv_` 구멍·고리가 유한하지 않아 더하지 않음, 카탈로그 `HOST_FLUX_APER_k` NaN).
    한계 등급에는 넣지 않는다.
  - 0.7.12의 차영상 forced 값(dia 포함)을 다시 내려면, 기본 `forced_photometry` 섹션 기준으로
    `config={'forced_photometry_sub': dict(annulus_width_arcsec=None, n_empty_apertures=None, host_poisson=False)}`. `forced_photometry` 섹션에
    `annulus_width_arcsec=X`를 직접 주던 호출은 차영상에서 그 값이 이 섹션 값으로 덮이므로 `forced_photometry_sub=dict(annulus_width_arcsec=X,
    annulus_errors=True, n_empty_apertures=None, host_poisson=False)`로 낸다.
  - 자원: 차영상 forced 한 번에 전체 크기 배열 몇 개(bkgrms², 무효 마스크, `conv_` 사본)와 빈 구멍 측정이 더해진다. T08635 실측(60.9 Mpx 차영상 60장,
    4개씩): 워커 최대 RSS는 변하지 않았고(3.4–3.7 GB), 영상당 약 +1.5 s(+4 %, 60장 540 → 562 s)다.
  그 밖의 영상은 영상 옆 `.bkgmap`/`.bkgrms`
  사이드카(헤더 ZP를 정한 process_images/stack 측광의 배경, 스택은 전파 bkgrms). 사이드카가 없으면 영상에서 메모리로 계산한다 —
  소스 마스크 없이 시그마 클리핑으로만 소스를 빼므로 process_images 값과 다를 수 있다. 영상 옆에 마스크·배경 파일을 쓰거나 지우지 않는다.
- 열(영상 × 타겟 한 행): `path`, `target`(ra/dec 안의 번호), `ra`, `dec`, `telname`, `filter`, `obsdate`, `exptime`, `calibrate`, `success`, `step`, `error`,
  `n_ref`(`reference_stars`의 ZP 기준성 수, `header`는 빈 칸), `bkg_source`·`bkgrms_source`(`sidecar` | `image` | `none` | 차영상 고리면 `annulus`),
  `catalog`(`.circ.cat` 경로), `MAGSKY_APER_3`, `MAGERR_APER_3`, `ZP_APER_3`, `ZPERR_APER_3`, `UL5SKY_APER_3`, `noise_ratio`, `host_source`
  (뒤 둘은 차영상만, 그 밖은 빈 칸).
- 실패 표현: 한 영상의 타겟 행들은 `success`·`step`·`error`가 같다. 성공 판정은 두 방식 모두 카탈로그의 `ZP_APER_3`가 있고 유한한가이고,
  `reference_stars`는 `n_ref > 0`도 필요하다. 아니면 `success=False`이고 `step='apply_zp'`(header: 헤더에 `ZP_APER_3`가 없거나 유한하지 않음; dia도 같다) 또는 `step='photometric_calibration'`(reference_stars: 기준성 0개 등). 영상을 열 수 없으면 `step='load'`, refcat이 없으면 `step='refcat'`.
  `UL5SKY_APER_3`만 빈 칸이면 한계 등급을 구할 수 없었다는 뜻이고 성공은 그대로다. 99 같은 센티널은 쓰지 않는다(재지 못한 값은 NaN).
- 입력 FITS는 다시 쓰지 않는다. 영상 옆에 쓰는 파일: `<img>.fits.circ.cat`(+ `.info`), `reference_stars`면 `<img>.fits.circ.refcat`(+ `.info`).
  같은 영상에 이미 있는 `.circ.cat`(+ `.info`, 다른 도구가 만든 것 포함)은 덮어쓴다. `header`로 다시 돌려도 이전 `reference_stars`의 `.circ.refcat`은
  지우지 않고 남는다(새 `.circ.cat`과 무관한 낡은 파일).
  배경을 빼는 동안 임시 `subbkg_<img>` 영상 객체를 만들고 지우는데(파일로 쓰지 않음), 그 이름으로 시작하는 파일(`subbkg_<img>*`)이 영상 폴더에 남아 있으면
  지워진다(process_images 측광과 같은 동작).
- 차영상(`sub_`)에 dia와 다른 좌표로 돌리면 dia가 쓴 `sub_<name>.fits.circ.cat`을 덮어쓴다. `visualize --dia`는 dia 요약의 `ra`/`dec`에서 1″ 안의
  `.circ.cat` 행을 forced 위치로 쓰고(그런 행이 없거나 요약의 `ra`/`dec`가 비어 있으면 그 프레임의 DIA 그림·점 없음, 상태에 이유), `ra`/`dec` 열이 없는
  옛 요약이면 첫 행을 쓴다. 값은 dia 요약에서 가져온다.

### target_photometry
```python
target_photometry(catalogs, ra, dec, matching_radius_arcsec=5.0, out=None, plot=False, verbose=True, dedupe=False)
    -> astropy.table.Table | None
```
- `catalogs`: `CatalogSet` 또는 `Catalog` 목록. 반경 안 가장 가까운 소스의 카탈로그별 한 행. 맞는 소스가 없으면 `None`(CSV 없음).
- 열: 메타데이터(`filter`, `exptime`, `obsdate`, `mjd`, `jd`, `seeing`, `depth`, `observatory`, `telname`, `zp_err` 중 있는 것) + 카탈로그에 있는
  `MAGSKY_*`/`MAGERR_*`/`ZPERR_*`/`UL5SKY_*`(`APER_3`, `AUTO`, `APER`, `APER_1`, `APER_2`, `APER_4`) + (표에 `path`가 있으면) `SATDIST`(카탈로그에 열이 없으면 NaN).
- `out=` 기본 `~/ezphot/log/recipes/<첫 카탈로그 objname>/target_photometry_<stamp>.csv`, 끝에 쓴다. progress 파일 없음.
- CLI의 DataBrowser 검색(`--paths` 없이, `load_catalogs`)은 파생 접두어(`coadd_`·`sub_` 등)와 forced 카탈로그 `*.circ.cat`을 버린다.
  `--pattern`(`pattern=`)이 `.circ.cat`으로 끝나면 forced 카탈로그를 버리지 않는다(파생 접두어는 여전히 버림).
  `--paths`나 `catalogs`로 직접 준 카탈로그는 그대로 쓴다.

### qa
```python
qa(summary, zperr_max=0.1, nref_min=10, seeing_max=5.0, depth_min=None, color_max=0.2) -> pandas.DataFrame
```
- `summary`: process_images/stack 요약(CSV 경로 또는 DataFrame). 열 `chk_success`, `chk_zperr`, `chk_nref`, `chk_seeing`, (`depth_min`이면) `chk_depth`,
  (`color_max`가 None이 아니면) `chk_color`, 그리고 `qa_pass`를 붙여 돌려준다. 파일을 쓰지 않는다. 색 항 값이 없으면 `chk_color`는 통과다.

## 3. 구멍 번호의 뜻

- 개구 목록은 `aperture_diameter_arcsec`(지름 ″, 기본 `[3, 5, 7, 10, 15]`) 순서대로, 그 뒤에 `aperture_diameter_seeing`(기본 `[2.5, 3.5]`) × seeing.
  기본값은 `defaults.PHOTOMETRY`(SExtractor `.cat`)와 `defaults.FORCED_PHOTOMETRY`(forced `.circ.cat`)에 있다.
- 열 접미사는 그 목록의 인덱스다. **인덱스 0은 접미사가 없고**(`MAG_APER`), k ≥ 1은 `_k`(`MAG_APER_1` …). 기본 목록에서
  `APER` = 3″, `APER_1` = 5″, `APER_2` = 7″, **`APER_3` = 10″**(`defaults.APER10_KEY`), `APER_4` = 15″, `APER_5` = 2.5 × seeing, `APER_6` = 3.5 × seeing.
- seeing 배수 구멍은 영상마다 크기가 다르다. SExtractor `.cat`: process_images 2차 패스가 돌았으면 1차 패스 ZP 기준성의 seeing, 2차 패스가 없으면
  SExtractor가 영상에서 추정한 seeing. forced: 영상 헤더 `SEEING`(2차 패스가 돌았으면 2차 photcal 값). forced에서 헤더에 seeing이 없으면
  seeing 배수 구멍이 빠진다(그 카탈로그는 `APER_4`까지).
- 그래서 `header` 방식 forced(dia 포함)의 `MAGSKY_APER_5`/`_6`은 SExtractor 구멍에서 맞춘 헤더 `ZP_APER_5`/`_6`을 크기가 다른 forced 구멍에 붙인 값이다.
  구멍끼리 비교하거나 값을 맞출 때는 고정 지름 구멍 `APER`(0) … `APER_4`를 쓴다.
- `config`로 구멍 목록의 순서나 개수를 바꾸면 같은 이름이 다른 구멍을 가리키고, 열 이름만으로는 알 수 없다. 기본 목록과 `APER10_KEY`를 바꾸는 것은 계약 변경이다.
- 구멍마다의 열: `MAG_APER_k`·`MAGERR_APER_k`·`FLUX_APER_k`·`FLUXERR_APER_k`·`NPIX_APER_k`(기기 등급, ZP 전), `ZP_APER_k`·`ZPERR_APER_k`,
  `MAGSKY_APER_k`(= MAG + ZP), `UL3SKY_APER_k`·`UL5SKY_APER_k`(3σ·5σ 한계 등급). forced 카탈로그에는 `SKYSIG_APER_k`(구멍 안 배경 잡음 ADU),
  `UL3_APER_k`·`UL5_APER_k`(ZP 전 한계 등급)도 있고, `UL5SKY_APER_k` = `UL5_APER_k` + ZP다(두 `calibrate` 방식 모두).
- forced 오차·한계 등급의 정의(ADU, 구멍 k마다). 기본(고리 없음, 또는 고리를 플럭스에만 쓰는 차영상):
  - `SKYSIG_APER_k` = r_k × √(Σ_구멍 bkgrms²), `UL5_APER_k` = −2.5 log10(5 `SKYSIG_APER_k`), `UL3_APER_k`는 3σ.
  - `FLUXERR_APER_k`² = r_k² Σ bkgrms² + Σ max(영상, 0)/EGAIN(구멍 안 소스 Poisson, `calc_total_error`) + H_k, `MAGERR_APER_k` = 1.0857 `FLUXERR`/`FLUX`.
  - r_k = 1, H_k = 0이 기본이다. 차영상(`forced_photometry_sub`)에서는:
    - r_k(`NOISE_RATIO_APER_k`): 같은 영상의 무작위 빈 구멍 최대 N = 500개에서 타겟과 같은 방식(구멍 합 − 고리 평균 × 면적)으로 잰 값의 실측 산포
      (4σ 클립 표준편차) ÷ 그 위치의 √(Σ bkgrms²). 위치: 고정 시드, 가장 큰 구멍과 고리가 영상 안, 가장 큰 구멍 안에 NaN·무효(|값| < 1e-20: 0과
      hotpants의 무효 채움값 1e-30)·마스크 픽셀 없음. `conv_`가 convolve된 reference이면(`host_source='conv'`) 그런 위치 중 처음 2N개에서
      `conv_`의 가장 큰 구멍 플럭스(그 위치의 `conv_` 고리 평균 × 면적을 뺀 값)가 그 위치들의 중앙값보다 예측 잡음 3배 넘게 밝은 곳(템플릿 소스)을 뺀다.
      상관 잡음(재투영·convolve)과 고리 평균의 잡음을 담는다. 쓸 수 있는 위치가 max(N/4, 10)개(기본 125) 미만이면 NaN이고 r_k = 1로 둔다.
      r_k의 표본 오차는 약 3 %다(N = 500; T08635 r_3 1.117/1.214/1.376, 지난 QA의 독립 추정과 차이 sd 0.053 = 두 추정의 표본 잡음 수준).
    - 한계: `conv_`가 없거나 convolve된 science(`CONVOL00=IMAGE`)이면, 또 `host_poisson=False`이면(`conv_`를 열지 않으므로) 빈 구멍의 소스 회피를
      하지 않고 무효 픽셀 회피와 4σ 클립에만 기댄다.
      붐비는 필드에서는 차영상의 별 잔차가 섞여 r_k가 몇 % 클 수 있다(그만큼 `UL5SKY`가 얕고 `MAGERR`가 크다).
    - H_k: `conv_<name>`(convolve된 reference, 차영상과 같은 격자·science 플럭스 단위)의 구멍 합에서 그 영상 자신의 고리 평균 × 면적을 뺀 값
      (`HOST_FLUX_APER_k`)을 0에서 자르고 EGAIN으로 나눈 것(호스트 Poisson). 유한하지 않으면 0이다(`HOST_FLUX_APER_k` NaN, `host_source='not finite'`).
      `FLUXERR`·`MAGERR`에만 들어가고 `SKYSIG`·한계 등급에는 들어가지 않는다. 호스트 위 타겟에서는 몫이 크다(T08635에서 `FLUXERR` 분산의
      0.07/0.31/0.43, 최소/중앙/최대). 그래서 `host_source`가 `conv`가 아닌 행은 오차가 작게 나오고, 한 광도곡선에 섞으면 오차가 일관되지 않는다.
  - 고리를 오차에도 쓰는 경우(`annulus_errors=True`, 차영상 밖에서 `annulus_width_arcsec`를 준 호출)는 이전 정의 그대로다: `SKYSIG` = 고리 std × √면적,
    `FLUXERR`² = 위 bkgrms 식(r=1, H=0) + (면적 × 고리 std / √고리 픽셀 수)².

## 4. forced 카탈로그(`*.circ.cat`) 읽기 계약

| 항목 | 내용 |
| --- | --- |
| 파일 | `<img>.fits` → `<img>.fits.circ.cat`, **원본 FITS와 같은 폴더**(심볼릭 링크면 따라간 실제 파일의 폴더). 차영상은 `sub_<name>.fits.circ.cat`. 같은 영상에 다시 재면 덮어쓴다(dia와 forced_photometry가 같은 차영상을 재면 마지막 것, 2절 forced_photometry의 `visualize --dia` 주의) |
| 형식 | 헤더 줄이 있는 ASCII 표(`Catalog.write()` 기본, astropy `format='ascii'`). 옆의 `.info`(JSON)는 있으면 쓰고 없어도 된다 |
| 컬럼 | `X_WORLD`, `Y_WORLD`(deg, 요청한 좌표, 이름 고정), `MAGSKY_APER_3`, `MAGERR_APER_3`, `ZPERR_APER_3`, `UL5SKY_APER_3`. 그 밖의 열은 계약 밖 |
| 픽셀 좌표 | `X_IMAGE`/`Y_IMAGE`는 0-based(WCS origin 0). SExtractor `.cat`은 1-based |
| info | `obsdate`(isot), `filter`, `observatory`, `telname` — `.circ.cat.info` 또는 옆 FITS 헤더. `Catalog(path=...)`는 같은 폴더나 한 단계 위에서 FITS를 찾아 채운다 |
| 행 | 요청한 좌표마다 한 행. `reference_stars` 카탈로그에는 기준성 행도 있다. **행 순서는 계약이 아니다** — `X_WORLD`/`Y_WORLD`로 좌표 매칭해서 읽는다 |
| 값 | 99 같은 센티널 없음, 재지 못한 값은 NaN. 구멍이 영상과 전혀 겹치지 않거나 구멍 안에 NaN 픽셀(무효 영역)이 있으면 NaN, 영상 가장자리에 걸치면 겹친 부분만 합해 실제보다 어둡다(요약은 성공, 위치가 영상 안인지는 표시하지 않음). `MAGERR_APER_3`에 ZP 오차는 들어 있지 않다(`ZPERR_APER_3`를 따로 더한다). 차영상은 국소 고리 배경을 뺀 값이고(`BACKGROUND` 열), `MAGERR`는 3절의 r_k와 호스트 항을, `UL5SKY`는 r_k만 담는다(`NOISE_RATIO_APER_k`, `HOST_FLUX_APER_k` 열) |
| ZP | `header`·dia: 그 영상 헤더의 `ZP_APER_k`(차영상은 science 영상에서 이어진 차영상 헤더). `reference_stars`: 그 카탈로그의 기준성에서 구한 ZP(헤더 ZP와 다를 수 있다). 카탈로그(`.circ.cat`·`.info`)에는 어느 방식의 ZP인지 기록이 없다 — 같은 영상에 두 방식을 섞어 돌리지 않거나 요약 CSV(`calibrate` 열)를 함께 보관한다. 이전 `reference_stars`의 `.circ.refcat`이 남아 있어도 지금 `.circ.cat`이 그 방식이라는 뜻은 아니다 |
| 보정 실패 | forced_photometry가 쓴 `.circ.cat`에 `ZP_APER_3` 열이 없으면 보정에 실패한 중간 산출물이다(그 실행 요약은 `success=False`; 이전의 정상 파일을 이미 덮어썼을 수 있다). dia도 같다: 차영상 헤더에 ZP가 없으면 `sub_*.circ.cat`은 ZP 열 없이 남고 dia 요약은 `success=False`, `step='apply_zp'`다 |

## 5. `calibrate='reference_stars'`의 한계

- 기준성 위치는 `<img>.fits.refcat`(process_images/stack의 마지막 photcal이 쓴 ZP 기준성)의 `X_WORLD`/`Y_WORLD`다. 없으면 그 영상은 `step='refcat'`으로 실패한다.
- forced 카탈로그에는 `FWHM_WORLD`·`CLASS_STAR`·`ELONGATION`·`FLAGS`·`IMAFLAGS_ISO`·`FLUX_MAX`·`ISOAREA_IMAGE`가 없어서 photcal의 FWHM·별 판별·
  타원율·플래그 컷과 SATDIST(밝은 별 헤일로) 컷이 건너뛰어진다.
- 고립 컷은 잰 위치(기준성 + 타겟)끼리만 본다. 실제 이웃 소스에 대한 고립은 보장되지 않는다.
- 타겟이 참조 카탈로그의 별과 맞고 등급 범위 안이면 타겟 자신이 ZP 기준성에 들어갈 수 있다. 타겟 근처(고립 반경 안)의 기준성은 빠진다.
- 기준성 선택 등급은 `MAG_APER_3`(10″)다. ZP는 구멍마다 따로 구하므로 이 키는 기준성 선택에만 영향을 준다.
  등급 범위는 `defaults.MAG_RANGE`의 필터별 값(process_images의 `mag_faint_shift`는 없음).
- 영역 컷은 하지 않는다(`ra_deg=dec_deg=None`, `inner_fraction=1.0`). refcat 별은 이미 process_images/stack의 영역 컷(그때의 타겟 중심 `radius_arcsec`,
  타겟이 없었으면 `inner_fraction`)을 거친 집합이고, 이렇게 해야 기준성이 넘긴 타겟의 위치·순서와 무관하다(고립 컷과 타겟 자기 포함은 예외).
- 기준성이 `min_number_of_sources`보다 적어도 등급 범위를 넓혀 가며 진행한다. 0개면 그 영상은 `step='photometric_calibration'`으로 실패하고,
  1개면 `ZPERR_APER_3`가 0이다. 요약 `n_ref`를 확인한다.
- 영상 헤더와 FITS는 바꾸지 않는다(`update_header=False`, `save_image=False`). 같은 영상에 ZP가 둘(헤더 / forced 기준성)이 되므로 요약 `calibrate`로 구분한다.

## 6. 이 계약을 바꿀 때

- 계약에 드는 것: 위 함수의 이름·인자(이름·순서·기본값)·반환 형태, 요약 열의 이름과 뜻, `out=` 기본 규칙, 실패 표현, 종료 코드,
  기본 구멍 목록과 `APER10_KEY`, forced 카탈로그의 파일 이름·위치·형식·4절 컬럼과 info 필드, 단위·센티널, 같은 입력에서의 측광 값.
- 바꾸는 변경은 `CHANGELOG.md`의 `## Unreleased`에 항목을 쓰고 분류(호환 추가 / 수치 변경 / 파괴적)와 **사용처 영향**(읽는 쪽이 무엇을 확인·수정해야 하는지)을 적는다.
  수치 변경은 어느 열이 얼마나 바뀌는지 적는다. 이 문서도 같은 변경에서 고친다.
- 버전(0.x 동안, `ezphot/__init__.py`의 `__version__`): 호환 추가(새 인자는 기본값 = 기존 동작, 새 열·새 레시피)와 수치 변경은 patch,
  파괴적 변경(이름·인자 뜻·열 이름이나 뜻·구멍 순서·파일 이름이나 위치 변경)은 minor이고, 가능하면 한 버전 동안 옛 이름을 `DeprecationWarning`과 함께 남긴다.
