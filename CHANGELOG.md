# Changelog

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
