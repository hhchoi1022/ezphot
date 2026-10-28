# AllBRICQS 퀘이사로 7DT 투과곡선 검증 — 작업 인수인계 문서

작성 2026-09-10 18:05 KST (Claude 세션 e9ba5c7a). **다른 Claude 세션이 이어서 작업하기 위한 기록.**
규칙: `ezphot` 라이브러리·설정·카탈로그 아카이브는 절대 수정하지 않는다. 새 파일은 `ezphot/analysis/` 아래에만.
실행 환경: `/home/hhchoi1022/anaconda3/envs/py311/bin/python`, `MPLCONFIGDIR=/tmp/ezphot-matplotlib`. **CPU 코어는 16개까지만 사용(사용자 지시).**

## 1. 목적과 배경

- 2026-09-08~10에 GaiaXP 합성측광 기준 카탈로그(GAIAXP_CORR_LAMOST)와 7DT 중대역 측광의 색 의존성을 조사해 **유효 투과곡선 37개**를 추정했다
  (`ezphot/analysis/effective_curves/`, 보고서 `ezphot/analysis/260908_zpcolor_report.md`, 웹 페이지
  https://claude.ai/code/artifact/eeb48091-c195-4d17-829c-270024710bb0, 로그 `/home/hhchoi1022/ezphot/log/20260910_effective_curves/`).
- M형 별로 검증하려 했으나 XP 합성등급이 M 별에서 밴드별 0.05–0.17 mag 틀려(TiO 띠머리 번짐) 검증 불가
  (`/home/hhchoi1022/ezphot/log/20260910_xp_mdwarfs/`, 그림 F5–F9). CALSPEC M3.5V 표준성 BD-11 3759만 XP 없이 검증 가능: rms 0.018 → 0.014.
- **새 타겟: AllBRICQS 밝은 퀘이사** (Onken+2023 PASA 40 e010 남반구 156개; Choi+2025 ApJS 280 73 북반구 62개). 적색이동에 따라 넓은 방출선이 밴드 가장자리에 걸리므로
  곡선 이동에 민감. 7DT 원본 프레임이 있는 퀘이사 115개(중대역 15개 필터 이상 113개)를 GW 포털에서 찾았고, 그 서베이 타일들을 **단일 프레임 측광까지만** ezphot으로 처리 중.

## 2. 현재 상태 (2026-09-10 18:25) — 배치 1 이미지 처리 완료, 그 이후 작업은 시작하지 않음

**배치 1 완료** (`260910_allbricqs_process.py --tiles T05785 T10796 T11579 T06894 T01609 T14710 --n-proc 16`, 17:36–18:24, 로그 `/home/hhchoi1022/ezphot/log/20260910_allbricqs/process/batch1.raw.log`, 요약 `process/summary_*.csv`):

| 타일 | 퀘이사 | 프레임 | 성공 | 실패 | 비고 |
|---|---|---|---|---|---|
| T05785 | J0529-4351 (z 3.93) | 60 | 57 | 3 | 7DT02/m475 3장 astrometry 실패(영점 없음); 7DT02/m450은 ZPERR 0.23 불량 |
| T10796 | J0220-2519 (z 2.17) | 63 | 63 | 0 | ZPERR 중앙값 0.015 |
| T11579 | J0400-2257 (z 0.16) | 63 | 63 | 0 | |
| T06894 | J1252-3928 (z 0.23) | 66 | 66 | 0 | |
| T01609 | J2331-6642 (z 0.27) | 60 | 60 | 0 | 관측 필터에 m825/m850 없음 → 가장자리 감도 없음(참고용) |
| T14710 | J2242-1316 (z 0.27) | 60 | 60 | 0 | |

- 산출물: `/home/hhchoi1022/ezphot/data/scidata/7DT/7DT_C361K_HIGH_1x1/<타일>/7DT??/<필터>/*.fits` + 사이드카(`.cat`, `.cat.info`, `.refcat`, `.status`; 영점은 헤더 `ZP_APER_3`/`ZPERR_APER_3`, 색항 `K_COLOR_*`).
- 실행 중인 프로세스 없음. **배치 2는 큐에서 취소함**(사용자 지시: 처리 후 다음 작업 자동 시작 금지). 필요하면 수동 실행:
  `MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 python ezphot/analysis/260910_allbricqs_process.py --tiles T08969 T05197 T12487 T12488 T00602 T13622 --n-proc 16`
  (관측 필터 기준 감도 순위 다음 6개, 약 444장, 16워커로 약 75분).
- 후속 분석(3절)은 **시작하지 않았다**. `collect/`에는 T05785의 harvest 결과만 있다(전체 수집 시 덮어씀).

## 3. 다음 단계 (수동 시작)

1. 배치 1 완료 확인 후 수집: `python ezphot/analysis/260910_allbricqs_collect.py --stage all --tiles T05785 T10796 T11579 T06894 T01609 T14710 --n-proc 8`
   - harvest → `collect/frames.csv`(T05785는 이미 있음, 전체 다시 돌리면 덮어씀), refit → `refit_frames.csv`, `refit_stars.csv.gz`(기존 `260908_zpcolor_collect.stage_refit` 재사용, GAIAXP·CORR_LAMOST 두 카탈로그), extract → `tilestar_spectra.npz`(참조 별 XP flux_cor).
   - 출력 디렉터리 `/home/hhchoi1022/ezphot/log/20260910_allbricqs/collect/`.
2. 퀘이사 잔차: `python ezphot/analysis/260910_allbricqs_photometry.py --kind new --stage all --n-proc 16`
   - `--kind new`는 `collect/` 디렉터리를 프레임 표·재적합·참조 별 스펙트럼 소스로 쓴다(`--frames-dir`로 바꿀 수 있음).
   - 하는 일: 퀘이사를 각 프레임 `.cat`에서 1.5″ 매칭 → 관측 등급(MAG_APER_3 + 헤더 영점) vs XP 합성등급(공칭 곡선 / 유효 곡선 + 참조 별로 재유도한 영점) → 연속 밴드(|dm/dnm| < 0.005)로 선형 변광 보정 → 선 밴드 잔차. 그림 `figures/Q_<퀘이사>_new.png`, 표 `qso_resid_new.csv`.
   - 검토 포인트: 선이 걸린 밴드 쌍(예 J0400-2257의 m750/m775, J0220-2519의 m400)에서 공칭 vs 유효 곡선 잔차 크기와 부호가 템플릿 감도(`catalog/allbricqs_edge_sensitivity.csv`의 dm_per_nm)와 맞는지.
3. 결과가 유의하면 배치 2 처리 → 같은 순서. 2밤 이상 관측된 퀘이사(J1544-2016 T12487/T12488, J0715-4951 T04464 7밤 414장, J1547-1449, J1040-3324)는 변광·재현성 검사용.
4. XP 분해능 문제(4절) 검증: 북반구 62개의 공개 스펙트럼(VizieR J/ApJS/280/73 table6, `catalog/allbricqs_north_J_ApJS_280_73_table6.csv`, 파장 Å·플럭스 erg/s/cm²/Å)과 XP(`/home/hhchoi1022/ezphot/log/20260910_xp_mdwarfs/allbricqs_xp_spectra.npz`, 143개)의 밴드별 합성등급 비교 — 아직 안 함.

## 4. 알려진 문제·주의

- **XP 분해능**: XP LSF(10–15 nm)가 넓은 방출선(FWHM 5–20 nm)을 번지게 하므로 가장자리 밴드의 선 플럭스 배분이 틀릴 수 있음. M 별에서 이것이 치명적이었음. 단일 가우시안 선은 LSF를 넣은 순방향 모델로 보정 가능.
- **변광**: Gaia 변광 플래그 218개 중 192개. XP 시기(2014–17) ≠ 7DT 시기. 연속 밴드 선형 보정으로 흡수. 타일 관측은 한 밤에 전 필터 동시(유닛별)이므로 밴드 간 시차 없음.
- **포털 좌표 질의는 프레임 중심 기준**: `GWPortalConnector('raw').query(ra=, dec=, radius=)`는 중심이 반경 안인 프레임만 반환. 반경 1.0° + `vertices` 폴리곤 검사로 실제 시야 포함 여부를 판정해야 함(스크립트 scratchpad `portal_qso_cone.py` 로직은 `portal/qso_raw_frames.csv.gz`에 반영됨).
- 2024–25년 타일은 격자 필터 17–20개뿐(새 필터 m438/m483/…/m586/m615 없음). 감도 순위는 **관측된 필터 기준** `portal/qso_coverage_ranked_observed.csv`를 써야 함. J0529-4351(z 3.93)은 Lyα 밴드가 없어 C IV의 m750/m775만 남음; J2331-6642(T01609)는 m825/m850이 없어 감도 없음.
- J2320-0524(FEIGE110 필드, Mg II가 m650/m675 경계)는 Gaia DR3에 XP가 없음. J0154-5559(T03245 코애드)는 코애드 측광이 인접 밴드끼리 0.1–0.2 mag 튐(원인 미조사, 등급오차도 비정상적으로 큼).
- `260910_allbricqs_photometry.py`의 검출 컷: FLAGS ≤ 3, FLUX_MAX < 45000, MAGERR < 0.2. 필요하면 조정.
- pandas 함정: `df.pivot`, `df.flags`, `df.filter`, `df.corr`는 메서드 → 반드시 `df["..."]`.
- 유효 곡선 파일: `ezphot/analysis/effective_curves/<필터>.csv` (`lam`, `trans`), 메타 `curves_metadata.csv`. `260910_xp_mdwarfs_photometry.load_effective(f)`로 읽음.
- 배치 처리 중 pkill -f 패턴이 자기 셸까지 죽인 사고가 있었음. 프로세스 종료는 PID로.

## 5. 파일 목록

| 경로 | 내용 |
|---|---|
| `/home/hhchoi1022/ezphot/log/20260910_allbricqs/catalog/` | `allbricqs_all.csv`(218개: name, source_id, ra, dec, BP, RP, W1W2, z, hemi), `allbricqs2023_tableA1.csv`, `allbricqs2023.pdf`, 북반구 VizieR 표 2개, `allbricqs_edge_sensitivity.csv`(퀘이사×37밴드 dm/dnm, 템플릿 기반), `allbricqs_edge_summary.csv` |
| `.../portal/` | `qso_raw_coverage.csv`(218개 커버리지), `qso_raw_frames.csv.gz`(퀘이사가 시야 안에 든 원본 프레임 11,127장: filepath, filter, unit, night, exptime, object_name), `qso_coverage_ranked.csv`, `qso_coverage_ranked_observed.csv`(**이걸 써라**), `calspec_raw_frames.csv.gz`(CALSPEC 관측 2024-08/09만) |
| `.../process/` | 처리 로그(`batch1.raw.log`, `smoke_T05785.log`), 요약 CSV, 취소된 `waiter_batch2.sh` |
| `.../collect/` | 수집 산출물(현재 T05785 harvest만) |
| `.../figures/`, `qso_phot_coadd.csv.gz`, `qso_resid_coadd.csv` | J0154-5559 코애드 미니 테스트 |
| `/home/hhchoi1022/ezphot/log/20260910_xp_mdwarfs/allbricqs_xp_spectra.npz` | 퀘이사 143개 XP(raw, cor, err, bp_rp, g, tile) |
| `ezphot/analysis/260910_allbricqs_process.py` | 단일 프레임 처리(AlertProcessor; DIA·강제측광·스태킹 끔, photcal radius None) |
| `ezphot/analysis/260910_allbricqs_collect.py` | harvest/refit/extract |
| `ezphot/analysis/260910_allbricqs_photometry.py` | 퀘이사 잔차(`--kind single|coadd|new`, `--frames-dir`) |
| `ezphot/analysis/260910_xp_mdwarfs*.py`, `260910_xp_calspec_fidelity.py` | M 별 검사·XP 충실도(참고) |
| `ezphot/analysis/260723_proces_images_for_Young.py`, `260722_process_images_for_YIdo.py` | 사용자 원본 처리 예제(CALSPEC·타일) |

## 6. 처리 파이프라인 요약 (AlertProcessor, BRIDGE 패키지 `/lyman/data1/factory_hhchoi/code/BRIDGE`)

`GWPortalConnector('raw').query(tile_name=...)` → 파일 경로 → `AlertProcessor.load_images_from_path` → `config.max_workers=16`, `single_process['do_DIA']=False`, `['do_forced_photometry']=False`, `photcal['radius_arcsec']=None` →
`pipeline_before_stacking(Alert(objname=tile, ra, dec))` = 전처리(bias/dark/flat) → astrometry → 마스크 → 배경 → SExtractor 측광 → 영점(GAIAXP_CORR_LAMOST, MAG_AUTO 기준, 등급 범위 필터별 `mag_range_dict`, 100 s 노출이면 기본값 그대로).
프레임당 약 10 s(16병렬), 타일(60장)당 6–11분. 실패 프레임은 헤더에 ZP 없음·`.status`의 ASTROMETRY false.
