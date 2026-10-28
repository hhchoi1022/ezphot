# SN 2026kid (NGC 5907) — LOAO 처리 현황 인수인계 (2026-08-31)

작업 규칙: `ezphot/analysis/` 안에서만 작업. 라이브러리 패키지는 읽기 전용.
실행 환경: conda `py311` (`/home/hhchoi1022/anaconda3/envs/py311/bin/python`) —
base anaconda는 numpy/erfa가 깨져 있어 사용 금지. `MPLCONFIGDIR=/tmp/ezphot-matplotlib` 권장.

## 데이터 위치

- 원본(읽기 전용): `/qso/data6/obsdata/LOAO/`
- 산출물: `/home/hhchoi1022/ezphot/data/` (→ `/lyman/data1/factory_hhchoi/data` 심링크)
  - 과학 이미지·스택: `scidata/LOAO/{LOAO_E2V_2x2,LOAO_iKon_1x1}/SN2026kid/1.0-m KASI/<필터>/`
  - 최종 측광: `scidata/LOAO/SN2026kid/lightcurve/`
  - reference: `scidata/LOAO/SN2026kid/reference_images/`
  - stage report: `scidata/LOAO/SN2026kid/stage_reports/`
- QC (그림 1장 + 로그 1개 per stage): `/home/hhchoi1022/ezphot/log/20260806_stage_run_qc/` (3~14번)

## 완료된 파이프라인 (71밤, 2026-04-25 ~ 07-14)

0~2. 마스터프레임·전처리·astrometry.net 측성 — 기존 260805 스크립트 (1,020장)
3~9. 마스킹→배경→RMS→측광→APASS 캘리브레이션→스택→스택측광 —
     `260806_stage_photometry_loao_sn2026kid.py` (stage별 `--stage` 실행, 밤별 report)
     - 유효 1,012장 (10장은 `rejected_frames.json`: 별 없는 구름 프레임)
     - 스택 283개 (NZPSTAR<5인 4장은 스택 입력에서 제외)
10. Light curve — `260805_plot_*` (10″) + `260808_*_apertures.py` (3/5/10″)
    + `260808_forced_photometry_*.py` (스택 forced)

## DIA (완료)

- Reference: PS1 HiPS2FITS g/r/i, 스택 그리드 규격 — `260810_query_reference_ngc5907.py`
  매핑: B102→g, V103→g, R104→r, I105→i (진짜 BVRI 서베이 없음; LOAO 폭발 전 관측 없음)
- 차감: `260811_dia_stage_b_subtract.py` — astroalign(검증 게이트) + WCS재투영 fallback,
  HOTPANTS bgo=2 nrx/nry=4. 280/283 성공 (`sub_stack.*.fits`), 3개 별-없음 제외
- 측광: `260811_dia_stage_c_photometry.py` — **갱신 SN 좌표 RA 228.9881590 Dec +56.3090884**
  (차감 이미지 centroid 139개 중앙값, 발견좌표에서 0.85″) 고정,
  {DIA,직접}×{annulus 10″,없음} 4변형 × 3/5/10″, 스택 aperture별 ZP 보정
- 최종 품질 컷: `260811_quality_filtered_lightcurves.py` — ZP 자동컷 폐기,
  **육안 검토로 4개 밤(2026_0531, 0610, 0616, 0626)을 전 필터 제외** → 264/280
- **주 결과: `lightcurve/`의 `_qcut` 파일들, `dia_ann` (DIA+annulus) 5″ aperture**

## 핵심 결정 사항 (되돌리지 말 것)

- SCAMP 미사용 — GAIA/APASS-FILE 기준 모두 테스트했고 TAN 재적합(`fit_wcs_from_points`,
  APASS)이 2~3배 우수 (`260811_test_scamp_single_frames.py`, QC 12_scamp_test)
- GaiaXP 미사용, APASS DR9 기준 (R/I는 r/i 변환)
- 배경은 마스터가 아닌 각 과학 이미지에서 SEP 직접 추정 (host 타원 마스크 포함)
- LC는 filter offset 비활성화, fixed_width magnitude table

## 알려진 이슈/한계

- 단일 프레임 astrometry.net 해에 0.3~1.6″ 오프셋 (APASS 기준) — 스택별 TAN 재적합으로
  대응 완료. 전면 재측성은 불필요 판정
- B-band DIA는 g template 색 불일치 계통오차 가능 — SN 소멸 후 LOAO B template 촬영 권장
- 라이브러리 버그 (CODEBASE_GUIDE.md 기록): SExtractor 공백 경로(`1.0-m KASI`) 문제
  → 스크립트들이 /tmp 스테이징으로 우회, iKon 픽셀스케일 레지스트리 오류(0.794→0.356 패치),
  `ImageQuerier.query()`의 stack_swarp 호출 시그니처 불일치
- `260805_photometry_*.py`는 3개 스크립트가 importlib으로 로드하는 공유 모듈 — 이름 변경 금지

## Claude 메모리 (로컬, 계정 무관)

`~/.claude/projects/-lyman-data1-factory-hhchoi-code-ezphot/memory/`
- `analysis-folder-only.md` — 작업 범위 규칙
- `loao-sn2026kid-pipeline.md` — 파이프라인 상태·결정 이력 (이 문서와 동일 내용)
