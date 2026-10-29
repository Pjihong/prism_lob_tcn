> **이전 대응표:** 현재 표·그림 안내는 [PAPER_MAP.md](../PAPER_MAP.md)를 따른다.

# 논문 수정안과 네 PDF의 주장·구현·증거 대응

## 현재 활성 코드의 기준: 제공된 논문 수정안

원문은 `manuscripts/reference/revision_20261021.txt`에 그대로 보존했다.
원고 안의 `IMPLEMENTATION CHECK` 주석은 미확정 사항의 기록으로 읽었으며,
이번 정리에서 실험이나 수식을 변경하라는 지시로 실행하지 않았다.

| 수정안의 근거 | 남긴 구현과 결과 경로 |
|---|---|
| §3.1·§4.1: toxic k10 / vacuum k30, T100/F40, identity 입력, 고정 분할 | `lob_xai/data.py`, `microstructure_labels.py`, `paper_protocol.py`; 라벨·분할 감사와 테스트 |
| §3.2·§4.2·부록 A: 네 TCN 분기, positional, 공통 학습과 TLOB 예외 | `lob_xai/models.py`, `training.py`, `metrics.py`; `models/`의 세 비교 모델; 두 trainer |
| §3.2·§5.3·부록 C: signed HiResCAM–Grad-CAM 항등식 | `lob_xai/gradcam.py`; `prism_figure_revision/scripts/plot_factorization.py` |
| §3.3·§5.4·부록 B: 고신뢰 삭제·삽입 및 별도 신뢰도 층화 삭제 | `lob_xai/counterfactual_test.py`, `scripts/common/run_cf_response_carpet.py`, `cf_effect_stats.py`; `figures/cf_*` |
| §3.3·§5.2·부록 C: 학습 평균 특징 대체, signed drop-one, Kendall τ | `scripts/common/probe_branch_structure.py`, `axis_a_min.py`, `branch_xai_common.py`; `figures/axis_a_positional/` |
| §5.5: 경보 카드와 seed별 사례, 정답/오답 보완 비교 | `prism_figure_revision/scripts/plot_alert_card.py`, `gallery_qualitative.py` 및 reference provenance |
| 부록 A: positional 선택 실패 이력과 multi-scale 항등식 적용 범위 | GAP·multi-scale 구현, pooling 실험 도구, 기존 checkpoint·manifest·실패 결과 |
| §5.6·결론: 한계와 추가 대조 | 현재 결과를 보존하며 fixed-recent 등 계획을 완료 결과로 바꾸지 않음 |

VAR/WABL, 반합성 위치 찾기, Shapley 분해, 과거 그림 1–14 생성기와 일회성
복구/학습 래퍼는 이 수정안의 활성 실행 경로에서 제외했다.
기존 네 PDF가 언급하는 자료는 아래 표와 원본 ZIP의 파일별 출처로 추적한다.
CAM sanity·hash·라벨·분할 검사는 실행의 정확성을 확인하는 도구로 유지한다.

## 기존 네 PDF와 보존 자료

대상은 `manuscripts/reference/`의 네 원본이다.
`lob_tcn 1.pdf`는 9쪽(인쇄 쪽수 20–28), `lob_tcn 2.pdf`는 13쪽(20–32)의
이미지 발췌본으로 전 페이지를 렌더링해 확인했다.
`Prism_LOB_TCN1.pdf`는 35쪽, `Prism_LOB_TCN2.pdf`는 33쪽 전체 원고다.
파일 이름만으로 어느 버전이 최종 승인본인지 결정하지 않는다.

아래 '증거'는 제공된 산출물의 위치다. 모든 논문 숫자를 독립적으로 재현했다는
뜻이 아니다. 정리 범위와 보존 방법은 `CLEANUP.md`를 참고한다.

| 주장/계약 | PDF의 내용 | 코드와 보존 자료 |
|---|---|---|
| 위험 proxy·시간순 분할 | 전체 원고 §3–4; toxic k10, vacuum k30; T100/F40; 단일 시간순 분할, 별도 test | `lob_xai/microstructure_labels.py`, `data.py`, `paper_protocol.py`; `data/FI2010/`; label/split 테스트 |
| 네 의미 branch·positional readout | §3.2; ask/bid × price/volume; 입력 독립 softmax 시간 가중 | `lob_xai/branching.py`, `models.py`, `input_norm.py`; pooling 테스트 |
| 탐지 비교 | TCN2 Table 1; 이미지1 Table1, 이미지2 Table2 | `models/`, `scripts/train*_microstructure.py`, `evaluate_ours_checkpoint.py`, `aggregate_final_test.py`; `loggers/results/` |
| 공통 positional 선택 | TCN2 §4.2/Appendix B; vacuum ΔMCC +.052, toxic +.009로 gate 미달 | GAP/multi_scale와 과거 manifest 유지; `frozen/paper_positional_checkpoints.json`은 그림 출처 positional 모델 |
| signed CAM 인수분해 | 전체 원고 §3.2/5.3; 이미지1 p22–24, 이미지2 p23–25 | `lob_xai/gradcam.py`, `models.py`; `prism_figure_revision/scripts/plot_factorization.py`; 수정본 provenance |
| α 해석 | seed1 잔량 recency/가격 준균등; α만으로 표본별 saliency가 결정되지 않음 | `plot_positional_readout.py`, `inspect_pool_weights.py`, 관련 `figures/`; GAP-era와 최종 결과 구별 |
| Deletion/insertion | TCN2 §3.3/4.4/5.2; 이미지 발췌 CF 그림·표 | `lob_xai/counterfactual_test.py`; `figures/cf_final_positional/`, `cf_response_carpet/`, `cf_mirror_src_positional/`; 수정본 input manifest |
| CF 설계 | train-only pool30000, K50/M10, rho .05–.50; 비중첩 곡선; insertion rho0 무조건부 donor | CF 구현, `paper_protocol.py`, `evaluation_protocol.py`; random/anti 및 오류 검사 유지 |
| 두 cohort | TCN2 §4.3/Appendix C; top-confidence와 stratified 구분 | `select_top_confidence_alerts`, `run_cf_response_carpet.py`, `cf_effect_stats.py`; 창별 CSV/NPZ |
| Branch drop-one·관찰 수렴 | toxic-up ask-volume Δlogit13.41±6.96, share .613±.036; Kendall .730/.649/.655 | `axis_a_min.py`, `branch_xai_common.py`, `plot_branch_importance.py`; branch·Axis 결과 |
| W_abl 결합·VAR | 특히 이미지2 p26–27; 잔량 간 약 .95, VAR 비정상성으로 관찰 구조 강등 | 결과·PDF 보존: `figures/wabl_network/` 등. 제거한 `axis_b_branch_structure.py`·`plot_wabl_network.py` 원본은 `lob-xai.zip`에 보존 |
| 반합성 국소화 | 과거 연구/보조 진단; 현재 결과로 혼동하지 않음 | 제공 결과 보존. 제거한 `axis_c_synthetic_localization.py`는 원본 ZIP에서 회수 가능 |
| 경보 카드 | vacuum #7654, toxic-up #20152; CAM [76,100], random [3,27] | `prism_figure_revision/scripts/plot_alert_card.py`, 수정본 reference provenance, 원본·통합 원고 그림 |
| 정성·seed 갤러리 | 고신뢰 정답/오답; seed별 rank1; 사례와 집계 구분 | `prism_figure_revision/scripts/gallery_qualitative.py`, `caption_stats.json` |
| CAM sanity | 입력 섭동 안정성과 randomization 이후 남는 상관 한계 | `cam_sanity_checks.py`, `figures/cam_sanity*`; 약점도 유지 |
| 한계 | 단일 FI-2010, 중첩 창, purge0, vacuum val 양성103, seed SD≠모집단 CI, test 개발 이력, 모델 개입≠시장 인과 | 원본 PDF·통합 TeX·기존 감사 문서 |
| 미실행 대조 | TCN2 §5.6/결론; fixed-last span, 입력 그룹 사후 귀속 비교, 독립 자료 | 통합 원고 `Control_Experiments.md`, `planned_controls.tex`는 **계획**으로 보존 |

`scripts/common/`에 있는 파일은 표에서 파일명만 적은 경우가 있다.

## 버전 차이와 유지할 해석

- **Vacuum MCC:** 이미지 `lob_tcn 1.pdf` Table1과 TCN2는 `.239`;
  이미지 `lob_tcn 2.pdf` Table2와 TCN1 abstract는 `.229`.
  그림 수정 패키지는 `.209/.229/.239` 불일치도 기존 문제로 기록한다.
  `.209`는 해당 문서의 언급이지 네 PDF 모두에서 확인한 수치가 아니다.
  저장된 run 선택·집계 근거 확정 전 원고 숫자를 임의 수정하지 않는다.
  이번에 `figures/final_test/vacuum_seed{1,2,3}.npz`의 저장된 정답·예측으로 MCC를
  다시 계산한 결과는 `.1701306421/.2199203643/.2382969506`,
  평균 `.2094493190`, 모집단 SD `.0287969153`이다.
  같은 폴더의 JSON 및 `summary.csv`와 일치한다. 원고의 `.229/.239`는 이 제공 결과로
  뒷받침되지 않는다. 새 checkpoint 추론을 수행한 검증은 아니며 원고를 자동 수정하지 않았다.
- **선택 이력:** 기존 `EXPERIMENT_STATUS.md`는 toxic=multi_scale/revin_c와
  vacuum=positional/identity를 기록한다. PDF의 공통 positional/identity와 다르다.
  기존 `positional_6ckpt_manifest.json`도 초기 poolab 모델이다.
  새 manifest는 9월 그림 패키지의 checkpoint hash를 그대로 옮겼다.
- **Cohort:** toxic-up seed3 deletion173/insertion174 차이를 수정 패키지가 기록한다.
  임의로 표본을 버리거나 양 방향이 같은 cohort라고 표기하지 않는다.
- **수학 범위:** signed 항등식과 표시용 절댓값·정규화 map을 구별한다.
  선형 readout에서 평균과 집약이 교환된다고 최종 비선형 head 로짓까지
  branch 기여의 단순 합으로 분해되는 것은 아니다.
- **W_abl/VAR:** 개입 결합과 VAR 관찰 모델 실패를 모두 보존하며 시장 인과로 승격하지 않는다.
- **변경된 서사:** GAP-era의 'Prism-LOB 분산형/DeepLOB 응집형' 구별이 최종 positional에서는
  유지되지 않았다는 PDF의 한계를 유지한다.
- **추가 대조:** fixed-recent 대비 CAM 선택의 이득, 입력 그룹 사후 귀속과의 비교가
  아직 확인되지 않았다는 TCN2의 경계를 유지한다.

이 표는 원문 전체를 대신하지 않는다. PDF·TeX·결과 원본을 남기고 주요 실증·수학·실패·한계
주장을 구현 경로에 연결한다.
