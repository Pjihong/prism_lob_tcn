# 논문 표·그림과 코드

아래 번호와 label은 2026-10-21 수정안과 코드의 연결을 위한 식별자입니다. 미공개 원고는 저장소에서 제공하지 않으며 공개 후 링크와 최종 번호를 갱신할 예정입니다. [모델 해시 기록](../configs/paper/checkpoints.json)은 파일 식별 정보이며 학습 가중치는 포함하지 않습니다.

새 평가·귀속·개입에는 직접 학습하거나 별도로 공급한 checkpoint가 필요합니다. `figures/`는 논문용 PNG만 제공하는 폴더입니다. 과거 표 집계·곡선 입력 파일은 제공하지 않으므로 아래 표는 **구현 위치와 새 실행에 필요한 입력**을 안내합니다. PNG에서 원래 수치를 복원하거나 모든 표를 자동 재현할 수 있다는 뜻은 아닙니다.

## 표와 분석 코드

| 원고 / label | 내용 | 구현·실행 경로 | 필요한 입력과 현재 범위 |
|---|---|---|---|
| 표 1 `tab:detection` | 탐지 성능 | [고정 checkpoint 평가](REPRODUCE.md#5-고정-checkpoint-평가) → `scripts/aggregate_final_test.py` | 데이터·checkpoint로 평가 JSON을 새로 생성해야 함. Vacuum 수치와 baseline run 선택은 미확정 |
| 표 2 `tab:cf_main` | 고신뢰 삭제·삽입 전환·회복률 | [개입 실험](REPRODUCE.md#6-고신뢰-경보와-분기-분석), `lob_xai/counterfactual_test.py` | 데이터·checkpoint가 필요하며 과거 요약 CSV는 미포함 |
| 표 3 `tab:cf_appendix` | 고신뢰 ΔAUC와 층화 삭제 전환율 | 위 개입 실험, `scripts/run_cf_response_carpet.py`, `scripts/cf_effect_stats.py` | 두 표본군의 분석 결과를 각각 생성해야 함. 완성 TeX 표 자동 생성기는 없음 |
| 표 4 `tab:tuned_hparams` | 학습 설정 | [논문 설정](../configs/paper/README.md), `scripts/train_baseline_microstructure.py` | 설정과 checkpoint metadata를 대조. TLOB optimizer 예외는 별도 |
| 표 5 `tab:cf_figure_samples` | CF 그림의 방향별 표본 수 | `lob_xai_viz/render.py --only cf` | 기존 CF 원자료를 별도로 공급해야 재집계 가능. 원고의 수치만으로 새 실험의 표본 수를 정하지 않음 |
| 표 6 `tab:branch_importance` | Toxic-up 분기 비중·대체 효과 | `scripts/probe_branch_structure.py`, `scripts/analyze_branch_importance.py` | 데이터·checkpoint로 signed CSV를 새로 생성해야 함. 관련 그림 폴더를 제거해도 표의 분석 코드는 유지 |

`aggregate_final_test.py`는 한 입력 폴더에 두 과제의 seed 1–3 평가 JSON을 요구합니다. 서로 다른 모델이나 설정의 결과를 섞지 말고, 최종 checkpoint와 수치의 일치 여부를 먼저 확인하세요.

## 그림과 PNG

| 원고 / label | 내용 | 현재 PNG | 새로 그리는 방법과 필요한 입력 |
|---|---|---|---|
| 그림 1 `fig:architecture` | 모델 구조 | 원본 자산 미포함 | 원고 참조 `model_arch.png`의 원본 필요. 모델 구현은 `lob_xai/models.py` |
| 그림 2 `fig:factorization` | Toxic 귀속 분해 | [PNG](../figures/factorization/factorization_toxic_seed1.png) | `--only factorization`: FI-2010 CSV와 지정 checkpoint |
| 그림 3 `fig:cf` | 시간 구간 대체·복원 | [PNG](../figures/cf_mirror/cf_mirror.png) | `--only cf`: 별도로 준비한 기존 CF NPZ와 입력 manifest의 해시 일치 필요 |
| 그림 4 `fig:alert_vacuum` | Vacuum 대표 경보 | [PNG](../figures/alert_card/alert_card_vacuum_vacuum_seed1_rank1.png) | `--only cards`: 데이터와 vacuum seed 1 checkpoint, 대표 창 7654 |
| 그림 5 `fig:alert_toxic` | Toxic-up 대표 경보 | [PNG](../figures/alert_card/alert_card_toxic_toxic_up_seed1_rank1.png) | `--only cards`: 데이터와 toxic seed 1 checkpoint, 대표 창 20152 |
| 그림 6 `fig:alert_typicality` | Seed별 대표 경보 | [PNG](../figures/qual_gallery/fig11_alert_typicality.png) | `--only gallery`: 데이터와 toxic seed 1–3 checkpoint |
| 그림 7 `fig:factorization_vacuum` | Vacuum 귀속 분해 | [PNG](../figures/factorization/factorization_vacuum_seed1.png) | `--only factorization`: 데이터와 vacuum seed 1 checkpoint |

위 옵션은 `python -m lob_xai_viz.render`에 사용합니다. 새 실행 결과는 `figure_revision_output/`에 저장됩니다. 입력이 없는 CF 실행은 중단됩니다. [그림 생성 안내](../lob_xai_viz/README.md)를 참고하세요.

알고리즘 1 `alg:cf`의 구현은 `lob_xai/counterfactual_test.py`입니다. CF 삭제와 삽입의 표본군은 다르며, 분기 대체의 로짓 변화는 부호를 유지합니다.

수치·출처의 불일치는 그림 폴더 정리로 해결되지 않습니다. 현재 확인이 필요한 항목은 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)에 유지합니다.
