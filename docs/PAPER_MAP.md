# 논문 표·그림과 현재 공개 코드

그림 번호는 이전 수정안의 식별자이며 최종 원고 공개 후 확정할 수 있습니다. 미공개 원고와 상세 수치 자료는 배포하지 않습니다.

| 내용 | 현재 코드와 범위 |
|---|---|
| 탐지 모델·지표 | `lob_xai/models.py`, `training.py`, `metrics.py`; 사용자가 새로 학습·평가한 결과를 계산 |
| 탐지 결과 집계 | `scripts/aggregate_final_test.py`; 별도로 준비한 평가 JSON 필요. 공통 고정 checkpoint 평가 CLI는 미제공 |
| 시간 구간 대체·복원 | `lob_xai/counterfactual_test.py`, `scripts/run_cf_response_carpet.py`; 데이터와 호환 모델 필요 |
| CF 결과 통계 | `scripts/cf_effect_stats.py`; 사용자가 준비한 수치 자료 필요 |
| 분기 분석 | `lob_xai/branch_analysis.py`의 공통 계산은 유지. 새 분기 probe·중요도 계산 CLI는 미제공 |
| 학습 설정 | [positional 설정](../configs/paper/README.md), 비교 모델 학습 코드 |
| 그림 | [공개 PNG 7개](../figures/README.md)와 [독립 그림 도구](../lob_xai_viz/README.md) |

## 공개 그림

| 내용 | 파일 |
|---|---|
| Toxic 귀속 분해 | [PNG](../figures/factorization/factorization_toxic_seed1.png) |
| Vacuum 귀속 분해 | [PNG](../figures/factorization/factorization_vacuum_seed1.png) |
| CF 대체·복원 곡선 | [PNG](../figures/cf_mirror/cf_mirror.png) |
| Toxic 대표 경보 | [PNG](../figures/alert_card/alert_card_toxic_toxic_up_seed1_rank1.png) |
| Vacuum 대표 경보 | [PNG](../figures/alert_card/alert_card_vacuum_vacuum_seed1_rank1.png) |
| Seed별 정답 경보 | [PNG](../figures/qual_gallery/fig11_alert_typicality.png) |
| 정답·오답 사례 비교 | [PNG](../figures/qual_gallery/fig00_qual_grounding_positional.png) |

CF deletion과 insertion은 다른 표본군이며, 갤러리는 선택된 사례입니다. 상세 결과를 비공개 보관한 사실이 출처·수치의 기존 불일치를 해결하지는 않습니다. [현재 한계](KNOWN_ISSUES.md)
