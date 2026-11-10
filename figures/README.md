# 공개 그림 7개

기존 최종 PNG 6개와 정답·오답 사례 비교 1개를 보존합니다. 파일 내용은 원본과 동일하며, 새 실험으로 다시 생성한 결과가 아닙니다.

| 내용 | PNG |
|---|---|
| Toxic 귀속 분해 | [factorization_toxic_seed1.png](factorization/factorization_toxic_seed1.png) |
| Vacuum 귀속 분해 | [factorization_vacuum_seed1.png](factorization/factorization_vacuum_seed1.png) |
| 시간 구간 대체·복원 곡선 | [cf_mirror.png](cf_mirror/cf_mirror.png) |
| Toxic-up 대표 경보 | [alert_card_toxic_toxic_up_seed1_rank1.png](alert_card/alert_card_toxic_toxic_up_seed1_rank1.png) |
| Vacuum 대표 경보 | [alert_card_vacuum_vacuum_seed1_rank1.png](alert_card/alert_card_vacuum_vacuum_seed1_rank1.png) |
| Seed별 정답 경보 | [fig11_alert_typicality.png](qual_gallery/fig11_alert_typicality.png) |
| 정답·오답 사례 비교 | [fig00_qual_grounding_positional.png](qual_gallery/fig00_qual_grounding_positional.png) |

마지막 그림은 클래스별 높은 신뢰도의 정답·오답 사례를 비교합니다. 선택된 사례이므로 전체 예측 성능이나 일반적인 사례를 대표하지 않습니다.

CF mirror의 deletion은 신뢰도 층화 정답 경보, insertion은 고신뢰 정답 경보입니다. 같은 표본군의 paired 결과로 해석하지 않습니다.

이 폴더에는 PNG와 목록만 둡니다. 결과 CSV·NPZ·상세 JSON·압축파일·이전 그림은 공개하지 않습니다. 그림은 수치 재집계 입력을 대신하지 않습니다. [그림 도구](../lob_xai_viz/README.md) · [현재 한계](../docs/KNOWN_ISSUES.md)
