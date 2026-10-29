# 논문 그림

연구 결과 PNG 6개를 한곳에 모았습니다. 그림 번호는 2026-10-21 수정안 기준이며, 최종 원고 공개 후 번호와 링크를 갱신할 예정입니다. 구조도는 현재 배포하지 않습니다.

| 원고 | 그림 | PNG |
|---|---|---|
| 그림 2 | Toxic 귀속 분해 | [factorization_toxic_seed1.png](factorization/factorization_toxic_seed1.png) |
| 그림 3 | 시간 구간 대체·복원 | [cf_mirror.png](cf_mirror/cf_mirror.png) |
| 그림 4 | Vacuum 대표 경보 | [alert_card_vacuum_vacuum_seed1_rank1.png](alert_card/alert_card_vacuum_vacuum_seed1_rank1.png) |
| 그림 5 | Toxic-up 대표 경보 | [alert_card_toxic_toxic_up_seed1_rank1.png](alert_card/alert_card_toxic_toxic_up_seed1_rank1.png) |
| 그림 6 | Seed별 대표 경보 | [fig11_alert_typicality.png](qual_gallery/fig11_alert_typicality.png) |
| 그림 7 | Vacuum 귀속 분해 | [factorization_vacuum_seed1.png](factorization/factorization_vacuum_seed1.png) |

```text
figures/
  alert_card/      # 대표 경보 2개
  cf_mirror/       # 삭제·삽입 곡선 1개
  factorization/   # 귀속 분해 2개
  qual_gallery/    # seed별 대표 경보 1개
  README.md
```

경보 카드는 논문 checkpoint에 대응하는 positional 결과를 사용합니다. 갤러리는 모델 이름을 Prism-LOB으로 표시한 버전입니다. CF 그림의 삭제는 신뢰도 층화 정답 경보, 삽입은 고신뢰 정답 경보를 사용하므로 동일 표본군의 두 결과로 해석하지 않습니다.

이 폴더에는 PDF 사본, 중간 CSV·JSON·NPZ·로그, 이전 모델의 그림을 포함하지 않습니다. PNG는 열람용 결과이며 수치 재집계의 입력을 대신하지 않습니다. 새 평가·분석 명령과 필요한 입력은 [실행·재현 안내](../docs/REPRODUCE.md), 표·그림과 코드의 연결은 [논문–코드 대응표](../docs/PAPER_MAP.md)를 참고하세요.

CF PNG는 최종 수정 패키지의 그림과 SHA-256이 동일합니다. 그림 생성 코드는 [lob_xai_viz](../lob_xai_viz/README.md)에 통합했습니다. 완성 그림은 이 폴더에서 관리하며 그림 생성 코드와 분리합니다.
