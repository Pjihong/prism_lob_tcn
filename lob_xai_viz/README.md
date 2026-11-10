# 독립 그림 도구

완성된 공개 그림은 [figures/](../figures/README.md)의 PNG 7개입니다. 이 패키지는 사용자가 준비한 데이터·모델·결과로 새 그림을 만듭니다. 기존 상세 reference JSON과 원 그림의 수치 동일성 자동 대조 기능은 공개판에 포함하지 않습니다.

| 모듈 | 필요한 입력 |
|---|---|
| `plot_factorization` | positional 모델·데이터, 또는 합성 `--demo` |
| `plot_alert_card` | positional 모델·데이터, 또는 합성 `--demo` |
| `gallery_qualitative` | 명시한 positional 모델·데이터 |
| `plot_positional_readout` | 명시한 positional 모델 |
| `plot_cf_mirror` | 별도로 준비한 CF 곡선 입력 |
| `plot_branch_importance` | 별도로 준비한 분기 CSV·CAM 시간 NPZ 캐시 |

```sh
python -m lob_xai_viz.plot_factorization --help
python -m lob_xai_viz.plot_factorization --demo --task toxic --target 1 --out-dir figure_revision_output/demo_factorization
python -m lob_xai_viz.plot_alert_card --demo --task toxic --target 1 --out-dir figure_revision_output/demo_card
```

데모는 합성 입력과 학습 전 모델이며 논문 결과가 아닙니다. 실제 모델을 쓰는 명령은 `--help`에 따라 입력·출력 경로를 지정하세요. 새 출력은 `figure_revision_output/` 등 로컬 경로에 두고 기존 PNG를 덮어쓰지 마세요.

Positional 가중치의 균등 기준선은 각 시점에 `1/T`를 주는 수학적 비교 기준입니다. 모델의 실행 설정은 positional입니다. 새 그림·수치 파일을 생성할 수 있다는 사실이 기존 결과 파일을 배포한다는 뜻은 아닙니다.

CF 그림의 seed별 min–max 음영은 신뢰구간이 아니며, deletion과 insertion은 동일 표본군이 아닙니다. 선택된 정성 사례도 일반적인 성능을 대표하지 않습니다. [해석 한계](../docs/KNOWN_ISSUES.md)
