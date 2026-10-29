# 그림 생성 코드

완성 그림은 [figures/](../figures/README.md), 그림을 만드는 코드는 이 패키지에서 관리합니다.
`prism_figure_revision`의 최종 수정 렌더러를 이곳으로 합쳤습니다.
공통 모델·데이터 적재는 `lob_xai.checkpoints`, 분기 개입은 `lob_xai.branch_analysis`에 있습니다.

## 논문에 사용한 입력으로 다시 그리기

저장소 루트에서 실행합니다. 기존 가중치와 CF 원자료는 배포하지 않으므로 별도로 공급해야 합니다.

| 명령 | 입력 | 담당 모듈 |
|---|---|---|
| `python -m lob_xai_viz.render --only factorization` | FI-2010 CSV, 지정 checkpoint | `plot_factorization.py` |
| `python -m lob_xai_viz.render --only cards` | 같은 데이터와 checkpoint | `plot_alert_card.py` |
| `python -m lob_xai_viz.render --only gallery` | 같은 데이터와 toxic 3 seed checkpoint | `gallery_qualitative.py` |
| `python -m lob_xai_viz.render --only cf` | 기존 CF NPZ 원자료 | `plot_cf_mirror.py` |

기본 장치는 CPU이며 `--device`로 바꿀 수 있습니다. 실행기는 `reference/input_manifest.json`의
입력 SHA-256과 checkpoint를 검사합니다. `reference/`의 나머지 작은 JSON은 카드 확률·대표 창·
귀속 항등식 등의 대조값입니다. 그림과 중복된 부가 파일이 아니라 내보내기 전 검증에 실제로 사용됩니다.
불일치하면 내보내기를 중단합니다. `--only all`에는 CF 원자료도 필요합니다.

새 출력은 Git에서 제외한 `figure_revision_output/<실행>/` 아래에 생성됩니다.
`fig/`에는 PNG/PDF, `notes/`에는 수치·검사 자료, `work/`에는 중간 자료가 생깁니다.
배포할 PNG는 검토 후 `figures/`에 반영하며 실행기가 기존 논문 PNG를 자동 교체하지 않습니다.

## 새 모델의 그림 / 데이터 없는 데모

각 렌더러는 `python -m lob_xai_viz.<모듈> --help`로 인자를 확인할 수 있습니다.
직접 실행할 때는 checkpoint와 출력 경로를 명시하세요. 새 모델의 그림은 기존 논문 결과와 구분합니다.
데이터 없는 다음 예시는 합성 입력과 학습 전 모델을 사용하며 논문 결과가 아닙니다.

```sh
python -m lob_xai_viz.plot_factorization --demo --task toxic --target 1 --out-dir figure_revision_output/demo_factorization
python -m lob_xai_viz.plot_alert_card --demo --task toxic --target 1 --out-dir figure_revision_output/demo_card
```

`plot_branch_importance.py`와 `plot_positional_readout.py`는 추가 분석 그림용 도구입니다.
전자는 분기 분석 CSV·checkpoint, 후자는 checkpoint의 readout 가중치를 읽습니다.
새 실험 실행 순서는 [scripts 안내](../scripts/README.md)와 [REPRODUCE.md](../docs/REPRODUCE.md)에 있습니다.

## 그림과 캡션을 읽을 때

- 귀속 분해: positional 가중치, signed Grad-CAM, 직접 계산한 HiResCAM과 재구성값을 비교합니다.
- 경보 카드: 입력, 분기별 |CAM|, CAM 구간 대체 결과입니다. random 확률은 대조 결과입니다.
- 갤러리: 색은 창 안에서 정규화한 |CAM|이며, 표시 tick은 1–100, 통계의 `peak_t`는 0–99입니다.
- CF: 선은 seed별 평균의 동일 가중 평균이고 음영은 seed 평균의 min–max이며 신뢰구간이 아닙니다.
  삭제는 신뢰도 층화 정답 경보, 삽입은 고신뢰 정답 경보입니다. Toxic-up seed 3의 표본 수는
  각각 173/174이므로 같은 paired cohort로 표시하지 않습니다.

논문에는 선택한 PNG를 넣고 최종 크기로 확인하세요. 원고를 자동 변경하는 과거 TeX 패치는 제거했습니다.
수치·출처의 미해결 사항은 [KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md)에 있습니다.
