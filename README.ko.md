# Prism-LOB

호가창(limit order book)의 toxic-flow·liquidity-vacuum 위험 대리 라벨을 예측하고, 예측에 기여한 가격·수량 분기와 시간 구간을 분석하는 연구 코드입니다.

> Jihong Park, Eunho Koo. *Prism-LOB: Structured Evidence and Perturbation Audits for Limit Order Book Risk Prediction.*

Prism은 Positional Readout over Interpretable Semantic Modules의 약자입니다. 이 공개판의 Prism-LOB은 **positional readout**을 사용합니다. 100개 시점 × 40개 특징을 매도 가격·매도 수량·매수 가격·매수 수량의 네 분기로 나누고, TCN 특징을 학습된 시간 가중치로 집계한 뒤 분류합니다. DeepLOB·TLOB·BiN-CTABL 비교 모델은 별도 구현입니다.

[English](README.md) · [설치·실행](docs/REPRODUCE.md) · [데이터](docs/DATA.md) · [그림 7개](figures/README.md) · [현재 한계](docs/KNOWN_ISSUES.md)

## 공개된 자료

- 모델·라벨·분할·학습·평가 지표·귀속·개입 코드와 positional 학습 설정.
- 선택된 PNG **7개**: 기존 최종 그림 6개와 정답·오답 사례 비교 갤러리 1개.
- 테스트, 의존성 목록, 현재 사용법과 재현 범위 설명.

기존 학습 가중치, 실행 로그, 실험 결과 CSV·NPZ·상세 JSON, 결과 압축파일과 과거 설정·정리 문서는 이 공개판에 포함하지 않습니다. 원본 데이터는 재배포하지 않으며 이번 정리에서 수정하지 않았습니다. 완성 그림의 열람과 원 수치의 재집계는 다릅니다. [공개 범위](docs/PUBLICATION_SCOPE.md)

## 먼저 실행하기

Python 3.10 이상을 사용하며 저장소 루트에서 실행합니다.

```sh
python -m venv .venv
```

PowerShell: `.\.venv\Scripts\Activate.ps1` / Bash: `source .venv/bin/activate`.

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/check_docs.py
python scripts/smoke_pooling.py --pool positional
python -m lob_xai_viz.plot_factorization --demo --task toxic --target 1 --out-dir figure_revision_output/demo_factorization
```

위 점검과 데모에는 FI-2010 데이터나 기존 학습 가중치가 필요하지 않습니다. 데모는 합성 입력과 학습 전 모델을 사용하므로 논문 결과가 아닙니다.

입력 CSV를 준비한 뒤에는 다음처럼 새 모델을 학습합니다.

```sh
python scripts/train_microstructure.py "@configs/paper/toxic.args" --seed 1 --name paper_toxic_seed1
python scripts/train_microstructure.py "@configs/paper/vacuum.args" --seed 1 --name paper_vacuum_seed1
```

Seed 2·3은 이름도 바꿔 실행합니다. 기본 출력은 `runs/NAME/`이며 `--output-dir`으로 바꿀 수 있습니다. 기존 출력 폴더는 재사용하지 않습니다. 설정 파일은 positional/identity와 validation 기반 모델 선택을 사용합니다. 입력 파일, 새 개입 실험과 그림 명령은 [실행 안내](docs/REPRODUCE.md)에 있습니다.

## 데이터와 예측 과제

FI-2010은 Ntakaris 등의 공개 벤치마크입니다. [원 논문](https://arxiv.org/abs/1705.03233) · [배포 기록](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)

| 과제 | 미래 구간 | 라벨의 의미 |
|---|---|---|
| Toxic flow | 10 step | 최우선 매도호가 상승·매수호가 하락에 따른 stationary / toxic up / toxic down |
| Liquidity vacuum | 30 step | 최우선 수량 감소와 spread proxy 증가에 따른 normal / vacuum |

이 라벨은 위험의 대리 지표입니다. 코드는 FI-2010의 native 방향 라벨 대신 [연구용 위험 라벨](lob_xai/microstructure_labels.py)을 계산합니다. 원 배포본을 이 코드의 CSV로 변환하는 정확한 이력은 아직 불완전합니다. [입력 형식과 현재 한계](docs/DATA.md)

## 코드 읽는 순서

1. [실험 상수](lob_xai/paper_protocol.py) → [위험 라벨](lob_xai/microstructure_labels.py) → [입력 데이터](lob_xai/data.py)
2. [모델](lob_xai/models.py) → [학습](lob_xai/training.py) → [평가 지표](lob_xai/metrics.py)
3. [귀속](lob_xai/gradcam.py) → [시간 구간 대체·복원](lob_xai/counterfactual_test.py)
4. [실행 스크립트](scripts/README.md) → [그림 도구](lob_xai_viz/README.md)

현재 공개판은 저장된 논문 수치를 자동으로 복원하는 패키지가 아닙니다. 표·그림별 지원 범위는 [논문 대응표](docs/PAPER_MAP.md), 해결되지 않은 출처·수치 문제는 [현재 한계](docs/KNOWN_ISSUES.md)에 있습니다.

## 인용과 이용

저자·제목·저장소는 [CITATION.cff](CITATION.cff)에 기록되어 있습니다. 출판 DOI와 오픈소스 라이선스는 아직 선언하지 않았습니다. FI-2010의 이용 조건은 원 배포처를 따릅니다.
