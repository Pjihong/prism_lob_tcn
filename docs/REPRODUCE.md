# 설치와 새 실험 실행

저장소 루트에서 실행합니다. 이 공개판의 Prism-LOB readout은 positional입니다. 아래 명령은 현재 코드 실행법이며 과거 논문 수치의 재현 완료 선언이 아닙니다.

## 1. 데이터 없는 점검

```sh
python -m venv .venv
```

PowerShell: `.\.venv\Scripts\Activate.ps1` / Bash: `source .venv/bin/activate`.

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/check_docs.py
python scripts/smoke_pooling.py --pool positional
python scripts/verify_baseline_models.py --device cpu
python -m lob_xai_viz.plot_factorization --demo --task toxic --target 1 --out-dir figure_revision_output/demo_factorization
python -m lob_xai_viz.plot_alert_card --demo --task toxic --target 1 --out-dir figure_revision_output/demo_card
```

데모는 합성 입력과 학습 전 모델을 사용합니다. 생성 그림은 논문 결과가 아닙니다. `constraints-validated.txt`는 이전 코드 점검의 의존성 버전 목록이며 원 학습 환경 lockfile이 아닙니다.

## 2. 데이터와 출력 경로

[DATA.md](DATA.md)의 두 CSV를 로컬에 준비합니다. 원 배포본에서 기대 CSV까지의 정확한 변환 이력은 불완전합니다. 원본 데이터는 공개 저장소와 이번 정리 대상에 포함하지 않습니다.

`runs/`, `validation_output/`, `figure_revision_output/`은 새 실행의 로컬 출력 경로이며 Git에서 제외합니다. 별도로 지정하는 출력 경로도 공개 전에 확인하세요. 완성된 공개 그림 7개를 새 실행으로 덮어쓰지 마세요.

## 3. Positional 학습

```sh
python scripts/train_microstructure.py "@configs/paper/toxic.args" --seed 1 --name paper_toxic_seed1
python scripts/train_microstructure.py "@configs/paper/vacuum.args" --seed 1 --name paper_vacuum_seed1
```

Seed 2·3은 이름과 함께 변경합니다. 기본 출력은 `runs/NAME/`, 별도 경로는 `--output-dir`입니다. 기존 출력 폴더는 재사용하지 않습니다.

두 설정은 type4 TCN, positional/identity, T100, toxic horizon10 또는 vacuum horizon30, inverse-frequency weighted CE, AdamW, cosine schedule, validation macro-F1 기반 선택을 사용합니다. `--dev-only --skip-report`는 test 추론과 보고서 생성을 학습에서 분리합니다. YAML 실행 로그는 생성하지 않지만 모델·예측·요약 등 새 실험 산출물은 로컬에 저장합니다.

새 checkpoint에는 `config.pool="positional"`이 기록됩니다. Prism-LOB을 읽는 독립 도구는 이 설정을 명시한 호환 checkpoint가 필요합니다. `--help`에서 실제 입력 인자를 확인하세요.

## 4. 새 개입 실험

독립 CF 구현은 유지합니다. 아래 자리표시자를 실제 파일 경로로 바꿉니다.

```sh
python -m lob_xai.counterfactual_test --model ours --ckpt PATH_TO_MODEL.pt --task toxic --seed 1 --direction both --device cpu --out validation_output/toxic_cf.png
```

입력 데이터와 모델을 별도로 준비해야 합니다. 고신뢰 정답 경보와 신뢰도 층화 정답 경보는 서로 다른 표본군입니다. 새로운 결과를 과거 논문 수치와 같다고 자동으로 간주하지 않습니다.

## 5. 그림 도구와 저장된 결과 집계

현재 공개 PNG는 [그림 목록](../figures/README.md)에서 바로 열람할 수 있습니다. 새 그림은 `python -m lob_xai_viz.<모듈> --help`로 입력을 확인해 실행합니다. [그림 도구](../lob_xai_viz/README.md)

`aggregate_final_test.py`, `collect_branch_dev_results.py`, `cf_effect_stats.py`는 사용자가 별도로 준비한 결과 파일을 집계합니다. 기존 수치·reference JSON은 배포하지 않으므로, 공개판만으로 과거 표·곡선을 재집계하거나 과거 그림과의 수치 동일성을 자동 검증할 수는 없습니다.

공통 checkpoint 평가·분기 분석 명령은 현재 제공하지 않습니다. 남아 있는 코드와 미해결 과학적 문제는 [논문 대응표](PAPER_MAP.md)와 [현재 한계](KNOWN_ISSUES.md)를 따릅니다.
