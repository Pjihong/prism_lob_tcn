# 실행과 재현

모든 명령은 저장소 루트에서 실행한다. PowerShell과 Bash 모두 Python 명령은 동일하다. 본문은 실행 인터페이스이며 모든 학습 실험의 재현 완료 선언이 아니다.

## 1. 설치와 데이터 없는 점검

```sh
python -m venv .venv
```

PowerShell: `.\.venv\Scripts\Activate.ps1`. Bash: `source .venv/bin/activate`.

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/check_docs.py
python scripts/smoke_pooling.py --pool positional
python scripts/verify_baseline_models.py --device cpu
python scripts/verify_preservation.py
```

위 점검은 기존 학습 가중치 없이 실행할 수 있다. 기존 run과 checkpoint 파일은 배포하지 않으므로 학습된 모델의 적재·평가에는 별도로 준비한 파일이 필요하다. 보존 검사는 남겨 둔 파일의 해시를 확인하며 논문 성능을 검증하는 명령이 아니다.

Windows에서 검증에 사용한 환경은 Python 3.12, CPU PyTorch 2.6.0이다. 재현용 의존성 제약은 `constraints-validated.txt`에 있다. 이는 원래 학습 환경의 lockfile이 아니다.

```sh
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . -c constraints-validated.txt
```

## 2. 저장된 CF 그림

완성된 [CF PNG](../figures/cf_mirror/cf_mirror.png)와 [논문 그림 목록](../figures/README.md)은 데이터 없이 열람할 수 있습니다.

`figures/`의 CSV·JSON·NPZ 중간 자료를 제거했으므로 현재 checkout만으로 과거 CF 곡선을 재집계할 수는 없습니다. `python -m lob_xai_viz.render --only cf --device cpu`는 기존 원자료를 별도로 준비한 경우에만 실행합니다. 입력 경로와 SHA-256은 `lob_xai_viz/reference/input_manifest.json`에 기록되어 있으며 불일치하거나 누락되면 중단합니다.

기존 그림 열람과 새로운 개입 실험은 구분합니다. 새 CF 분석은 아래 §6에서 데이터·checkpoint를 지정해 실행하며, 그 결과가 과거 논문 수치와 같다고 자동으로 간주하지 않습니다.

## 3. 데이터 준비

[DATA.md](DATA.md)의 형식과 SHA-256을 확인한다. 아래 단계부터는 실제 FI-2010 CSV가 필요하다. 현재 정확한 원 배포본→동결 CSV 변환 절차는 미확정이다.

```sh
python scripts/verify_preservation.py --include-data
```

## 4. 논문용 학습 설정

```sh
python scripts/train_microstructure.py "@configs/paper/toxic.args" --seed 1 --name paper_toxic_seed1
python scripts/train_microstructure.py "@configs/paper/vacuum.args" --seed 1 --name paper_vacuum_seed1
```

새 학습의 출력은 `loggers/`에 생성되며 `.gitignore`로 Git 추적에서 제외한다. 원고와 기존 학습 결과는 현재 배포하지 않는다.

`@` 파일은 표준 argparse의 인자 파일이며 한 줄에 한 인자를 저장한다. 따옴표는 PowerShell의 `@` 해석을 피하기 위해 양쪽 셸에서 사용한다. Seed 2와 3도 별도 실행한다.

설정: type4 TCN, channels 32/32/64, positional, identity, T100, toxic k10/vacuum k30, inverse-frequency weighted CE, AdamW 1e-3/weight decay 1e-5, cosine, batch256, 최대100 epoch, patience10, validation macro-F1 선택. `--dev-only --skip-report`로 test 평가와 보고서 생성을 학습에서 분리한다.

뒤에 명시한 인자는 preset을 덮어쓴다. 다른 설정으로 실행했다면 논문 preset과 같은 실험이라고 보고하지 않는다. 과거 CLI와 Config의 기본 GAP는 호환성을 위해 유지한다. TLOB은 원고에 별도로 기술한 optimizer 예외가 있으므로 이 ours preset을 TLOB에 적용하지 않는다.

## 5. 고정 checkpoint 평가

먼저 §4의 학습을 실행하거나 호환 checkpoint를 준비한다. 아래 `PATH_TO_TOXIC_SEED1_MODEL.pt`는 예시 자리표시자이며 실제 모델 파일 경로로 바꿔야 한다. 저장소에는 기존 학습 가중치가 포함되어 있지 않다.

```sh
python scripts/evaluate_ours_checkpoint.py --checkpoint PATH_TO_TOXIC_SEED1_MODEL.pt --test-csv data/FI2010/FI2010_test.csv --confirm-final-test --device cpu --out-json validation_output/toxic_seed1.json --out-npz validation_output/toxic_seed1.npz
```

`--confirm-final-test`는 실제 test 추론 실행을 명시한다. 과거 결과 파일을 덮어쓰지 않도록 새 출력 경로를 쓴다. Vacuum을 포함한 논문 표의 출처 불일치는 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)에 남긴다.

## 6. 고신뢰 경보와 분기 분석

아래 명령은 PowerShell/Bash 공통이며 실제 추론을 수행한다. `PATH_TO_TOXIC_SEED1_MODEL.pt`를 준비한 toxic seed 1 모델의 실제 경로로 바꾼다.

```sh
python -m lob_xai.counterfactual_test --model ours --ckpt PATH_TO_TOXIC_SEED1_MODEL.pt --task toxic --seed 1 --direction both --device cpu --out validation_output/toxic_seed1_cf.png
python scripts/probe_branch_structure.py --ckpt PATH_TO_TOXIC_SEED1_MODEL.pt --task toxic --k 10 --seed 1 --target 1 --config-out validation_output/branch_config.json
python scripts/analyze_branch_importance.py --ckpt PATH_TO_TOXIC_SEED1_MODEL.pt --task toxic --k 10 --seed 1 --target 1 --config validation_output/branch_config.json --baseline mean --out-dir validation_output/branch_audit
```

`analyze_branch_importance.py`는 과거 `axis_a_min.py`의 이름을 역할에 맞게 바꾼 실행 파일이다. 기존 `axisA_min_*` 출력 이름은 분석 코드와의 연결을 위해 유지한다. 현재 원고의 학습 평균 특징 대체·signed drop-one·Kendall 비교에 대응한다. 고신뢰 CF 결과를 신뢰도 층화 결과와 합치지 않는다. 분기 표에는 signed 원자료를 사용한다.

## 7. 데이터가 필요한 그림

그림 생성 코드는 현재 지정된 checkpoint 경로와 해시를 사용한다. 기존 가중치를 제거했으므로 데이터만 준비해서는 아래 명령을 실행할 수 없다. 지정된 모델과 입력을 별도로 공급해야 한다. 그림 생성 코드는 `lob_xai_viz`에 통합되어 있으며 단계별 연결은 [실행 코드 안내](../scripts/README.md)에 있다.

```sh
python -m lob_xai_viz.render --only factorization --device cpu
python -m lob_xai_viz.render --only cards --device cpu
python -m lob_xai_viz.render --only gallery --device cpu
```

그림 실행기는 checkpoint·원본 입력·대표 창을 확인하며 검사가 실패하면 최종 그림 내보내기를 중단한다. 세부사항은 [그림 생성 안내](../lob_xai_viz/README.md)와 [논문 대응표](PAPER_MAP.md)를 따른다.
