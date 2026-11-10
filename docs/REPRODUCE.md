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

위 점검은 기존 학습 가중치 없이 실행할 수 있다. 전체 모델 checkpoint 파일은 배포하지 않으므로 학습된 모델의 적재·평가에는 별도로 준비한 파일이 필요하다. 보존 검사는 남겨 둔 파일의 해시를 확인하며 논문 성능을 검증하는 명령이 아니다.

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

Ours 학습 출력은 `runs/NAME/`에 생성되며 `.gitignore`로 Git 추적에서 제외한다. `--output-dir`으로 다른 경로를 지정할 수 있다. 같은 이름·출력 폴더를 재사용하면 중단하므로 run마다 새 이름 또는 경로를 쓴다. YAML 실행 기록은 생성하지 않는다. 기존 결과 압축파일은 공개 범위 검토 중이며, [현재 범위](PUBLICATION_SCOPE.md)를 참고한다.

`@` 파일은 표준 argparse의 인자 파일이며 한 줄에 한 인자를 저장한다. 따옴표는 PowerShell의 `@` 해석을 피하기 위해 양쪽 셸에서 사용한다. Seed 2와 3도 별도 실행한다.

설정: type4 TCN, channels 32/32/64, positional, identity, T100, toxic k10/vacuum k30, inverse-frequency weighted CE, AdamW 1e-3/weight decay 1e-5, cosine, batch256, 최대100 epoch, patience10, validation macro-F1 선택. `--dev-only --skip-report`로 test 평가와 보고서 생성을 학습에서 분리한다.

뒤에 명시한 인자는 preset을 덮어쓴다. 다른 설정으로 실행했다면 논문 preset과 같은 실험이라고 보고하지 않는다. 과거 CLI와 Config의 기본 GAP는 호환성을 위해 유지한다. TLOB은 원고에 별도로 기술한 optimizer 예외가 있으므로 이 ours preset을 TLOB에 적용하지 않는다.

## 5. 고정 checkpoint 평가

기존 공통 로더에 의존한 `evaluate_ours_checkpoint.py`와 `verify_positional_checkpoint.py`는 제거했다. 이 배포는 해당 고정 모델 평가 명령을 제공하지 않는다. `aggregate_final_test.py`는 별도로 준비된 평가 JSON을 집계하는 도구로 남아 있으며 추론 도구를 대신하지 않는다. 제거 범위는 [공개 범위 안내](PUBLICATION_SCOPE.md)를 참고한다.

## 6. 고신뢰 경보와 분기 분석

독립 구현을 사용하는 아래 CF 명령은 유지한다. `PATH_TO_TOXIC_SEED1_MODEL.pt`는 실제 모델 파일로 바꾼다. 입력 데이터와 호환 모델을 별도로 준비해야 한다.

```sh
python -m lob_xai.counterfactual_test --model ours --ckpt PATH_TO_TOXIC_SEED1_MODEL.pt --task toxic --seed 1 --direction both --device cpu --out validation_output/toxic_seed1_cf.png
```

기존 분기 probe·분기 중요도 계산·CAM sanity 명령은 공통 로더와 함께 제거했다. `plot_branch_importance.py`는 이미 준비된 분석 CSV와 CAM 시간 NPZ 캐시만 읽는다. 캐시가 없으면 계산을 새로 수행하지 않고 중단한다. 고신뢰 CF 결과와 신뢰도 층화 결과는 서로 다른 표본군이므로 합치지 않는다.

## 7. 데이터가 필요한 그림

그림 생성 코드는 현재 지정된 checkpoint 경로와 해시를 사용한다. 기존 가중치를 제거했으므로 데이터만 준비해서는 아래 명령을 실행할 수 없다. 지정된 모델과 입력을 별도로 공급해야 한다. 그림 생성 코드는 `lob_xai_viz`에 통합되어 있으며 단계별 연결은 [실행 코드 안내](../scripts/README.md)에 있다.

```sh
python -m lob_xai_viz.render --only factorization --device cpu
python -m lob_xai_viz.render --only cards --device cpu
python -m lob_xai_viz.render --only gallery --device cpu
```

그림 실행기는 checkpoint·원본 입력·대표 창을 확인하며 검사가 실패하면 최종 그림 내보내기를 중단한다. 세부사항은 [그림 생성 안내](../lob_xai_viz/README.md)와 [논문 대응표](PAPER_MAP.md)를 따른다.
