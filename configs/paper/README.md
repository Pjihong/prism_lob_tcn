# Positional 학습 설정

`toxic.args`와 `vacuum.args`는 `scripts/train_microstructure.py`가 읽는 argparse 인자 파일입니다. 한 줄에 한 토큰을 저장합니다.

```sh
python scripts/train_microstructure.py "@configs/paper/toxic.args" --seed 1 --name paper_toxic_seed1
python scripts/train_microstructure.py "@configs/paper/vacuum.args" --seed 1 --name paper_vacuum_seed1
```

두 과제 모두 positional readout, identity 정규화, validation macro-F1 기반 모델 선택을 사용합니다. `--dev-only --skip-report`는 test 추론과 보고서 생성을 분리합니다. Seed 2·3도 고유한 이름으로 실행하세요.

출력은 `runs/NAME/` 또는 `--output-dir`에 저장하며 기존 디렉터리는 재사용하지 않습니다. 기존 checkpoint 파일과 상세 식별 원장은 공개하지 않습니다. [실행 안내](../../docs/REPRODUCE.md)
