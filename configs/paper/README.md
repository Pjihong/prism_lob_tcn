# 논문 실행 설정

`docs/REPRODUCE.md`는 사람이 읽는 실행 순서이고, 이 폴더는 Python이 실제로 읽는 설정입니다.
학습 인자를 문서에서 매번 복사하지 않도록 두 과제의 기본 실험 조건을 파일로 고정합니다.

| 파일 | 읽는 코드 | 역할 |
|---|---|---|
| `toxic.args` | `scripts/train_microstructure.py` | Toxic 과제의 학습 인자 |
| `vacuum.args` | 같은 학습 스크립트 | Vacuum 과제의 학습 인자 |
| `checkpoints.json` | `scripts/verify_positional_checkpoint.py --manifest` | 기존 논문 모델 6개의 경로·SHA-256 식별 기록 |

```sh
python scripts/train_microstructure.py "@configs/paper/toxic.args" --seed 1
python scripts/train_microstructure.py "@configs/paper/vacuum.args" --seed 1
```

`.args`는 한 줄에 한 토큰을 쓰는 argparse 인자 파일입니다. PowerShell에서는 `@경로`를 따옴표로 감쌉니다.
두 설정은 positional readout, identity 정규화, validation 기반 checkpoint 선택을 사용합니다.
`--dev-only`이므로 test 평가는 따로 실행합니다. 뒤에 쓴 CLI 인자가 앞의 설정을 덮어쓰며,
변경한 경우 논문 설정과 다른 실행임을 기록해야 합니다. Seed는 1–3을 각각 실행합니다.

`checkpoints.json`은 학습 가중치가 아닙니다. 과거 `frozen/`의 현재 논문 모델 목록을 이곳으로 옮겼습니다.
여기서 frozen은 **선택한 모델과 실험 조건을 고정해 평가에 사용하는 것**을 뜻하며,
모든 파라미터에 `requires_grad=False`를 설정한다는 의미가 아닙니다.
기존 가중치는 배포하지 않으므로 목록에 있는 파일을 별도로 준비한 경우에만 다음 검사를 실행합니다.

```sh
python scripts/verify_positional_checkpoint.py --manifest configs/paper/checkpoints.json
```

새로 학습한 모델은 해시가 달라질 수 있습니다. 그 파일을 검증하려면 `--checkpoint 실제경로/model.pt`를 사용하세요.
전체 순서는 [실행·재현 안내](../../docs/REPRODUCE.md)를 따릅니다.
