> **역사 기록:** 아래 내용은 당시 버전의 작업·실험 상태다. 현재 실행법은 [REPRODUCE.md](REPRODUCE.md), 현재 결과 출처는 [PAPER_MAP.md](PAPER_MAP.md), 남은 문제는 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)를 따른다. 경로와 명령을 현재 실행 지시로 사용하지 않는다.

# 다른 정리본(`lob_tcn_xai_clean`)과의 비교

외부에서 제안된 from-scratch 정리본(src-layout, YAML config, CLI entry point, CI)과 이 저장소의 차이, 그리고 무엇을 채택하고 무엇을 채택하지 않았는지를 기록한다.

## 결론 요약

제안본은 **코드 구조는 현대적이지만, 이 논문의 재현 패키지로는 사용할 수 없다.** 과제 정의·checkpoint 호환·프로토콜·baseline·provenance가 현재 실험과 단절되기 때문이다. 대신 제안본의 좋은 아이디어 다섯 가지를 이 저장소에 흡수했다.

## 제안본을 그대로 쓸 수 없는 이유 (결과 계약 단절)

| 항목 | 제안본 | 이 저장소 (논문 계약) |
|---|---|---|
| **과제/label** | CSV 마지막 5열의 제공 mid-price 방향 label (horizon 10/20/30/50/100) | z-feature에서 직접 계산하는 toxic(k=10, 3-class)/vacuum(k=30, 2-class) 미시구조 label — 논문의 핵심 정의 |
| **checkpoint 호환** | `encoders.ask_price.*`, `pools.ask_price.logits` (ModuleDict) | frozen 12+ run의 `cnns.{i}.*`, `pool.pools.{i}.logits` — 제안본 구조로는 어떤 동결 checkpoint도 로드 불가, 전부 재학습 필요 |
| **아키텍처** | GELU head, classifier_hidden 단일, input_norm 없음 | LeakyReLU fc, `identity/dain/revin_c` 입력 정규화(devsel에서 과제별 선택됨), `multi_scale` readout 부재 |
| **baseline** | 없음 | DeepLOB/TLOB/BiN-C(TABL) — 탐지 비교표의 전제 |
| **반사실 프로토콜** | 간이 재구현 (mean/std/slope context) | 감사를 거친 동결 프로토콜 (보존 context 거리, train-only donor, K=50/M=10, seed 규칙) |
| **XAI 체계** | CAM 3종 + 지표 | + Axis A/B/C branch 분해, response carpet, positional readout 감사, saliency shape |
| **provenance** | 없음 | 데이터/checkpoint SHA-256, frozen manifest, run manifest, `--confirm-final-test` 봉인 |
| **위생** | zip 안에 `pip-build-tracker-*`, `pip-download-*` 임시 폴더 포함 | — |

또한 제안본의 YAML config는 `paper_protocol.py`와 이중 기준을 만든다. 이 저장소는 "결과에 영향을 주는 수치는 코드 상수 단일 기준"이라는 감사 원칙을 유지한다.

## 제안본에서 채택한 것

| 아이디어 | 이 저장소의 구현 |
|---|---|
| CI에서 테스트 자동 실행 | `.github/workflows/ci.yml` — unittest + pooling smoke + baseline 검증 (CPU) |
| parameter-randomization sanity check (Adebayo 2018) | `scripts/common/cam_sanity_checks.py` — head/마지막 block 재초기화 후 CAM 상관이 **떨어지는지** 확인 |
| 입력 섭동 안정성 + method 간 일치도 (Spearman/top-k Jaccard) | 같은 스크립트 — hirescam/gradcam/tsr 3종, prediction-preserved 창만 사용 |
| deletion 효과의 paired bootstrap CI + one-sided Wilcoxon | `scripts/common/cf_effect_stats.py` — carpet `.npz`에서 창 단위 paired AUC 통계 (겹침 의존성 경고 포함) |
| train/val 경계 purge gap | `split_within_segments(..., purge_gap=N)` opt-in (기본 0 = 동결 동작 보존) + 테스트 |
| 데이터 배치 문서 | `data/README.md` — 기대 경로와 동결 SHA-256 |

## 채택하지 않은 것과 이유

- **src-layout + console entry point 전환**: frozen checkpoint·run 스크립트·`PYTHONPATH=.` 워크플로와의 호환을 깨고 얻는 것이 적다. 재현 패키지에서는 안정성이 우선.
- **YAML config**: `paper_protocol.py` 단일 기준과 충돌 (이중 기준은 이전 감사에서 결함으로 판정된 패턴).
- **모델 재작성(ModuleDict/GELU)**: 동결 자산 전체를 무효화. 성능/해석 개선 근거 없음.
- **제공 label 기반 방향 예측 과제**: 1세대 과제로의 회귀. 현 원고의 연구 질문이 아님.
- **ruff 강제**: 레거시 코드 전반의 스타일 재작성을 유발. 필요하면 신규 파일에만 점진 적용.
