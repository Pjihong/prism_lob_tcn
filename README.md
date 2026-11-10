# Prism-LOB

Research code for predicting toxic-flow and liquidity-vacuum risk proxies in limit order books, and examining branch attribution and responses to temporal span replacement/restoration.

> Jihong Park, Eunho Koo. *Prism-LOB: Structured Evidence and Perturbation Audits for Limit Order Book Risk Prediction.*

Prism stands for Positional Readout over Interpretable Semantic Modules. This public implementation uses **positional readout**: 100 snapshots × 40 features are separated into ask price, ask volume, bid price and bid volume branches, processed by TCNs, and combined through learned temporal weights and a classification head. DeepLOB, TLOB and BiN-CTABL remain separate comparison models.

[한국어](README.ko.md) · [Execution guide (Korean)](docs/REPRODUCE.md) · [Data](docs/DATA.md) · [Seven figures](figures/README.md) · [Known limitations](docs/KNOWN_ISSUES.md)

## Public contents

- Model, labeling, splitting, training, metrics, attribution and intervention code; positional training presets.
- **Seven PNGs**: six selected final figures and one gallery comparing correct and incorrect alerts.
- Tests, dependency specifications and current usage documentation.

Trained weights, run logs, experimental CSV/NPZ results, detailed result JSON, result archives, and historical configuration/cleanup documents are excluded from the current public tree. Raw dataset files are not redistributed or modified by this cleanup. Viewing stored PNGs does not permit numerical reaggregation. See [publication scope](docs/PUBLICATION_SCOPE.md).

## Quick start

Use Python 3.10 or later and run from the repository root.

```sh
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in Bash.

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/check_docs.py
python scripts/smoke_pooling.py --pool positional
python -m lob_xai_viz.plot_factorization --demo --task toxic --target 1 --out-dir figure_revision_output/demo_factorization
```

These checks and the demo do not require FI-2010 or trained weights. The demo uses synthetic inputs and an untrained model; it is not a paper result.

With the expected CSV files available, train new models using:

```sh
python scripts/train_microstructure.py "@configs/paper/toxic.args" --seed 1 --name paper_toxic_seed1
python scripts/train_microstructure.py "@configs/paper/vacuum.args" --seed 1 --name paper_vacuum_seed1
```

Repeat with seeds 2 and 3 and distinct names. Outputs go to `runs/NAME/` or `--output-dir`; existing output directories are never reused. Presets use positional readout, identity normalization and validation-based model selection. Data preparation, new interventions and standalone figure commands are described in the [execution guide](docs/REPRODUCE.md).

## Data and prediction tasks

FI-2010 is the benchmark introduced by Ntakaris et al. [Dataset paper](https://arxiv.org/abs/1705.03233) · [Distribution record](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)

| Task | Horizon | Prediction |
|---|---|---|
| Toxic flow | 10 steps | stationary / toxic up / toxic down, from best-ask rises and best-bid falls |
| Liquidity vacuum | 30 steps | normal / vacuum, from simultaneous volume drops and an increased spread proxy |

The code computes [risk-proxy labels](lob_xai/microstructure_labels.py) rather than using FI-2010's native direction labels. The exact conversion history from the original distribution to the expected CSVs is incomplete. See the [data schema and limits](docs/DATA.md).

## Code reading order

1. [Protocol constants](lob_xai/paper_protocol.py) → [risk labels](lob_xai/microstructure_labels.py) → [input data](lob_xai/data.py)
2. [Model](lob_xai/models.py) → [training](lob_xai/training.py) → [metrics](lob_xai/metrics.py)
3. [Attribution](lob_xai/gradcam.py) → [span replacement/restoration](lob_xai/counterfactual_test.py)
4. [Scripts](scripts/README.md) → [figure tools](lob_xai_viz/README.md)

The current public tree is not a complete package for reconstructing stored paper numbers. Consult the [paper map](docs/PAPER_MAP.md) and [known limitations](docs/KNOWN_ISSUES.md) for the supported scope and unresolved provenance/numerical questions.

## Citation and reuse

[CITATION.cff](CITATION.cff) records the authors, title and repository. A publication DOI and an open-source license have not been declared. FI-2010 use is subject to its provider's terms.
