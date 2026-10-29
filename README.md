# Prism-LOB

Research code for predicting **toxic-flow** and **liquidity-vacuum** risk in limit order books (LOBs), and examining the evidence behind those predictions through **branch attribution** and **perturbation audits (span replacement and restoration)**.

> Jihong Park, Eunho Koo. **Prism-LOB: Structured Evidence and Perturbation Audits for Limit Order Book Risk Prediction.**

Prism stands for **P**ositional **R**eadout over **I**nterpretable **S**emantic **M**odules. It separates ask price, ask volume, bid price and bid volume into four semantic groups to ask: **“Which group, at which time, contributed to this prediction?”** Attribution and interventions examine that evidence. A manuscript link will be added after public release or publication.

**Prism-LOB** is the model name in the paper; **`prism_lob_tcn`** is the GitHub repository name.

[한국어 안내](README.ko.md) · [Reproduction](docs/REPRODUCE.md) · [Paper-to-code map](docs/PAPER_MAP.md) · [Documentation index](docs/README.md)

## Where does FI-2010 come from?

This research uses the **FI-2010** benchmark introduced by Ntakaris et al. The original dataset contains normalized order-book representations for five NASDAQ Nordic stocks over ten consecutive trading days.

| Source | Link |
|---|---|
| Dataset paper | Ntakaris et al., *Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods*, Journal of Forecasting. [DOI](https://doi.org/10.1002/for.2543) · [Full paper](https://arxiv.org/abs/1705.03233) |
| Dataset distribution record | [FI-2010 on Fairdata / Etsin](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) |

The dataset is not included in this repository. Training and evaluation use these CSV files:

```text
data/FI2010/FI2010_train.csv   # Development: 362,400 rows
data/FI2010/FI2010_test.csv    # Test: 31,937 rows
```

Each row supplies 40 price/volume features from the top ten book levels, using FI-2010's z-score values. The code computes this study's risk labels instead of using FI-2010's native price-direction labels.

**The public dataset source and the provenance of the experiment CSVs are distinct.** The exact source files, periods and row selections used to construct these CSVs are not yet fully documented. [Data documentation](docs/DATA.md) records their schema, chronological split, SHA-256 hashes and remaining reproduction requirements.

## What is predicted, and how are explanations checked?

The input is a window of 100 order-book snapshots. Labels compare the best quotes at the window endpoint with a future snapshot.

| Prediction task | Horizon | What is predicted? |
|---|---|---|
| Toxic flow | 10 steps | `stationary / toxic up / toxic down`, based on a rise in the best ask or a fall in the best bid |
| Liquidity vacuum | 30 steps | `normal / vacuum`, based on a simultaneous drop in best-quote volume and increase in a spread proxy |

These labels are **risk proxies**. Exact thresholds and conditions are in the [label implementation](lob_xai/microstructure_labels.py) and [experiment constants](lob_xai/paper_protocol.py).

After prediction, the audits change the features and time spans identified by attribution and measure the model's response.

| Explanation audit | What does it examine? | Code and records |
|---|---|---|
| Branch attribution vs. feature replacement | When a price/volume group is replaced with its training mean, does the ranking of target-logit changes agree with the attribution ranking? | [Branch analysis](scripts/analyze_branch_importance.py) · [Table analysis](docs/PAPER_MAP.md) |
| Temporal weights and attribution | How do learned temporal weights enter the relationship between Grad-CAM and HiResCAM? | [Attribution implementation](lob_xai/gradcam.py) · [Manuscript/figure map](docs/PAPER_MAP.md) |
| Span replacement and restoration | Does replacing a CAM-selected span change the prediction more often, and does restoring it recover the target prediction more often, than random or anti-CAM spans? | [Intervention implementation](lob_xai/counterfactual_test.py) · [Result index](figures/README.md) |

Replacement selects neighboring training examples using the unmasked context. Restoration starts from training examples sampled without conditioning on that context and restores the original span. Branch replacement retains signed logit changes, which need not always be decreases. These audits describe **model responses** under specified cohorts and interventions.

### An example from the stored results

For correct, high-confidence toxic-up alerts at a 50% span replacement ratio, the manuscript reports approximately **82%** decision flips for CAM-selected spans versus **25%** for random spans, averaged across three seeds. Table 2 in the [paper-to-code map](docs/PAPER_MAP.md) identifies the corresponding analysis code and cohort requirements.

![Target-class probability under temporal span replacement and restoration](figures/cf_mirror/cf_mirror.png)

*The deletion curves above use confidence-stratified correct alerts; the insertion curves use high-confidence correct alerts. These are different cohorts. The 82%/25% summary above and the plotted deletion curves should not be read as results from the same cohort.*

The [paper-to-code map](docs/PAPER_MAP.md) connects detection performance and explanation audits to their code and input requirements. Outstanding detection-table provenance/numerical discrepancies and data-conversion gaps are recorded in [known issues](docs/KNOWN_ISSUES.md).

## Quick start

Use Python 3.10 or later. Run these commands from the repository root:

```sh
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` in Bash.

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/check_docs.py
python scripts/smoke_pooling.py --pool positional
```

**These checks do not require FI-2010 data.** View the completed figures in the [paper PNG index](figures/README.md). The original CF curve inputs are not included, so the current checkout alone cannot reaggregate those curves.

With the expected data available, train using the paper configuration files:

```sh
python scripts/train_microstructure.py "@configs/paper/toxic.args" --seed 1 --name paper_toxic_seed1
python scripts/train_microstructure.py "@configs/paper/vacuum.args" --seed 1 --name paper_vacuum_seed1
```

Repeat with seeds 2 and 3, changing the run names accordingly. These configurations select positional readout and run training/validation only. Use the configuration files to avoid the bare CLI's historical GAP default. Follow the [reproduction guide](docs/REPRODUCE.md) for test evaluation, checkpoint checks and new attribution/intervention experiments.

## Repository layout and code reading order

| Path | Role |
|---|---|
| [`lob_xai/`](lob_xai/) | Core data, risk labels, model, training/evaluation, attribution and intervention implementations |
| [`models/`](models/) | DeepLOB, TLOB and BiN-CTABL comparison models |
| [`configs/paper/`](configs/paper/), [`scripts/`](scripts/) | Paper training settings and executable training, evaluation and analysis workflows |
| [`lob_xai_viz/`](lob_xai_viz/) | Paper figure generation using specified inputs and checkpoints |
| [`tests/`](tests/) | Checks for labels, models and experimental procedures |
| [`docs/`](docs/) | Data and execution guidance, with links to table/figure analysis code |

Read the code in experimental order:

1. [Experiment constants](lob_xai/paper_protocol.py) → [risk labels](lob_xai/microstructure_labels.py) and [input data](lob_xai/data.py)
2. [Model](lob_xai/models.py) → [training](lob_xai/training.py) and [evaluation metrics](lob_xai/metrics.py)
3. [Attribution](lob_xai/gradcam.py) → [span replacement/restoration](lob_xai/counterfactual_test.py)

The execution flow is **data → training → checkpoint → evaluation / intervention → figures**. [Scripts by stage](scripts/README.md) connects each step to its implementation. `configs/paper/*.args` are inputs read by the training command; `docs/REPRODUCE.md` explains how to run it. Model/data loading is shared in `lob_xai/checkpoints.py`, and branch analysis in `lob_xai/branch_analysis.py`.

## Where are the experiment records?

| Material | Location |
|---|---|
| Paper figure PNGs | [figures/README.md](figures/README.md): alert cards, CF curves, attribution factorization and representative-alert gallery |
| Model identity records | [configs/paper/checkpoints.json](configs/paper/checkpoints.json): recorded hashes; weight files are not included |
| Table/figure analysis code and inputs for new runs | [Paper-to-code map](docs/PAPER_MAP.md), [reproduction guide](docs/REPRODUCE.md) |

Unpublished manuscripts and previous training runs, logs and weight files are not included. Evaluation requires training a model or supplying a compatible checkpoint. `figures/` contains PNGs and their index. Previous detection summaries, branch-analysis and CF inputs, and historical images are not included. Recomputing table values requires new analysis with the appropriate data and checkpoints, or separately supplied original analysis inputs.

## Citation and reuse

[CITATION.cff](CITATION.cff) records the authors, title and repository. No publication DOI or open-source license is declared. FI-2010 use is subject to the original provider's terms.
