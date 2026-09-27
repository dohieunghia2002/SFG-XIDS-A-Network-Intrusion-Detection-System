# SFG-XIDS — Semantic Feature-Group Fusion for Explainable Network Intrusion Detection

A network intrusion detector whose prediction comes with a **two-level explanation that is exact
by construction**, not estimated after the fact:

1. **Level 1 — feature groups.** The classifier head is a generalised additive model with pairwise
   interactions, `z = b + Σᵢ uᵢ(eᵢ) + Σᵢ<ⱼ vᵢⱼ(eᵢ, eⱼ)` over 4 semantic feature groups. Because of
   that structure each group's Shapley value φᵢ has a **closed form** — no sampling. Verified
   against exhaustive 2⁴-coalition Shapley: max error **1.9e-6**, and Σᵢφᵢ = z_c − E[z_c] holds to
   **3.8e-6**.
2. **Level 2 — individual features.** Expected Gradients inside the chosen group, integrated in
   embedding space, so that the sum over a group's features returns that group's φ up to quadrature
   error only.

Level 1 costs ~0.5 ms per flow on a CPU; level 2 costs seconds. That three-order gap is why the
explanation is split in two, and the demo app is built around it: level 1 runs on every flow,
level 2 only on the flow an analyst opens.

## Results

Macro-F1 (%) on the sealed test split, mean ± SD over 5 seeds. Everything was selected on
validation macro-F1 only; the test set was never used to choose anything.

| Model | NSL-KDD (5 classes) | UNSW-NB15 (10 classes) |
| --- | --- | --- |
| Random Forest | 47.86 ± 0.41 | 48.50 ± 0.33 |
| LightGBM | 54.23 ± 0.71 | **52.68 ± 0.29** |
| XGBoost | 57.79 ± 0.97 | 51.43 ± 0.29 |
| Flat-MLP | 58.76 ± 2.21 | 52.28 ± 0.55 |
| SFG-Concat | 55.98 ± 2.70 | 52.42 ± 0.80 |
| SFG-Attn | 59.63 ± 3.63 | 52.16 ± 0.70 |
| **SFG-XIDS (ours)** | **62.88 ± 1.89** | 52.35 ± 1.51 |

Read this honestly: on NSL-KDD the structured head is ahead of every baseline; on UNSW-NB15 it is
level with them, with LightGBM nominally higher and well inside one SD. The claim here is **not** a
new accuracy record — it is that an exact, checkable, two-level explanation costs no accuracy.

**The explanations recover a known intrusion-detection result.** Lee & Stolfo argued that content
features carry R2L/U2R while traffic features carry DoS/Probe. Without being told this, the learned
group profiles reproduce both, on all 5 seeds:

| Hypothesis | Measured | Verdict |
| --- | --- | --- |
| H1 — Content share higher for R2L/U2R than for DoS/Probe | 0.344 vs 0.098 | SUPPORTED (5/5 seeds) |
| H2 — Time+Host share higher for DoS/Probe than Content | 0.626 vs 0.098 | SUPPORTED (5/5 seeds) |

Group profiles are stable across seeds: for 100% of classes the top-1 group is unchanged in a
majority of seeds, with mean Kendall τ-b = 0.72 (NSL-KDD) and 0.85 (UNSW-NB15).

Every number above is reproducible from `sfg_results/`. `summary.json` holds the full grid,
including the ablations — A1 main effects only, A3 no group dropout, A4 random grouping, A5 TTL
features removed, A6 no periodic embedding — reported with their paired-by-seed differences,
whichever direction they went.

## Demo application

https://github.com/user-attachments/assets/4a4cb8d4-7c09-4e99-b225-ccb8cc36ef88

*Five minutes, unscripted: the flows are drawn at random from the test set, repeatedly, so nothing is cherry-picked. A copy is in [demo/demo.mp4](demo/demo.mp4). A scripted walkthrough with the exact test rows and the values each should produce is in [demo/DEMO.md](demo/DEMO.md).*

`app/app_sfg_xids.py` is a Streamlit app that loads the trained checkpoints and runs for real on
every flow — nothing in the interface is precomputed. Two modes:

* **Batch monitoring** — level 1 over a batch of flows, ranked into an alert queue by 1 − P(Normal),
  with each group's φ shown per flow. 900–1,800 flows/s on a laptop CPU.
* **Single-flow investigation** — adds level 2: per-feature Expected Gradients inside the group that
  drove the decision.

The app displays its own correctness checks live: the preprocessing round-trip error (~5e-7), the
Σᵢφᵢ = z_c − E[z_c] residual (~1e-6), and the gap between the summed level-2 attributions and the
group's φ (0.05–0.6% of total attribution mass at 16 quadrature nodes).

`demo/DEMO.md` is a walkthrough with specific test-set rows and the values each one should produce.

### Run it

```bash
cd app
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate elsewhere
pip install -r requirements.txt
streamlit run app_sfg_xids.py
```

Tested on Python 3.10 / Windows with the versions pinned in `app/requirements.txt` (streamlit
1.64.0, torch 2.14.0, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, pyarrow 25.0.1). A CPU build of
torch is enough, and XGBoost is **not** required to run the demo.

### Data (not in this repository)

Both datasets are public but redistributed under their own terms, so they are not committed here.
Put them in `app/data/`:

* **NSL-KDD** — `KDDTrain+.txt` (125,973 rows) and `KDDTest+.txt` (22,544 rows), from
  <https://www.unb.ca/cic/datasets/nsl.html> or the Kaggle mirror `hassan06/nslkdd`.
  `app/modules/step0_sfg.py` downloads them automatically when they are missing.
* **UNSW-NB15** — `UNSW_NB15_training-set.csv` (175,341 rows) and `UNSW_NB15_testing-set.csv`
  (82,332 rows), from <https://research.unsw.edu.au/projects/unsw-nb15-dataset>. Some Kaggle mirrors
  swap the two file names; the app detects that by row count and corrects it.

The app refuses to start on a dataset whose row counts differ from the official split: the
preprocessing statistics were fitted on that exact split, so anything else would make the
checkpoints read shifted features without any visible error.

## Repository layout

```
app/
    app_sfg_xids.py        Streamlit demo — two-level explanation, computed live
    requirements.txt       pinned, ASCII-only (pip reads this file with the OS locale codec)
    README.md              app-specific notes and troubleshooting
    modules/               research code, imported by the app
        step0_sfg.py       data prep; PLR embeddings, group encoders, GAI head;
                           closed-form group Shapley + exhaustive-Shapley self-test
        full_sfg.py        phase 1 — 7 models x 5 seeds, ablations A1/A3/A4/A5/A6, profiles
        phase2_sfg.py      phase 2 — level-2 Expected Gradients, binary task, FT-Transformer
        step0_unsw.py      UNSW-NB15 loading and the earlier baseline sweep
    models/                SFG-XIDS checkpoints, {nsl,unsw}_sfg_tax_{0..4}.pt
    data/                  datasets go here (git-ignored)
sfg_results/               metrics, group profiles, interaction heat maps, per-class F1
demo/                      DEMO.md walkthrough and the recorded clip
SFG_final.pdf              full report
results.xlsx               result tables
run_code_in_colab.ipynb    Colab driver for re-running the experiments on a T4
```

## Reproducing the experiments

Free-tier Colab with a T4 is enough; phase 1 takes roughly 2–3 hours.

```bash
pip install -U xgboost lightgbm kagglehub
python -u app/modules/step0_sfg.py  --selftest           # closed-form φ == exhaustive Shapley
python -u app/modules/full_sfg.py   --out_dir full_v1    # phase 1
python -u app/modules/phase2_sfg.py --p1_dir full_v1 --out_dir phase2_v1
```

Every script is restartable: re-run the same command and finished runs are skipped. All protocol
decisions — seeds, the σ and p grids, the epoch budget, the explanation sample — were fixed in
writing before the first run and are recorded in each script's docstring.

## Citation

```bibtex
@misc{sfgxids,
  author = {Hieu Nghia},
  title  = {SFG-XIDS: Semantic Feature-Group Fusion with Hierarchical Explainability
            for Network Intrusion Detection},
  year   = {2026},
  url    = {https://github.com/<user>/<repo>}
}
```

## License

Code is released under the MIT License (see `LICENSE`). NSL-KDD and UNSW-NB15 are not included and
remain under the terms set by their providers.
