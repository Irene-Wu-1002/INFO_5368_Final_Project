# Song Intelligence — Spotify Hit Predictor

Predict whether a song will reach the **Top 10** of the Spotify Weekly Global
Top 50, using audio and artist features. Models are implemented from scratch
with **NumPy** (and **Pandas** for CSV / preprocessing)—no scikit-learn,
PyTorch, or TensorFlow training APIs.

> INFO 5368 · PAML Final Project
> Charlotte Lin · Irene Wu · Jay Huang · Jessica Hsiao · Zoe Tseng

---

## Quick Start

Use **Python 3.11** and install into a fresh virtual environment so every
teammate's app looks identical:

```bash
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`. Pre-trained weights ship in
`artifacts/` (production and optional **benchmark** `*_benchmark.npz` /
`scaler_config_benchmark.json`), so no training is required to try the
dashboard.

**To retrain from scratch** (several minutes; longer with full 5-fold CV):

```bash
python src/train.py
```

Python **3.11** is pinned via `runtime.txt` / `.python-version` so local,
teammate, and Streamlit Cloud deployments all get the same chrome, icons, and
Plotly rendering.

---

## What You Get

A 3-page Streamlit dashboard:

1. **Song Hit Predictor** — audio + artist features → hit probability,
   Top-10 verdict, and feature importance. Optional **Benchmark** mode (when
   artifacts exist) adds **streams** and **weeks on chart** for a leakage
   comparison; **Production** stays deployable (12 features only).
2. **Data Explorer** — histogram, scatter, correlation heatmap, genre
   breakdown. Filter by year/artist.
3. **Model Comparison** — Accuracy / F1 / AUC-ROC table, ROC curves, grid
   search best params, temporal split results.
4. **Experiments notebook** (`notebooks/experiments.ipynb`) — reproduces every
   table and headline number reported in the final paper. Loads saved metrics
   from `artifacts/` and optionally re-runs the full pipeline. See
   "Reproducing the Experiments" below.

---

## Project Layout

```
.
├── app/streamlit_app.py        # the dashboard
├── notebooks/experiments.ipynb # reproduces the tables in the report
├── src/
│   ├── train.py                # training pipeline (+ benchmark models)
│   ├── eval_artist_stratified.py  # held-out-artist K-fold eval (optional)
│   ├── eval_ablation.py        # audio-only vs artist-only vs full (optional)
│   ├── models/                 # LR + ANN (from scratch, NumPy)
│   └── utils/                  # data prep, k-fold CV, metrics
├── data/                       # Spotify Top-50 dataset (CSV)
├── artifacts/                  # .npz weights, scalers, metrics.json;
│                               # optional: *_benchmark.*, ablation_metrics.json
├── .streamlit/config.toml      # forces light theme for everyone
└── requirements.txt
```

---

## Training (optional)

`python src/train.py` runs seven phases with live progress bars:

1. Load data → drop nulls → IQR-cap outliers → min-max scale → 80/20 split
2. Logistic Regression grid search (5-fold stratified CV)
3. ANN grid search (5-fold stratified CV)
4. Final **production** models (12 features) with early stopping
5. **Benchmark** models (+ `streams`, `weeks_on_chart`) for leakage study
6. Temporal evaluation (train pre-2025 / test 2025+)
7. Save production + benchmark `.npz` weights, scalers, and `metrics.json`

**Other offline eval (optional, does not change deployed weights):**

```bash
python src/eval_artist_stratified.py   # K-fold by primary artist
python src/eval_ablation.py            # audio vs artist feature blocks
python src/eval_ablation.py --out-json artifacts/ablation_metrics.json
```

Speed things up for a quick demo:

```bash
K_FOLDS=2 python src/train.py
```

---

## Reproducing the Experiments

The notebook `notebooks/experiments.ipynb` is the entry point for graders who
want to verify the numbers in the final report. It loads saved metrics from
`artifacts/metrics.json` and `artifacts/ablation_metrics.json` and renders the
four tables that appear in the paper:

| Notebook section | Reproduces | Source script |
|---|---|---|
| §2 — Tables I, II, III | Stratified 80/20 split · 2025 temporal split · chart-context benchmark | `src/train.py` |
| §3 — Table IV | Feature-block ablation (full / audio-only / artist-only) | `src/eval_ablation.py` |
| §5 — Artist-stratified evaluation | 5-fold CV grouped by `primary_artist` | `src/eval_artist_stratified.py` |

Open and run:

```bash
jupyter notebook notebooks/experiments.ipynb
```

Run all cells from a fresh kernel started at the **repo root**, not from inside
`notebooks/`. The first few cells load the saved JSON artifacts and print the
tables — no retraining required.

To regenerate every artifact from scratch (5–15 min on a laptop), either
uncomment the optional cell in §4 of the notebook, or run from the command
line:

```bash
python src/train.py                                              # Tables I, II, III + weights
python src/eval_ablation.py --out-json artifacts/ablation_metrics.json  # Table IV
python src/eval_artist_stratified.py --k 5                       # §III.D paragraph
```

> ⚠️ Re-running `src/train.py` overwrites the `.npz` weights and
> `metrics.json`. ANN training is stochastic (mini-batch shuffling, dropout,
> He init), so fresh numbers will land within roughly ±0.05 AUC of the values
> reported in the paper but will not match to four decimals.

---

## Troubleshooting

**UI looks dark or broken?** The app ships with `.streamlit/config.toml` and a
`darkreader-lock` tag to force light mode for everyone. If it still looks off:

- Make sure the hidden `.streamlit/` folder is present.
- Hard refresh (`Cmd/Ctrl + Shift + R`).
- Disable "Dark Reader" or similar browser extensions for this site.

**Import error on `utils.data`?** Run from the project root so the `src/`
imports resolve correctly.

**Icons or layout look different from a teammate's?** You're almost certainly
on different Streamlit / Python versions. Reinstall into a clean venv:

```bash
deactivate 2>/dev/null
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On **Streamlit Community Cloud**, open the app's *Settings → Advanced* and
set the Python version to **3.11** to match `runtime.txt`.

---

## Constraint

Only NumPy + Pandas for modeling. No scikit-learn, XGBoost, TensorFlow,
or PyTorch training APIs. Streamlit / Plotly / Matplotlib are UI-only.
