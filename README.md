# Song Intelligence — Spotify Hit Predictor

Predict whether a song will reach the **Top 10** of the Spotify Weekly Global
Top 50, using audio and artist features. Built from scratch with **NumPy only**
— no scikit-learn, PyTorch, or TensorFlow.

> INFO 5368 · PAML Final Project
> Zoe Tseng · Jay Huang · Charlotte Lin · Irene Wu · Jessica Hsiao

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
`artifacts/`, so no training is required to try the dashboard.

**To retrain from scratch** (~4 min):

```bash
python src/train.py
```

Python **3.11** is pinned via `runtime.txt` / `.python-version` so local,
teammate, and Streamlit Cloud deployments all get the same chrome, icons, and
Plotly rendering.

---

## What You Get

A 3-page Streamlit dashboard:

1. **Song Hit Predictor** — input audio + artist features → hit probability,
   Top-10 verdict, and feature importance.
2. **Data Explorer** — histogram, scatter, correlation heatmap, genre
   breakdown. Filter by year/artist.
3. **Model Comparison** — Accuracy / F1 / AUC-ROC table, ROC curves, grid
   search best params, temporal split results.

---

## Project Layout

```
.
├── app/streamlit_app.py        # the dashboard
├── src/
│   ├── train.py                # training pipeline
│   ├── models/                 # LR + ANN (from scratch, NumPy)
│   └── utils/                  # data prep, k-fold CV, metrics
├── data/                       # Spotify Top-50 dataset (CSV)
├── artifacts/                  # .npz weights + JSON metadata
├── .streamlit/config.toml      # forces light theme for everyone
└── requirements.txt
```

---

## Training (optional)

`python src/train.py` runs six phases with live progress bars:

1. Load data → drop nulls → IQR-cap outliers → min-max scale → 80/20 split
2. Logistic Regression grid search (5-fold stratified CV)
3. ANN grid search (5-fold stratified CV)
4. Final model training with early stopping
5. Temporal evaluation (train pre-2025 / test 2025+)
6. Save `.npz` weights + `metrics.json`

Speed things up for a quick demo:

```bash
K_FOLDS=2 python src/train.py
```

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
