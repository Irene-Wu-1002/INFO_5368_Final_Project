"""
Artist-stratified evaluation: each fold holds out a disjoint set of primary artists from
training so test rows never include artists seen during training.

Usage (from repo root):
  python src/eval_artist_stratified.py
  python src/eval_artist_stratified.py --k 5 --seed 42 --fast
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.ann import ANNClassifierScratch
from models.logistic_regression import LogisticRegressionScratch
from utils.data import (
    FEATURE_COLUMNS,
    artist_stratified_kfold_indices,
    prepare_xy_artist_fold,
    stratified_fit_val_indices,
)
from utils.metrics import auc_roc, best_threshold_by_f1, precision_recall_f1


def _safe_auc(y: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return auc_roc(y, prob)


def _metrics_at_threshold(y: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(int)
    _, _, f1 = precision_recall_f1(y, pred)
    return {"f1": float(f1), "auc": _safe_auc(y, prob)}


def _class_weights(y: np.ndarray) -> tuple[float, float]:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    pos_weight = neg / (pos + 1e-12)
    return pos_weight, 1.0


def _load_hparams_from_artifacts(artifacts_dir: Path) -> tuple[dict, dict]:
    path = artifacts_dir / "metrics.json"
    defaults_lr = {"lr": 0.03, "l2_lambda": 1e-3, "epochs": 1200}
    defaults_ann = {
        "hidden_dim": 64,
        "lr": 0.02,
        "epochs": 700,
        "batch_size": 128,
        "l2_lambda": 5e-4,
        "dropout_p": 0.3,
    }
    if not path.is_file():
        return defaults_lr, defaults_ann
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
        lr_p = dict(m["grid_search"]["logistic_regression"]["best_params"])
        ann_p = dict(m["grid_search"]["ann"]["best_params"])
        return lr_p, ann_p
    except (KeyError, json.JSONDecodeError, TypeError):
        return defaults_lr, defaults_ann


def _scale_epochs(cfg: dict, key: str, factor: float) -> dict:
    out = dict(cfg)
    out[key] = max(50, int(out[key] * factor))
    return out


def _per_artist_table(
    artists: np.ndarray,
    y: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    min_rows: int = 8,
    max_rows: int = 25,
) -> list[dict]:
    rows: list[dict] = []
    for a in np.unique(artists):
        m = artists == a
        ya = y[m]
        pa = prob[m]
        n = int(ya.size)
        if n < min_rows:
            continue
        auc_a = _safe_auc(ya, pa)
        _, _, f1_a = precision_recall_f1(ya, (pa >= threshold).astype(int))
        rows.append(
            {
                "artist": a,
                "n": n,
                "P(hit)": float(np.mean(ya)),
                "auc": auc_a,
                "f1": float(f1_a),
            }
        )
    rows.sort(key=lambda r: (-(0.0 if np.isnan(r["auc"]) else r["auc"]), -r["n"]))
    return rows[:max_rows]


def run_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lr_cfg: dict,
    ann_cfg: dict,
    val_ratio: float,
    val_seed: int,
    fast: bool,
) -> dict:
    y_tr_all = train_df["hit"].to_numpy(dtype=float)
    fit_idx, val_idx = stratified_fit_val_indices(y_tr_all, val_ratio=val_ratio, seed=val_seed)
    train_fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
    train_val_df = train_df.iloc[val_idx].reset_index(drop=True)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_xy_artist_fold(
        train_fit_df,
        train_val_df,
        test_df.reset_index(drop=True),
        FEATURE_COLUMNS,
        cap_outliers=True,
    )
    pos_w, neg_w = _class_weights(y_train)
    if fast:
        lr_cfg = _scale_epochs(lr_cfg, "epochs", 0.35)
        ann_cfg = _scale_epochs(ann_cfg, "epochs", 0.35)

    lr = LogisticRegressionScratch(pos_weight=pos_w, neg_weight=neg_w, **lr_cfg)
    lr.fit(X_train, y_train, verbose=False, early_stopping_patience=40)
    lr_thr, _ = best_threshold_by_f1(y_val, lr.predict_proba(X_val))
    lr_test = _metrics_at_threshold(y_test, lr.predict_proba(X_test), lr_thr)

    ann = ANNClassifierScratch(
        input_dim=X_train.shape[1],
        seed=42,
        pos_weight=pos_w,
        neg_weight=neg_w,
        **ann_cfg,
    )
    ann.fit(X_train, y_train, verbose=False, early_stopping_patience=25)
    ann_thr, _ = best_threshold_by_f1(y_val, ann.predict_proba(X_val))
    ann_test = _metrics_at_threshold(y_test, ann.predict_proba(X_test), ann_thr)

    artists_test = test_df["primary_artist"].to_numpy()
    lr_rows = _per_artist_table(artists_test, y_test, lr.predict_proba(X_test), lr_thr)
    ann_rows = _per_artist_table(artists_test, y_test, ann.predict_proba(X_test), ann_thr)

    return {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_test_artists": int(test_df["primary_artist"].nunique()),
        "P_hit_train": float(np.mean(y_train)),
        "P_hit_test": float(np.mean(y_test)),
        "logistic_regression": {**lr_test, "threshold": lr_thr, "per_artist_head": lr_rows},
        "ann": {**ann_test, "threshold": ann_thr, "per_artist_head": ann_rows},
    }


def main():
    parser = argparse.ArgumentParser(description="Artist-stratified K-fold evaluation.")
    parser.add_argument("--k", type=int, default=5, help="Number of artist folds.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for artist groups.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Row holdout from train for threshold tuning.")
    parser.add_argument("--fast", action="store_true", help="Shorter training (fewer epochs).")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV (default: data/spotify_top50_songs_features.csv under repo root).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    csv_path = Path(args.csv) if args.csv else root / "data" / "spotify_top50_songs_features.csv"
    artifacts_dir = root / "artifacts"

    lr_cfg, ann_cfg = _load_hparams_from_artifacts(artifacts_dir)

    df = pd.read_csv(csv_path)
    df["hit"] = (df["rank"] <= 10).astype(int)
    need = FEATURE_COLUMNS + ["hit", "primary_artist"]
    df = df.dropna(subset=need).copy()
    artists = df["primary_artist"].to_numpy()

    folds = artist_stratified_kfold_indices(artists, k=args.k, seed=args.seed)

    print("=" * 60)
    print("  Artist-stratified evaluation (held-out primary artists)")
    print("=" * 60)
    print(f"  CSV: {csv_path}")
    print(f"  Rows (cleaned): {len(df):,}  Artists: {df['primary_artist'].nunique():,}")
    print(f"  K={args.k} folds  val_ratio={args.val_ratio} (row holdout for threshold)")
    print(f"  LR hparams: {lr_cfg}")
    print(f"  ANN hparams: {ann_cfg}")
    print()

    fold_results: list[dict] = []
    for fold_i, (tr_idx, te_idx) in enumerate(folds, 1):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        test_df = df.iloc[te_idx].reset_index(drop=True)
        y_test_fold = test_df["hit"].to_numpy()
        if len(np.unique(y_test_fold)) < 2:
            print(f"  Fold {fold_i}: SKIP (test set has single class)")
            continue
        print(f"  --- Fold {fold_i}/{args.k} ---")
        print(
            f"      train rows={len(train_df):,}  test rows={len(test_df):,}  "
            f"test artists={test_df['primary_artist'].nunique():,}"
        )
        r = run_fold(train_df, test_df, lr_cfg, ann_cfg, args.val_ratio, args.seed + fold_i, args.fast)
        fold_results.append(r)
        lr_m, ann_m = r["logistic_regression"], r["ann"]
        print(
            f"      Logistic Regression  test AUC={lr_m['auc']:.4f}  F1={lr_m['f1']:.4f}  "
            f"(thr={lr_m['threshold']:.3f})"
        )
        print(
            f"      ANN                  test AUC={ann_m['auc']:.4f}  F1={ann_m['f1']:.4f}  "
            f"(thr={ann_m['threshold']:.3f})"
        )
        print("      Per-artist (top by AUC, min 8 rows, needs both classes for AUC):")
        for row in ann_m["per_artist_head"][:8]:
            auc_s = "nan" if np.isnan(row["auc"]) else f"{row['auc']:.3f}"
            print(
                f"        {row['artist'][:40]:<40} n={row['n']:<5} P(hit)={row['P(hit)']:.2f}  "
                f"AUC={auc_s}  F1={row['f1']:.3f}"
            )
        print()

    if fold_results:
        def mean_std(key, model):
            vals = [fr[model][key] for fr in fold_results]
            a = np.array(vals, dtype=float)
            return float(np.nanmean(a)), float(np.nanstd(a))

        print("  --- Mean ± std over completed folds ---")
        for model, label in [("logistic_regression", "Logistic Regression"), ("ann", "ANN")]:
            m_auc, s_auc = mean_std("auc", model)
            m_f1, s_f1 = mean_std("f1", model)
            print(f"  {label:22s}  AUC {m_auc:.4f} ± {s_auc:.4f}   F1 {m_f1:.4f} ± {s_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
