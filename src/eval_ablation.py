"""
Feature-block ablation: audio-only vs artist-only vs full columns.

Uses one stratified row split (same indices for every regime), IQR + min-max fit on
train only per regime, then trains scratch Logistic Regression and ANN with
hyperparameters taken from artifacts/metrics.json (production grid winners).

Usage (from repo root):
  python src/eval_ablation.py
  python src/eval_ablation.py --fast --out-json artifacts/ablation_metrics.json
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
    FEATURE_COLUMNS_ARTIST,
    FEATURE_COLUMNS_AUDIO,
    prepare_xy_ablation,
    stratified_train_test_indices,
    stratified_train_val_split,
)
from utils.metrics import accuracy_score, auc_roc, best_threshold_by_f1, precision_recall_f1


def _class_weights(y: np.ndarray) -> tuple[float, float]:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    return neg / (pos + 1e-12), 1.0


def _load_hparams(artifacts_dir: Path) -> tuple[dict, dict]:
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
        return dict(m["grid_search"]["logistic_regression"]["best_params"]), dict(
            m["grid_search"]["ann"]["best_params"]
        )
    except (KeyError, json.JSONDecodeError, TypeError):
        return defaults_lr, defaults_ann


def _scale_epochs(cfg: dict, factor: float) -> dict:
    out = dict(cfg)
    out["epochs"] = max(50, int(out["epochs"] * factor))
    return out


def _metrics(model, X, y, thr: float) -> dict:
    prob = model.predict_proba(X)
    pred = (prob >= thr).astype(int)
    _, _, f1 = precision_recall_f1(y, pred)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1),
        "auc": float(auc_roc(y, prob)),
        "threshold": float(thr),
    }


def run_regime(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lr_cfg: dict,
    ann_cfg: dict,
    val_seed: int,
    fast: bool,
) -> dict:
    pos_w, neg_w = _class_weights(y_train)
    if fast:
        lr_cfg = _scale_epochs(lr_cfg, 0.35)
        ann_cfg = _scale_epochs(ann_cfg, 0.35)

    X_fit, y_fit, X_val, y_val = stratified_train_val_split(
        X_train, y_train, val_ratio=0.15, seed=val_seed
    )

    lr_cv = LogisticRegressionScratch(pos_weight=pos_w, neg_weight=neg_w, **lr_cfg)
    lr_cv.fit(X_fit, y_fit, verbose=False, early_stopping_patience=60)
    thr_lr, _ = best_threshold_by_f1(y_val, lr_cv.predict_proba(X_val))
    lr = LogisticRegressionScratch(pos_weight=pos_w, neg_weight=neg_w, **lr_cfg)
    lr.fit(X_train, y_train, verbose=False, early_stopping_patience=80)
    lr_test = _metrics(lr, X_test, y_test, thr_lr)

    ann_cv = ANNClassifierScratch(
        input_dim=X_train.shape[1],
        seed=42,
        pos_weight=pos_w,
        neg_weight=neg_w,
        **ann_cfg,
    )
    ann_cv.fit(X_fit, y_fit, verbose=False, early_stopping_patience=25)
    thr_ann, _ = best_threshold_by_f1(y_val, ann_cv.predict_proba(X_val))
    ann = ANNClassifierScratch(
        input_dim=X_train.shape[1],
        seed=42,
        pos_weight=pos_w,
        neg_weight=neg_w,
        **ann_cfg,
    )
    ann.fit(X_train, y_train, verbose=False, early_stopping_patience=40)
    ann_test = _metrics(ann, X_test, y_test, thr_ann)

    return {
        "input_dim": int(X_train.shape[1]),
        "logistic_regression": lr_test,
        "ann": ann_test,
    }


def main():
    parser = argparse.ArgumentParser(description="Audio vs artist feature ablation.")
    parser.add_argument("--csv", type=str, default=None, help="CSV path (default: data/ under repo).")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Stratified test fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Train/test split seed.")
    parser.add_argument("--fast", action="store_true", help="Fewer epochs for a quick table.")
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Write results JSON (e.g. artifacts/ablation_metrics.json).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    csv_path = Path(args.csv) if args.csv else root / "data" / "spotify_top50_songs_features.csv"
    artifacts_dir = root / "artifacts"

    df = pd.read_csv(csv_path)
    df["hit"] = (df["rank"] <= 10).astype(int)
    df = df.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    y_all = df["hit"].to_numpy(dtype=float)
    train_idx, test_idx = stratified_train_test_indices(y_all, test_ratio=args.test_ratio, seed=args.seed)

    lr_cfg, ann_cfg = _load_hparams(artifacts_dir)

    regimes = [
        ("full", FEATURE_COLUMNS, "All 12 production features"),
        ("audio_only", FEATURE_COLUMNS_AUDIO, "Nine audio / spectral / MFCC / chroma features"),
        ("artist_only", FEATURE_COLUMNS_ARTIST, "Three artist popularity / listeners features"),
    ]

    print("=" * 62)
    print("  Feature ablation (same stratified row split for all regimes)")
    print("=" * 62)
    print(f"  CSV: {csv_path}")
    print(f"  Rows: {len(df):,}  train_idx={len(train_idx):,}  test_idx={len(test_idx):,}")
    print(f"  test_ratio={args.test_ratio}  seed={args.seed}")
    print(f"  LR params: {lr_cfg}")
    print(f"  ANN params: {ann_cfg}")
    print()

    summary: dict = {
        "csv": str(csv_path.resolve()),
        "test_ratio": args.test_ratio,
        "split_seed": args.seed,
        "n_rows": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "hyperparams_from": str((artifacts_dir / "metrics.json").resolve()),
        "regimes": {},
    }

    for i, (key, cols, desc) in enumerate(regimes):
        val_seed = 200 + i
        X_train, y_train, X_test, y_test = prepare_xy_ablation(df, cols, train_idx, test_idx)
        out = run_regime(X_train, y_train, X_test, y_test, lr_cfg, ann_cfg, val_seed, args.fast)
        summary["regimes"][key] = {"description": desc, "feature_names": list(cols), **out}
        lr_m, ann_m = out["logistic_regression"], out["ann"]
        print(f"  --- {key} ({out['input_dim']} features) ---")
        print(f"      {desc}")
        print(
            f"      LR  test  acc={lr_m['accuracy']:.3f}  f1={lr_m['f1']:.3f}  "
            f"auc={lr_m['auc']:.3f}  thr={lr_m['threshold']:.3f}"
        )
        print(
            f"      ANN test  acc={ann_m['accuracy']:.3f}  f1={ann_m['f1']:.3f}  "
            f"auc={ann_m['auc']:.3f}  thr={ann_m['threshold']:.3f}"
        )
        print()

    if args.out_json:
        out_path = Path(args.out_json)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  Wrote {out_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
