import json
import sys
from pathlib import Path

# Ensure the src/ directory is on the path when run as `python src/train.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from models.ann import ANNClassifierScratch
from models.logistic_regression import LogisticRegressionScratch
from utils.data import (
    load_and_prepare_data,
    save_scaler,
    stratified_kfold_indices,
    temporal_split_from_df,
)
from utils.metrics import (
    accuracy_score,
    auc_roc,
    best_threshold_by_f1,
    precision_recall_f1,
    roc_curve_points,
)


def evaluate_model(model, X, y, threshold=0.5):
    prob = model.predict_proba(X)
    pred = (prob >= threshold).astype(int)
    acc = accuracy_score(y, pred)
    _, _, f1 = precision_recall_f1(y, pred)
    auc = auc_roc(y, prob)
    fpr, tpr, _ = roc_curve_points(y, prob)
    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "threshold": float(threshold),
    }


def composite_score(m):
    return 0.4 * m["f1"] + 0.4 * m["auc"] + 0.2 * m["accuracy"]


def _avg_metrics(metrics_list):
    keys = ["accuracy", "f1", "auc"]
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def grid_search_logistic(X, y, pos_weight, neg_weight, k_folds=5, seed=42):
    """Grid search with stratified k-fold CV (proposal §4.1.3)."""
    candidates = [
        {"lr": 0.01, "l2_lambda": 1e-3, "epochs": 900},
        {"lr": 0.03, "l2_lambda": 1e-3, "epochs": 1200},
        {"lr": 0.08, "l2_lambda": 1e-3, "epochs": 1600},
        {"lr": 0.03, "l2_lambda": 1e-2, "epochs": 1400},
    ]
    folds = stratified_kfold_indices(y, k=k_folds, seed=seed)
    best_cfg = None
    best_threshold = 0.5
    best_score = -1.0
    all_results = []
    for i, cfg in enumerate(candidates, 1):
        print(f"  Config {i}/{len(candidates)}: lr={cfg['lr']}, l2={cfg['l2_lambda']}, epochs={cfg['epochs']}")
        fold_metrics = []
        fold_thresholds = []
        for f_idx, (tr_idx, va_idx) in enumerate(folds, 1):
            print(f"    Fold {f_idx}/{k_folds}")
            model = LogisticRegressionScratch(pos_weight=pos_weight, neg_weight=neg_weight, **cfg)
            model.fit(X[tr_idx], y[tr_idx], verbose=True, early_stopping_patience=50)
            val_prob = model.predict_proba(X[va_idx])
            thr, _ = best_threshold_by_f1(y[va_idx], val_prob)
            m = evaluate_model(model, X[va_idx], y[va_idx], threshold=thr)
            fold_metrics.append(m)
            fold_thresholds.append(thr)
        avg = _avg_metrics(fold_metrics)
        score = composite_score(avg)
        avg_threshold = float(np.mean(fold_thresholds))
        print(f"  -> CV score={score:.4f}  (acc={avg['accuracy']:.3f}, f1={avg['f1']:.3f}, auc={avg['auc']:.3f})")
        all_results.append({"params": cfg, "threshold": avg_threshold, "metrics": avg, "score": score})
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_threshold = avg_threshold
    return best_cfg, best_threshold, all_results


def grid_search_ann(X, y, input_dim, pos_weight, neg_weight, k_folds=5, seed=42):
    """Grid search with stratified k-fold CV (proposal §4.1.3)."""
    candidates = [
        {"hidden_dim": 64, "lr": 0.01, "epochs": 500, "batch_size": 128, "l2_lambda": 1e-3, "dropout_p": 0.2},
        {"hidden_dim": 64, "lr": 0.02, "epochs": 700, "batch_size": 128, "l2_lambda": 5e-4, "dropout_p": 0.3},
        {"hidden_dim": 128, "lr": 0.01, "epochs": 700, "batch_size": 128, "l2_lambda": 1e-3, "dropout_p": 0.3},
        {"hidden_dim": 128, "lr": 0.01, "epochs": 700, "batch_size": 256, "l2_lambda": 1e-3, "dropout_p": 0.5},
    ]
    folds = stratified_kfold_indices(y, k=k_folds, seed=seed)
    best_cfg = None
    best_threshold = 0.5
    best_score = -1.0
    all_results = []
    for i, cfg in enumerate(candidates, 1):
        print(
            f"  Config {i}/{len(candidates)}: hidden={cfg['hidden_dim']}, lr={cfg['lr']}, "
            f"epochs={cfg['epochs']}, dropout={cfg['dropout_p']}"
        )
        fold_metrics = []
        fold_thresholds = []
        for f_idx, (tr_idx, va_idx) in enumerate(folds, 1):
            print(f"    Fold {f_idx}/{k_folds}")
            model = ANNClassifierScratch(
                input_dim=input_dim,
                seed=seed,
                pos_weight=pos_weight,
                neg_weight=neg_weight,
                **cfg,
            )
            model.fit(X[tr_idx], y[tr_idx], verbose=True, early_stopping_patience=30)
            val_prob = model.predict_proba(X[va_idx])
            thr, _ = best_threshold_by_f1(y[va_idx], val_prob)
            m = evaluate_model(model, X[va_idx], y[va_idx], threshold=thr)
            fold_metrics.append(m)
            fold_thresholds.append(thr)
        avg = _avg_metrics(fold_metrics)
        score = composite_score(avg)
        avg_threshold = float(np.mean(fold_thresholds))
        print(f"  -> CV score={score:.4f}  (acc={avg['accuracy']:.3f}, f1={avg['f1']:.3f}, auc={avg['auc']:.3f})")
        all_results.append({"params": cfg, "threshold": avg_threshold, "metrics": avg, "score": score})
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_threshold = avg_threshold
    return best_cfg, best_threshold, all_results


def _section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def main():
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "spotify_top50_songs_features.csv"
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _section("Song Intelligence — Training Pipeline")

    print("\n[1/6] Loading and preparing data...")
    prepared = load_and_prepare_data(str(csv_path), test_ratio=0.2, seed=42)
    pos_count = np.sum(prepared.y_train == 1)
    neg_count = np.sum(prepared.y_train == 0)
    pos_weight = float(neg_count / (pos_count + 1e-12))
    neg_weight = 1.0
    print(
        f"  Train samples: {len(prepared.y_train)}  "
        f"(pos={int(pos_count)}, neg={int(neg_count)})  "
        f"Test samples: {len(prepared.y_test)}  "
        f"Features: {prepared.X_train.shape[1]}"
    )

    # Allow overriding k with the K_FOLDS env var; default to proposal's k=5.
    import os
    k_folds = int(os.environ.get("K_FOLDS", "5"))
    print(f"  Using {k_folds}-fold stratified cross-validation for grid search.")

    _section(f"[2/6] Grid Search — Logistic Regression ({k_folds}-fold CV, 4 configs)")
    best_lr_cfg, best_lr_threshold, lr_grid_results = grid_search_logistic(
        prepared.X_train,
        prepared.y_train,
        pos_weight=pos_weight,
        neg_weight=neg_weight,
        k_folds=k_folds,
    )
    print(f"\n  Best config: {best_lr_cfg}  threshold={best_lr_threshold:.3f}")

    _section(f"[3/6] Grid Search — ANN ({k_folds}-fold CV, 4 configs)")
    best_ann_cfg, best_ann_threshold, ann_grid_results = grid_search_ann(
        prepared.X_train,
        prepared.y_train,
        input_dim=prepared.X_train.shape[1],
        pos_weight=pos_weight,
        neg_weight=neg_weight,
        k_folds=k_folds,
    )
    print(f"\n  Best config: {best_ann_cfg}  threshold={best_ann_threshold:.3f}")

    _section("[4/6] Final Model Training on Full Train Set")
    print("\n  Training Logistic Regression...")
    lr_model = LogisticRegressionScratch(pos_weight=pos_weight, neg_weight=neg_weight, **best_lr_cfg)
    lr_model.fit(prepared.X_train, prepared.y_train, verbose=True, early_stopping_patience=80)
    lr_metrics = evaluate_model(lr_model, prepared.X_test, prepared.y_test, threshold=best_lr_threshold)
    print(f"  Test -> acc={lr_metrics['accuracy']:.3f}, f1={lr_metrics['f1']:.3f}, auc={lr_metrics['auc']:.3f}")

    print("\n  Training ANN...")
    ann_model = ANNClassifierScratch(
        input_dim=prepared.X_train.shape[1],
        seed=42,
        pos_weight=pos_weight,
        neg_weight=neg_weight,
        **best_ann_cfg,
    )
    ann_model.fit(prepared.X_train, prepared.y_train, verbose=True, early_stopping_patience=50)
    ann_metrics = evaluate_model(ann_model, prepared.X_test, prepared.y_test, threshold=best_ann_threshold)
    print(f"  Test -> acc={ann_metrics['accuracy']:.3f}, f1={ann_metrics['f1']:.3f}, auc={ann_metrics['auc']:.3f}")

    composite_lr = composite_score(lr_metrics)
    composite_ann = composite_score(ann_metrics)
    best_model_name = "ann" if composite_ann > composite_lr else "logistic_regression"

    _section("[5/6] Temporal Split Evaluation")
    temporal_lr_metrics = {}
    temporal_ann_metrics = {}
    X_time_train, y_time_train, X_time_test, y_time_test = temporal_split_from_df(
        prepared.full_df, prepared.feature_names, split_date="2025-01-01"
    )
    if len(X_time_train) > 0 and len(X_time_test) > 0:
        print(f"  Split date: 2025-01-01  train={len(y_time_train)}, test={len(y_time_test)}")
        t_pos_count = np.sum(y_time_train == 1)
        t_neg_count = np.sum(y_time_train == 0)
        t_pos_weight = float(t_neg_count / (t_pos_count + 1e-12))

        print("\n  Training temporal Logistic Regression...")
        lr_temporal = LogisticRegressionScratch(pos_weight=t_pos_weight, neg_weight=1.0, **best_lr_cfg)
        lr_temporal.fit(X_time_train, y_time_train, verbose=True)
        temporal_lr_metrics = evaluate_model(
            lr_temporal, X_time_test, y_time_test, threshold=best_lr_threshold
        )
        print(
            f"  Temporal LR -> acc={temporal_lr_metrics['accuracy']:.3f}, "
            f"f1={temporal_lr_metrics['f1']:.3f}, auc={temporal_lr_metrics['auc']:.3f}"
        )

        print("\n  Training temporal ANN...")
        ann_temporal = ANNClassifierScratch(
            input_dim=X_time_train.shape[1],
            seed=42,
            pos_weight=t_pos_weight,
            neg_weight=1.0,
            **best_ann_cfg,
        )
        ann_temporal.fit(X_time_train, y_time_train, verbose=True)
        temporal_ann_metrics = evaluate_model(
            ann_temporal, X_time_test, y_time_test, threshold=best_ann_threshold
        )
        print(
            f"  Temporal ANN -> acc={temporal_ann_metrics['accuracy']:.3f}, "
            f"f1={temporal_ann_metrics['f1']:.3f}, auc={temporal_ann_metrics['auc']:.3f}"
        )
    else:
        print("  Not enough data for temporal split — skipping.")

    _section("[6/6] Saving Artifacts")
    np.savez(
        artifacts_dir / "logistic_regression_weights.npz",
        w=lr_model.w,
        b=np.array([lr_model.b]),
    )
    np.savez(
        artifacts_dir / "ann_weights.npz",
        W1=ann_model.W1,
        b1=ann_model.b1,
        W2=ann_model.W2,
        b2=ann_model.b2,
    )
    save_scaler(artifacts_dir / "scaler_config.json", prepared.scaler, prepared.feature_names)
    print("  Saved model weights and scaler.")

    lr_feature_importance = np.abs(lr_model.w)
    lr_feature_importance = lr_feature_importance / (np.sum(lr_feature_importance) + 1e-12)

    metrics_payload = {
        "logistic_regression": lr_metrics,
        "ann": ann_metrics,
        "composite": {"logistic_regression": composite_lr, "ann": composite_ann},
        "best_model": best_model_name,
        "feature_names": prepared.feature_names,
        "logistic_feature_importance": lr_feature_importance.tolist(),
        "class_weights": {"pos_weight": pos_weight, "neg_weight": neg_weight},
        "decision_thresholds": {
            "logistic_regression": best_lr_threshold,
            "ann": best_ann_threshold,
        },
        "grid_search": {
            "logistic_regression": {"best_params": best_lr_cfg, "results": lr_grid_results},
            "ann": {"best_params": best_ann_cfg, "results": ann_grid_results},
        },
        "temporal_split": {
            "split_date": "2025-01-01",
            "logistic_regression": temporal_lr_metrics,
            "ann": temporal_ann_metrics,
        },
    }

    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print("  Saved metrics.json.")

    print(f"\n{'='*55}")
    print("  Training complete!")
    print(f"  Logistic Regression composite score : {composite_lr:.4f}")
    print(f"  ANN composite score                 : {composite_ann:.4f}")
    print(f"  Best model                          : {best_model_name}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

