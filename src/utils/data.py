import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "tempo",
    "energy",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_rolloff",
    "mfcc_1",
    "mfcc_2",
    "chroma_mean",
    "chroma_std",
    "primary_artist_popularity",
    "max_artist_popularity",
    "monthly_listeners",
]

# Chart-derived inputs for leakage / upper-bound experiments only (not production).
CHART_CONTEXT_FEATURES = ["streams", "weeks_on_chart"]
FEATURE_COLUMNS_BENCHMARK = FEATURE_COLUMNS + CHART_CONTEXT_FEATURES


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list
    scaler: dict
    target_name: str
    full_df: pd.DataFrame


def _min_max_scale_fit(X: np.ndarray):
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    span = np.where((x_max - x_min) == 0.0, 1.0, (x_max - x_min))
    return x_min, x_max, span


def min_max_scale_apply(X: np.ndarray, x_min: np.ndarray, span: np.ndarray):
    return (X - x_min) / span


def iqr_cap_outliers(X: np.ndarray, k: float = 1.5):
    """Cap outliers per-feature using Tukey's IQR rule (proposal §4.1.1)."""
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return np.clip(X, lower, upper)


def iqr_cap_outliers_train_apply(X_train: np.ndarray, *others: np.ndarray, k: float = 1.5):
    """Fit Tukey IQR bounds on ``X_train`` only, then clip ``X_train`` and each array in ``others``."""
    q1 = np.percentile(X_train, 25, axis=0)
    q3 = np.percentile(X_train, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    clipped_train = np.clip(X_train, lower, upper)
    clipped_others = tuple(np.clip(X, lower, upper) for X in others)
    return (clipped_train,) + clipped_others


def artist_stratified_kfold_indices(artists: np.ndarray, k: int = 5, seed: int = 42):
    """K-fold splits by ``primary_artist``: test folds contain only artists not seen in train.

    ``artists`` must have one entry per row (same length as the feature matrix / dataframe).
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(artists)
    rng.shuffle(unique)
    if len(unique) < k:
        raise ValueError(f"Need at least k={k} distinct artists, got {len(unique)}")
    groups = np.array_split(unique, k)
    folds = []
    for i in range(k):
        test_artists = groups[i]
        mask_test = np.isin(artists, test_artists)
        test_idx = np.where(mask_test)[0]
        train_idx = np.where(~mask_test)[0]
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        folds.append((train_idx, test_idx))
    return folds


def prepare_xy_artist_fold(
    train_fit_df: pd.DataFrame,
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    cap_outliers: bool = True,
):
    """IQR + min-max fit on ``train_fit_df`` features only; applied to val, full train concat, and test."""
    X_tf = train_fit_df[feature_cols].to_numpy(dtype=float)
    X_tv = train_val_df[feature_cols].to_numpy(dtype=float)
    X_tr = np.vstack([X_tf, X_tv])
    X_te = test_df[feature_cols].to_numpy(dtype=float)
    y_tf = train_fit_df["hit"].to_numpy(dtype=float)
    y_tv = train_val_df["hit"].to_numpy(dtype=float)
    y_train = np.concatenate([y_tf, y_tv])
    y_test = test_df["hit"].to_numpy(dtype=float)

    if cap_outliers:
        X_tf_c, X_tv_c, X_te_c = iqr_cap_outliers_train_apply(X_tf, X_tv, X_te, k=1.5)
        X_tr_c = np.vstack([X_tf_c, X_tv_c])
    else:
        X_tf_c, X_tv_c, X_tr_c, X_te_c = X_tf, X_tv, X_tr, X_te

    x_min, _, span = _min_max_scale_fit(X_tf_c)
    X_train = min_max_scale_apply(X_tr_c, x_min, span)
    X_val = min_max_scale_apply(X_tv_c, x_min, span)
    X_test = min_max_scale_apply(X_te_c, x_min, span)
    return X_train, y_train, X_val, y_tv, X_test, y_test


def random_oversample(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Random oversampling of the minority class (proposal §5 Risk: Class Imbalance).

    Returned in addition to (not replacing) class-weighted loss; users can choose.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y
    if len(pos_idx) < len(neg_idx):
        minority, majority = pos_idx, neg_idx
    else:
        minority, majority = neg_idx, pos_idx
    resampled = rng.choice(minority, size=len(majority), replace=True)
    combined = np.concatenate([majority, resampled])
    rng.shuffle(combined)
    return X[combined], y[combined]


def stratified_kfold_indices(y: np.ndarray, k: int = 5, seed: int = 42):
    """Generate stratified k-fold index splits (proposal §4.1.3)."""
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    pos_folds = np.array_split(idx_pos, k)
    neg_folds = np.array_split(idx_neg, k)
    folds = []
    for i in range(k):
        val_idx = np.concatenate([pos_folds[i], neg_folds[i]])
        train_idx = np.concatenate(
            [f for j, f in enumerate(pos_folds) if j != i] +
            [f for j, f in enumerate(neg_folds) if j != i]
        )
        rng.shuffle(val_idx)
        rng.shuffle(train_idx)
        folds.append((train_idx, val_idx))
    return folds


def stratified_fit_val_indices(y: np.ndarray, val_ratio: float = 0.15, seed: int = 42):
    """Row indices for a stratified holdout of ``y`` into (fit, val); indices in 0..len(y)-1."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n_val_pos = max(1, int(len(idx_pos) * val_ratio))
    n_val_neg = max(1, int(len(idx_neg) * val_ratio))
    val_idx = np.concatenate([idx_pos[:n_val_pos], idx_neg[:n_val_neg]])
    fit_idx = np.concatenate([idx_pos[n_val_pos:], idx_neg[n_val_neg:]])
    rng.shuffle(val_idx)
    rng.shuffle(fit_idx)
    return fit_idx, val_idx


def _stratified_split(X: np.ndarray, y: np.ndarray, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    n_test_pos = int(len(idx_pos) * test_ratio)
    n_test_neg = int(len(idx_neg) * test_ratio)

    test_idx = np.concatenate([idx_pos[:n_test_pos], idx_neg[:n_test_neg]])
    train_idx = np.concatenate([idx_pos[n_test_pos:], idx_neg[n_test_neg:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def load_and_prepare_data(csv_path: str, test_ratio=0.2, seed=42, cap_outliers: bool = True) -> PreparedData:
    df = pd.read_csv(csv_path)
    df["hit"] = (df["rank"] <= 10).astype(int)
    df = df.dropna(subset=FEATURE_COLUMNS + ["hit"]).copy()

    X_raw = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["hit"].to_numpy(dtype=float)

    # IQR outlier capping per proposal §4.1.1 to keep gradient descent stable
    # on features with heavy tails (e.g. monthly_listeners).
    if cap_outliers:
        X_raw = iqr_cap_outliers(X_raw, k=1.5)

    x_min, x_max, span = _min_max_scale_fit(X_raw)
    X = min_max_scale_apply(X_raw, x_min, span)
    X_train, y_train, X_test, y_test = _stratified_split(X, y, test_ratio=test_ratio, seed=seed)

    scaler = {"min": x_min.tolist(), "max": x_max.tolist(), "span": span.tolist()}

    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=FEATURE_COLUMNS,
        scaler=scaler,
        target_name="hit",
        full_df=df,
    )


def load_and_prepare_benchmark_data(
    csv_path: str, test_ratio: float = 0.2, seed: int = 42, cap_outliers: bool = True
) -> PreparedData:
    """Same pipeline as production, plus ``streams`` and ``weeks_on_chart`` (chart leakage risk)."""
    df = pd.read_csv(csv_path)
    df["hit"] = (df["rank"] <= 10).astype(int)
    cols = FEATURE_COLUMNS_BENCHMARK
    df = df.dropna(subset=cols + ["hit"]).copy()

    X_raw = df[cols].to_numpy(dtype=float)
    y = df["hit"].to_numpy(dtype=float)

    if cap_outliers:
        X_raw = iqr_cap_outliers(X_raw, k=1.5)

    x_min, x_max, span = _min_max_scale_fit(X_raw)
    X = min_max_scale_apply(X_raw, x_min, span)
    X_train, y_train, X_test, y_test = _stratified_split(X, y, test_ratio=test_ratio, seed=seed)

    scaler = {"min": x_min.tolist(), "max": x_max.tolist(), "span": span.tolist()}

    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=list(cols),
        scaler=scaler,
        target_name="hit",
        full_df=df,
    )


def save_scaler(path: str, scaler: dict, feature_names: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names, "scaler": scaler}, f, indent=2)


def stratified_train_val_split(X: np.ndarray, y: np.ndarray, val_ratio=0.2, seed=42):
    return _stratified_split(X, y, test_ratio=val_ratio, seed=seed)


def temporal_split_from_df(df: pd.DataFrame, feature_names: list, split_date: str):
    cutoff = pd.to_datetime(split_date)
    dates = pd.to_datetime(df["date"])
    train_df = df[dates < cutoff].copy()
    test_df = df[dates >= cutoff].copy()

    X_train_raw = train_df[feature_names].to_numpy(dtype=float)
    y_train = train_df["hit"].to_numpy(dtype=float)
    X_test_raw = test_df[feature_names].to_numpy(dtype=float)
    y_test = test_df["hit"].to_numpy(dtype=float)

    x_min, _, span = _min_max_scale_fit(X_train_raw)
    X_train = min_max_scale_apply(X_train_raw, x_min, span)
    X_test = min_max_scale_apply(X_test_raw, x_min, span)
    return X_train, y_train, X_test, y_test

