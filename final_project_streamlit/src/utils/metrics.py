import numpy as np


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(precision), float(recall), float(f1)


def roc_curve_points(y_true, y_prob, n_thresholds=200):
    thresholds = np.linspace(1.0, 0.0, n_thresholds)
    tpr_list = []
    fpr_list = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return np.array(fpr_list), np.array(tpr_list), thresholds


def auc_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve_points(y_true, y_prob)
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def best_threshold_by_f1(y_true, y_prob, n_thresholds=200):
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        _, _, f1 = precision_recall_f1(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)

