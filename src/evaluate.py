"""Model heads, temperature calibration, Youden threshold, metrics, bootstrap CIs."""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

HEADS = {"RF": "Stats+PCA+RF", "GBDT": "Stats+PCA+GBDT", "ADA": "Stats+PCA+ADA"}


def make_pipeline(head, cfg, seed):
    m = cfg["model"]
    if head == "RF":
        clf = RandomForestClassifier(n_estimators=m["rf_n_estimators"], class_weight=m["rf_class_weight"],
                                     random_state=seed, n_jobs=-1)
    elif head == "GBDT":
        clf = GradientBoostingClassifier(random_state=seed)
    elif head == "ADA":
        clf = AdaBoostClassifier(random_state=seed)
    else:
        raise ValueError(head)
    return Pipeline([("scaler", StandardScaler()),
                     ("pca", PCA(n_components=m["pca_variance"], svd_solver="full")),
                     ("clf", clf)])


def fit_pipeline(pipe, X, y, head, boost_balanced=True):
    """RF uses class_weight=balanced; GBDT/ADA get equivalent balanced sample weights [ours].
    boost_balanced=False: GBDT/ADA fit without sample weights (sklearn defaults)."""
    if head == "RF" or not boost_balanced:
        return pipe.fit(X, y)
    return pipe.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))


def val_split(groups, y, cfg, seed):
    """Person-disjoint validation part of the training data: one fold of StratifiedGroupKFold(round(1/frac))."""
    k = int(round(1 / cfg["calibration"]["val_fraction"]))
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(sgk.split(np.zeros(len(y)), y, groups))
    return fit_idx, val_idx


#calibration
P_CLIP = 1e-3


def logit(p):
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return np.log(p / (1 - p))


def fit_temperature(p_val, y_val):
    """Scalar T > 0 minimising validation NLL of sigmoid(logit(p) / T)."""
    z = logit(p_val)

    def nll(logT):
        q = 1 / (1 + np.exp(-z / np.exp(logT)))
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return -np.mean(y_val * np.log(q) + (1 - y_val) * np.log(1 - q))
    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20)), method="bounded")
    return float(np.exp(r.x))


def apply_temperature(p, T):
    return 1 / (1 + np.exp(-logit(p) / T))


def youden_threshold(p, y):
    """Threshold maximising TPR - FPR; ties -> highest threshold (first in sklearn's order). Positive if p >= thr."""
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(min(thr[i], 1.0)), float(j[i])


#metrics
def ece(p, y, n_bins=10):
    """Expected calibration error, equal-width bins on P(dementia)."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1], right=False), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def point_metrics(p, y, thr, n_bins=10):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "n_pos": int(y.sum()), "AUROC": float(roc_auc_score(y, p)),
            "AP": float(average_precision_score(y, p)), "F1": float(f1_score(y, yhat, zero_division=0)),
            "Acc": float((yhat == y).mean()), "recall": float(tp / (tp + fn)) if tp + fn else np.nan,
            "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
            "precision": float(tp / (tp + fp)) if tp + fp else np.nan, "thr": float(thr),
            "ECE": ece(p, y, n_bins), "Brier": float(brier_score_loss(y, p)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def bootstrap_ci(p, y, n, seed, groups=None, alpha=0.05):
    """Stratified bootstrap 95% CI for AUROC and AP (paper: 'stratified bootstrap').
    groups=None: resample clips within class. groups given: resample PERSONS within class (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    if groups is None:
        units = [np.where(y == c)[0][:, None] for c in (0, 1)]
        units = [[u for u in uc] for uc in units]
    else:
        units = []
        for c in (0, 1):
            gs = np.unique(groups[y == c])
            units.append([np.where((groups == g) & (y == c))[0] for g in gs])
    au, ap = [], []
    for _ in range(n):
        idx = np.concatenate([np.concatenate([uc[i] for i in rng.integers(0, len(uc), len(uc))]) for uc in units])
        au.append(roc_auc_score(y[idx], p[idx]))
        ap.append(average_precision_score(y[idx], p[idx]))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"AUROC_CI": [float(v) for v in np.percentile(au, q)], "AP_CI": [float(v) for v in np.percentile(ap, q)]}
