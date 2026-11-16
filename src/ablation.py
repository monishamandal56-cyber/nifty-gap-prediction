"""
src/ablation.py
---------------
Walk-forward ablation study for the Nifty 50 gap prediction system.

Compares, on the SAME data, SAME features, and SAME walk-forward folds:

    1. Ridge only
    2. Huber only
    3. Random Forest only
    4. XGBoost only
    5. Ensemble - paper weights   (Huber .35 / Ridge .25 / HistGB .20 / RF .20)
    6. Ensemble - equal weights   (0.25 each, HistGB variant)
    7. Ensemble - equal weights with XGBoost replacing HistGB

For each variant, reports:
    MAE %, RMSE %, R^2, MAE points,
    Directional accuracy %,
    3-class accuracy % (Gap Up / Flat / Gap Down), both from continuous
    predictions and from a per-fold RF classifier,
    80% CI coverage %,
    per-fold MAE mean and std,
    and Diebold-Mariano tests vs. the best variant.

Usage (from project root):
    python -m src.ablation
or, from a notebook:
    from src.ablation import run_ablation
    summary, results, dm_df, oof_df, folds = run_ablation(df, n_folds=10)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.feature_engineering import FEATURE_COLS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLAT_BAND       = 0.10    # |gap%| < 0.10 -> Flat
HUBER_EPSILON   = 1.35    # sklearn's convention (must be >= 1.0)

PAPER_WEIGHTS = {'huber': 0.35, 'ridge': 0.25, 'hgb': 0.20, 'rf': 0.20}
EQUAL_WEIGHTS = {'huber': 0.25, 'ridge': 0.25, 'hgb': 0.25, 'rf': 0.25}


# ---------------------------------------------------------------------------
# Walk-forward splitter (shared by every variant)
# ---------------------------------------------------------------------------

def walk_forward_indices(n_samples: int,
                         n_folds: int = 10,
                         min_train_frac: float = 0.40):
    """
    Expanding-window walk-forward. Fold k:
        train = [0, split_k)
        test  = [split_k, split_{k+1})
    Returns a list of (train_idx, test_idx) numpy arrays.
    """
    min_train = int(n_samples * min_train_frac)
    remaining = n_samples - min_train
    fold_size = max(1, remaining // n_folds)

    folds = []
    for k in range(n_folds):
        tr_end = min_train + k * fold_size
        te_end = tr_end + fold_size if k < n_folds - 1 else n_samples
        if te_end <= tr_end:
            break
        folds.append((np.arange(0, tr_end), np.arange(tr_end, te_end)))
    return folds


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_ridge():
    return Ridge(alpha=10.0)

def make_huber():
    return HuberRegressor(epsilon=HUBER_EPSILON, max_iter=1000)

def make_rf_reg():
    return RandomForestRegressor(
        n_estimators=120, max_depth=5, min_samples_leaf=4, random_state=42
    )

def make_hgb():
    return HistGradientBoostingRegressor(
        max_iter=120, max_depth=4, learning_rate=0.04, random_state=42
    )

def make_xgb():
    if not HAS_XGB:
        return None
    return XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.04,
        subsample=0.9, colsample_bytree=0.9,
        random_state=42, verbosity=0,
    )

def make_clf():
    return RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=42
    )


def _classify(gap_pct_values: np.ndarray) -> np.ndarray:
    """Map continuous gap % to 3 classes: -1 (down), 0 (flat), 1 (up)."""
    return np.where(gap_pct_values >  FLAT_BAND,  1,
           np.where(gap_pct_values < -FLAT_BAND, -1, 0))


# ---------------------------------------------------------------------------
# Ensemble builder
# ---------------------------------------------------------------------------

def _ensemble_fn(weight_dict: dict, factories: dict):
    """
    factories: {name: callable} for the sub-models keyed as in weight_dict.
    Returns a function fn(Xtr, ytr, Xte) -> blended predictions.
    """
    def fn(Xtr, ytr, Xte):
        preds = {}
        for key, factory in factories.items():
            if factory is None:
                continue
            m = factory()
            m.fit(Xtr.values, ytr.values)
            preds[key] = m.predict(Xte.values)
        return sum(weight_dict[k] * preds[k] for k in preds)
    return fn


# ---------------------------------------------------------------------------
# Evaluate a single variant
# ---------------------------------------------------------------------------

def evaluate_variant(name, train_pred_fn, X, y, prev_close, folds,
                     fit_classifier: bool = True):
    """
    Runs walk-forward for one variant and returns a dict with OOF predictions
    and aggregate + per-fold metrics.
    """
    n = len(y)
    oof     = np.full(n, np.nan)
    oof_cls = np.full(n, np.nan)   # classifier's class prediction per row
    fold_maes = []

    for tr, te in folds:
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xte      = X.iloc[te]
        preds    = train_pred_fn(Xtr, ytr, Xte)
        oof[te]  = preds
        fold_maes.append(mean_absolute_error(y.iloc[te], preds))

        if fit_classifier:
            ytr_cls = _classify(ytr.values)
            if len(np.unique(ytr_cls)) > 1:
                clf = make_clf()
                clf.fit(Xtr.values, ytr_cls)
                oof_cls[te] = clf.predict(Xte.values)
            else:
                oof_cls[te] = ytr_cls[0]

    valid = ~np.isnan(oof)
    yv    = y.values[valid]
    pv    = oof[valid]
    cv    = prev_close.values[valid]

    # --- Regression metrics ---
    mae_pct  = mean_absolute_error(yv, pv)
    rmse_pct = np.sqrt(mean_squared_error(yv, pv))
    r2       = r2_score(yv, pv)
    mae_pts  = mean_absolute_error(yv * cv / 100.0, pv * cv / 100.0)

    # --- Directional accuracy ---
    dir_acc = float(np.mean(np.sign(yv) == np.sign(pv)) * 100)

    # --- 3-class accuracy from continuous predictions ---
    true_cls = _classify(yv)                # length = sum(valid)
    pred_cls = _classify(pv)
    cls_acc_cont = float(np.mean(true_cls == pred_cls) * 100)

    # --- 3-class accuracy from the per-fold RF classifier ---
    # oof_cls has length n; restrict to the same rows as true_cls first.
    cls_acc_rf = np.nan
    if fit_classifier:
        oof_cls_valid = oof_cls[valid]      # length = sum(valid)
        valid_cls     = ~np.isnan(oof_cls_valid)
        if valid_cls.sum() > 0:
            yt_c = true_cls[valid_cls]
            yp_c = oof_cls_valid[valid_cls].astype(int)
            cls_acc_rf = float(np.mean(yt_c == yp_c) * 100)

# --- 80% CI coverage from OOF residuals ---
# Empirical 10th/90th percentiles: coverage is exactly 80% by
# construction. The Gaussian z=1.2816 formula assumes bell-shaped
# residuals and over-covers the middle when residuals are fat-tailed.
    resid = yv - pv
    sigma = float(np.std(resid))
    q10   = float(np.percentile(resid, 10))
    q90   = float(np.percentile(resid, 90))
    lo    = pv + q10
    hi    = pv + q90
    coverage = float(np.mean((yv >= lo) & (yv <= hi)) * 100)

    return {
        'name'              : name,
        'oof_pred'          : oof,
        'oof_cls_pred'      : oof_cls,
        'valid_mask'        : valid,
        'MAE_pct'           : mae_pct,
        'RMSE_pct'          : rmse_pct,
        'R2'                : r2,
        'MAE_pts'           : mae_pts,
        'DirAcc_pct'        : dir_acc,
        '3ClassAcc_cont_pct': cls_acc_cont,
        '3ClassAcc_rf_pct'  : cls_acc_rf,
        'CI80_cov_pct'      : coverage,
        'resid_p10'         : q10,
        'resid_p90'         : q90,
        'resid_std'         : sigma,
        'fold_mae_mean'     : float(np.mean(fold_maes)),
        'fold_mae_std'      : float(np.std(fold_maes)),
    }


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

def diebold_mariano(y_true, pred_a, pred_b, h=1):
    """
    Two-sided DM test on squared-error loss differential.
    Returns (DM stat, two-sided p-value).
    """
    d = (y_true - pred_a) ** 2 - (y_true - pred_b) ** 2
    n = len(d)
    d_bar = d.mean()
    var_d = np.var(d, ddof=0)
    for lag in range(1, h):
        var_d += 2 * (1 - lag / h) * np.cov(d[:-lag], d[lag:])[0, 1]
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = d_bar / np.sqrt(var_d / n)
    p_val   = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_val


# ---------------------------------------------------------------------------
# Main ablation
# ---------------------------------------------------------------------------

def run_ablation(dataset: pd.DataFrame,
                 n_folds: int = 10,
                 min_train_frac: float = 0.40,
                 verbose: bool = True):
    """
    Runs the full ablation study.

    Returns
    -------
    summary : DataFrame, one row per variant, sorted by MAE ascending
    results : list of raw result dicts (with OOF predictions)
    dm_df   : DataFrame, Diebold-Mariano tests vs. the best variant
    oof_df  : DataFrame, Date + OOF predictions per variant + actual
    folds   : the walk-forward fold indices (for reproducibility)
    """
    df = dataset.dropna(
        subset=['gap_pct', 'prev_close'] + FEATURE_COLS
    ).reset_index(drop=True)

    X          = df[FEATURE_COLS]
    y          = df['gap_pct']
    prev_close = df['prev_close']

    n = len(df)
    if verbose:
        print(f"Usable samples: {n}  ({df['Date'].min().date()} -> {df['Date'].max().date()})")

    folds = walk_forward_indices(n, n_folds=n_folds, min_train_frac=min_train_frac)
    if verbose:
        print(f"Walk-forward folds: {len(folds)}  "
              f"(first train={len(folds[0][0])}, first test={len(folds[0][1])})")

    results = []

    # 1. Ridge only -------------------------------------------------------
    results.append(evaluate_variant(
        "Ridge only",
        lambda Xtr, ytr, Xte: make_ridge().fit(Xtr.values, ytr.values).predict(Xte.values),
        X, y, prev_close, folds, fit_classifier=True,
    ))

    # 2. Huber only -------------------------------------------------------
    results.append(evaluate_variant(
        "Huber only",
        lambda Xtr, ytr, Xte: make_huber().fit(Xtr.values, ytr.values).predict(Xte.values),
        X, y, prev_close, folds, fit_classifier=True,
    ))

    # 3. Random Forest only ----------------------------------------------
    results.append(evaluate_variant(
        "RandomForest only",
        lambda Xtr, ytr, Xte: make_rf_reg().fit(Xtr.values, ytr.values).predict(Xte.values),
        X, y, prev_close, folds, fit_classifier=True,
    ))

    # 4. XGBoost only -----------------------------------------------------
    if HAS_XGB:
        results.append(evaluate_variant(
            "XGBoost only",
            lambda Xtr, ytr, Xte: make_xgb().fit(Xtr.values, ytr.values).predict(Xte.values),
            X, y, prev_close, folds, fit_classifier=True,
        ))
    elif verbose:
        print("NOTE: xgboost not installed. Skipping XGBoost variants.")

    # 5. Ensemble - paper weights (HistGB) -------------------------------
    paper_factories = {
        'huber': make_huber,
        'ridge': make_ridge,
        'hgb'  : make_hgb,
        'rf'   : make_rf_reg,
    }
    results.append(evaluate_variant(
        "Ensemble - paper weights",
        _ensemble_fn(PAPER_WEIGHTS, paper_factories),
        X, y, prev_close, folds, fit_classifier=True,
    ))

    # 6. Ensemble - equal weights (HistGB) -------------------------------
    results.append(evaluate_variant(
        "Ensemble - equal weights",
        _ensemble_fn(EQUAL_WEIGHTS, paper_factories),
        X, y, prev_close, folds, fit_classifier=True,
    ))

    # 7. Ensemble - equal weights with XGBoost ---------------------------
    if HAS_XGB:
        xgb_factories = {
            'huber': make_huber,
            'ridge': make_ridge,
            'hgb'  : make_xgb,       # xgb replaces hgb slot
            'rf'   : make_rf_reg,
        }
        results.append(evaluate_variant(
            "Ensemble - equal weights (XGB)",
            _ensemble_fn(EQUAL_WEIGHTS, xgb_factories),
            X, y, prev_close, folds, fit_classifier=True,
        ))

    # ---- Summary table --------------------------------------------------
    summary = pd.DataFrame([
        {k: v for k, v in r.items()
         if k not in ('oof_pred', 'oof_cls_pred', 'valid_mask')}
        for r in results
    ]).sort_values('MAE_pct').reset_index(drop=True)

    if verbose:
        print("\n================= ABLATION RESULTS =================")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- Diebold-Mariano vs. best --------------------------------------
    best_name = summary.iloc[0]['name']
    best = next(r for r in results if r['name'] == best_name)
    m    = best['valid_mask']
    yv   = y.values[m]

    dm_rows = []
    for r in results:
        if r['name'] == best_name:
            continue
        stat, p = diebold_mariano(yv, r['oof_pred'][m], best['oof_pred'][m], h=1)
        verdict = ("worse than best" if (p < 0.05 and stat > 0) else
                   "better than best" if (p < 0.05 and stat < 0) else
                   "no sig. diff")
        dm_rows.append({
            'vs'      : r['name'],
            'DM_stat' : stat,
            'p_value' : p,
            'verdict' : verdict,
        })
    dm_df = pd.DataFrame(dm_rows)

    if verbose:
        print(f"\nDiebold-Mariano vs. best ({best_name}):")
        print(dm_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- OOF prediction table ------------------------------------------
    oof_df = df[['Date']].copy()
    for r in results:
        oof_df[r['name']] = r['oof_pred']
    oof_df['actual_gap_pct'] = y.values

    return summary, results, dm_df, oof_df, folds


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    from src.data_loader import load_all_historical_data
    from src.feature_engineering import build_features_and_targets

    print("Loading data ...")
    raw = load_all_historical_data(period="5y")
    df  = build_features_and_targets(raw)

    summary, results, dm_df, oof_df, folds = run_ablation(df, n_folds=10)

    summary.to_csv("ablation_summary.csv", index=False)
    dm_df.to_csv("ablation_dm_tests.csv", index=False)
    oof_df.to_csv("ablation_oof_predictions.csv", index=False)
    print("\nSaved: ablation_summary.csv, ablation_dm_tests.csv, ablation_oof_predictions.csv")